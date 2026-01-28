from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    logger,
    PG_DSN,
    QDRANT_HOST,
    QDRANT_COLLECTION,
    EMBED_MODEL,
    RAG_TOP_K,
    RAG_MAX_CONTEXT_CHARS,
)

SQL_FETCH = """
SELECT
  c.chunk_id,
  c.text,
  a.title,
  a.source_url
FROM rag.chunks c
JOIN rag.articles a ON a.article_id = c.article_id
WHERE c.chunk_id = ANY(%s);
"""


class RagService:
    """
    Мини-RAG:
    - Qdrant: находим chunk_id
    - Postgres: вытаскиваем текст чанков
    - Собираем контекст строкой
    """
    def __init__(
        self,
        pg_dsn: str = PG_DSN,
        qdrant_url: str = QDRANT_HOST,
        collection: str = QDRANT_COLLECTION,
        embed_model: str = EMBED_MODEL,
        top_k: int = RAG_TOP_K,
        max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
    ):
        self._pg_dsn = pg_dsn
        self._collection = collection
        self._top_k = top_k
        self._max_context_chars = max_context_chars

        self._qd = QdrantClient(url=qdrant_url)
        self._model = SentenceTransformer(embed_model, device="cpu")

    async def build_context_for_question(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return ""

        loop = asyncio.get_running_loop()

        def _search() -> List[str]:
            vec = self._model.encode(["query: " + q], normalize_embeddings=True)[0].tolist()
            res = self._qd.query_points(
                collection_name=self._collection,
                query=vec,
                limit=self._top_k,
                with_payload=True,
            )
            points = res.points if hasattr(res, "points") else res

            chunk_ids: List[str] = []
            for p in points:
                payload = p.payload or {}
                cid = payload.get("chunk_id")
                if cid:
                    chunk_ids.append(cid)
            return chunk_ids

        chunk_ids = await loop.run_in_executor(None, _search)
        if not chunk_ids:
            return ""

        def _load() -> List[Dict[str, Any]]:
            with psycopg.connect(self._pg_dsn) as pg:
                with pg.cursor(row_factory=dict_row) as cur:
                    cur.execute(SQL_FETCH, (chunk_ids,))
                    return cur.fetchall()

        rows = await loop.run_in_executor(None, _load)
        if not rows:
            return ""

        parts: List[str] = []
        total = 0
        for i, ch in enumerate(rows, start=1):
            text = (ch["text"] or "").strip()
            block = (
                f"Источник [{i}]\n"
                f"Заголовок: {ch['title']}\n"
                f"URL: {ch['source_url']}\n"
                f"Текст:\n{text}"
            )
            if total + len(block) > self._max_context_chars:
                break
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)
