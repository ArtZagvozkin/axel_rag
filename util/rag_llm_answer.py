#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
import os
import time

import psycopg
from psycopg.rows import dict_row

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from dotenv import load_dotenv

from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")


# CONFIG

PG_DSN = "postgresql://postgres:postgres@localhost:5432/rag_corpus"
QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION = "axel_chunks"

EMBED_MODEL = "intfloat/multilingual-e5-small"

# Примеры моделей Groq (актуальные часто меняются):
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

TOP_K = 6
MAX_CONTEXT_CHARS = 10_000


# SQL
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


# RAG STEPS
def search_chunks(qd: QdrantClient, model: SentenceTransformer, question: str) -> List[Dict[str, Any]]:
    vec = model.encode(["query: " + question], normalize_embeddings=True)[0].tolist()

    res = qd.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=TOP_K,
        with_payload=True,
    )

    points = res.points if hasattr(res, "points") else res

    out: List[Dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        out.append({
            "chunk_id": payload.get("chunk_id"),
            "title": payload.get("title"),
            "url": payload.get("source_url"),
        })

    return [x for x in out if x.get("chunk_id")]


def load_chunks(pg: psycopg.Connection, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    with pg.cursor(row_factory=dict_row) as cur:
        cur.execute(SQL_FETCH, (chunk_ids,))
        return cur.fetchall()


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    total = 0

    for i, ch in enumerate(chunks, start=1):
        text = (ch["text"] or "").strip()
        block = f"""
Источник [{i}]
Заголовок: {ch['title']}
URL: {ch['source_url']}
Текст:
{text}
""".strip()

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# LLM (GROQ)
def _is_retryable_error(e: Exception) -> bool:
    """
    Groq SDK кидает разные исключения (BadRequestError / RateLimitError и т.п.).
    Не будем жёстко привязываться к типам — проверяем признаки временной ошибки.
    """
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    msg = str(e).lower()

    if status in (408, 409, 429, 500, 502, 503, 504):
        return True

    # иногда статус не прокидывается, но по тексту можно понять
    retry_markers = ("rate limit", "timeout", "temporarily", "overloaded", "try again", "service unavailable")
    if any(m in msg for m in retry_markers):
        return True

    return False


def _call_groq_with_retry(client: Groq, prompt: str, retries: int = 4) -> str:
    delay = 2.0
    last_err: Exception | None = None

    for _attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            last_err = e
            if _is_retryable_error(e):
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Groq API error: {e}") from e

    raise RuntimeError(f"Groq API failed after retries: {last_err}") from last_err


def ask_llm(client: Groq, question: str, context: str) -> str:
    prompt = f"""
Ты - интеллектуальный ассистент корпоративной базы знаний AxelNAC.

Отвечай строго на основе предоставленных источников.
Если информации недостаточно - прямо скажи об этом.

Требования:
- давай чёткий технический ответ
- используй ссылки вида [1], [2], ...
- не выдумывай факты

Контекст:
{context}

Вопрос пользователя:
{question}
""".strip()

    return _call_groq_with_retry(client, prompt)


# MAIN
def main() -> None:
    qd = QdrantClient(url=QDRANT_HOST)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    pg = psycopg.connect(PG_DSN)

    client = Groq(api_key=GROQ_API_KEY)

    while True:
        q = input("\nВопрос (или пусто для выхода): ").strip()
        if not q:
            break

        hits = search_chunks(qd, model, q)
        chunk_ids = [h["chunk_id"] for h in hits]

        chunks = load_chunks(pg, chunk_ids)
        context = build_context(chunks)

        answer = ask_llm(client, q, context)

        print("\n" + "=" * 80)
        print(answer)
        print("\nИсточники:")
        for i, ch in enumerate(chunks, start=1):
            print(f"[{i}] {ch['title']} — {ch['source_url']}")
        print("=" * 80)

    pg.close()


if __name__ == "__main__":
    main()
