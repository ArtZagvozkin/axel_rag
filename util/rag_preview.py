#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

PG_DSN = "postgresql://postgres:postgres@localhost:5432/rag_corpus"
QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION = "axel_chunks"
MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 8


SQL_FETCH = """
SELECT
  c.chunk_id,
  c.text,
  c.section_path,
  a.title,
  a.source_url
FROM rag.chunks c
JOIN rag.articles a ON a.article_id = c.article_id
WHERE c.chunk_id = ANY(%s);
"""


def qdrant_search(qd: QdrantClient, model: SentenceTransformer, query: str, limit: int) -> List[Dict[str, Any]]:
    q_vec = model.encode(["query: " + query], normalize_embeddings=True)[0].tolist()

    res = qd.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=limit,
        with_payload=True,
    )
    points = res.points if hasattr(res, "points") else res

    out: List[Dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        out.append({
            "score": getattr(p, "score", None),
            "chunk_id": payload.get("chunk_id"),
            "title": payload.get("title"),
            "source_url": payload.get("source_url"),
            "section_path": payload.get("section_path") or [],
        })
    # фильтр на всякий случай
    return [x for x in out if x.get("chunk_id")]


def fetch_chunks_from_pg(pg: psycopg.Connection, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}

    with pg.cursor(row_factory=dict_row) as cur:
        cur.execute(SQL_FETCH, (chunk_ids,))
        rows = cur.fetchall()

    # быстрый доступ по chunk_id
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_id[r["chunk_id"]] = r
    return by_id


def main() -> None:
    qd = QdrantClient(url=QDRANT_HOST)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    pg = psycopg.connect(PG_DSN)

    while True:
        q = input("\nВопрос (или пусто для выхода): ").strip()
        if not q:
            break

        hits = qdrant_search(qd, model, q, TOP_K)
        chunk_ids = [h["chunk_id"] for h in hits]

        by_id = fetch_chunks_from_pg(pg, chunk_ids)

        print("\n" + "=" * 100)
        print(f"QUERY: {q}")
        print("=" * 100)

        for i, h in enumerate(hits, start=1):
            cid = h["chunk_id"]
            row = by_id.get(cid)

            score = h.get("score")
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"

            title = (row or {}).get("title") or h.get("title")
            url = (row or {}).get("source_url") or h.get("source_url")
            section_path = (row or {}).get("section_path") or h.get("section_path") or []
            text = (row or {}).get("text") or ""

            print(f"\n[{i}] score={score_s} chunk_id={cid}")
            print(f"    title: {title}")
            print(f"    url:   {url}")
            if section_path:
                print(f"    path:  {' / '.join(section_path)}")
            print("-" * 100)
            print(text[:2000])  # чтобы не утонуть в выводе; можно убрать срез
            print("-" * 100)

        print("\nИсточники:")
        seen = set()
        n = 1
        for h in hits:
            cid = h["chunk_id"]
            row = by_id.get(cid) or {}
            url = row.get("source_url") or h.get("source_url")
            title = row.get("title") or h.get("title")
            key = (url, title)
            if not url or key in seen:
                continue
            seen.add(key)
            print(f"[{n}] {title} — {url}")
            n += 1

    pg.close()


if __name__ == "__main__":
    main()
