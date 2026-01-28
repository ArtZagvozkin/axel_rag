#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION = "axel_chunks"
MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    qd = QdrantClient(url=QDRANT_HOST)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    while True:
        q = input("\nВопрос (или пусто для выхода): ").strip()
        if not q:
            break

        # E5: query prefix (важно)
        q_vec = model.encode(["query: " + q], normalize_embeddings=True)[0].tolist()

        # В некоторых версиях qdrant-client нет qd.search(), используем qd.query_points()
        res = qd.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=8,
            with_payload=True,
        )

        hits = res.points if hasattr(res, "points") else res

        for i, h in enumerate(hits, start=1):
            p = h.payload or {}
            chunk_id = p.get("chunk_id") or str(h.id)

            # score может называться score (обычно), но перестрахуемся
            score = getattr(h, "score", None)
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"

            print(f"\n#{i} score={score_s} chunk_id={chunk_id}")
            print(f"   title: {p.get('title')}")
            print(f"   url:   {p.get('source_url')}")
            print(f"   path:  {' / '.join(p.get('section_path') or [])}")

if __name__ == "__main__":
    main()
