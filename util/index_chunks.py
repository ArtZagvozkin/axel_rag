#!/usr/bin/env python3

import hashlib
from typing import List, Dict, Any

import psycopg
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


load_dotenv()
PG_DSN = os.getenv("PG_DSN")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32

SQL = """
SELECT
  c.chunk_id,
  c.text,
  c.section_path,
  c.block_types,
  c.chunk_hash,
  a.article_id,
  a.source_url,
  a.title,
  a.revision,
  a.updated_at_src
FROM rag.chunks c
JOIN rag.articles a ON a.article_id = c.article_id
ORDER BY c.article_id, c.chunk_ord;
"""

# HELPERS
def qdrant_point_id_from_chunk_id(chunk_id: str) -> int:
    h = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def chunk_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],

        "article_id": row["article_id"],
        "source_url": row["source_url"],
        "title": row["title"],
        "revision": row["revision"],
        "updated_at_src": row["updated_at_src"],

        "section_path": row["section_path"],
        "block_types": row["block_types"],

        "chunk_hash": row["chunk_hash"],
    }


# MAIN
def upsert_batch(qd: QdrantClient, model: SentenceTransformer,
                 rows: List[Dict[str, Any]], texts: List[str]) -> int:
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    points: List[PointStruct] = []
    for row, vec in zip(rows, vectors):
        cid = row["chunk_id"]
        pid = qdrant_point_id_from_chunk_id(cid)

        points.append(
            PointStruct(
                id=pid,                     # uint64
                vector=vec.tolist(),
                payload=chunk_payload(row),
            )
        )

    qd.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def main() -> None:
    pg = psycopg.connect(PG_DSN)
    qd = QdrantClient(url=QDRANT_HOST)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    total = 0
    upserted = 0

    with pg, pg.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(SQL)

        batch_rows: List[Dict[str, Any]] = []
        batch_texts: List[str] = []

        for row in cur:
            text = (row["text"] or "").strip()
            if not text:
                continue

            batch_rows.append(row)
            batch_texts.append("passage: " + text)

            if len(batch_rows) >= BATCH_SIZE:
                upserted += upsert_batch(qd, model, batch_rows, batch_texts)
                total += len(batch_rows)
                print(f"[OK] processed={total}, upserted={upserted}")
                batch_rows, batch_texts = [], []

        if batch_rows:
            upserted += upsert_batch(qd, model, batch_rows, batch_texts)
            total += len(batch_rows)

    print(f"[DONE] processed={total}, upserted={upserted}")


if __name__ == "__main__":
    main()
