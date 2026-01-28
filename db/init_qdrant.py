#!/usr/bin/env python3

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION = "axel_chunks"
VECTOR_SIZE = 384

def main():
    client = QdrantClient(url=QDRANT_HOST)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"[OK] collection exists: {COLLECTION}")
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[OK] created collection: {COLLECTION}")

if __name__ == "__main__":
    main()


# Test qdrant
# curl http://localhost:6333/
# curl http://localhost:6333/healthz
# curl http://localhost:6333/collections
# curl http://localhost:6333/axel_chunks
# curl http://localhost:6333/collections/axel_chunks

# source .venv/bin/activate
