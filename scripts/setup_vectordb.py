"""Scaffold script to create Qdrant collection and set index params.

This script is a minimal helper for local development. It expects Qdrant to be
reachable at QDRANT_URL environment variable.
"""
import os
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "agri_knowledge")


def ensure_collection():
    client = QdrantClient(url=QDRANT_URL)
    try:
        if not client.get_collection(COLLECTION_NAME):
            # Create a collection with 1024-dim float vectors by default.
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 1024, "distance": "Cosine"},
            )
        print(f"Collection '{COLLECTION_NAME}' is ready at {QDRANT_URL}")
    except Exception as e:
        print(f"Failed to ensure collection: {e}")


if __name__ == "__main__":
    ensure_collection()
