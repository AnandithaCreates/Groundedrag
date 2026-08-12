"""
Run this after adding your own .md/.txt files to data/sample_docs/,
and after setting QDRANT_URL / QDRANT_API_KEY in your .env
(create a free cluster at cloud.qdrant.io first).

    python scripts/build_index.py

Recreates the Qdrant collection from scratch every time -- fine for a
corpus that changes occasionally by hand, which is the case for a
project like this. A live-updating system would use incremental
upserts instead of drop-and-rebuild.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import load_and_chunk
from app.vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "sample_docs")

if __name__ == "__main__":
    print(f"Loading documents from {DATA_DIR} ...")
    chunks = load_and_chunk(DATA_DIR)
    print(f"Created {len(chunks)} chunks from the corpus.")

    store = VectorStore()
    print("Embedding chunks locally (downloads the MiniLM model on first run)...")
    print(f"Upserting into Qdrant collection '{store.collection}'...")
    store.build(chunks)
    print("Done. Chunks are now searchable in Qdrant Cloud.")
