"""
Vector storage and retrieval layer.

Responsibilities:
    - Generate embeddings
    - Store document chunks in Qdrant
    - Perform semantic retrieval
    - Perform multi-query retrieval
    - Merge duplicate evidence
"""

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.config import settings


# -------------------------------------------------------------------
# Lazy-loaded resources
# -------------------------------------------------------------------

_embedder = None
_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(settings.EMBED_MODEL)

    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert text into normalized embedding vectors.
    """

    model = get_embedder()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype="float32")


def get_client() -> QdrantClient:
    global _client

    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    return _client


# -------------------------------------------------------------------
# Document chunk representation
# -------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source: str


# -------------------------------------------------------------------
# Vector store
# -------------------------------------------------------------------

class VectorStore:

    def __init__(self, collection: str = None):

        self.collection = (
            collection
            or settings.QDRANT_COLLECTION
        )

        self.client = get_client()

    # ---------------------------------------------------------------
    # Indexing
    # ---------------------------------------------------------------

    def build(self, chunks: List[Chunk]):
        """
        Build the vector collection from document chunks.
        """

        if not chunks:
            return

        vectors = embed_texts(
            [chunk.text for chunk in chunks]
        )

        dimension = vectors.shape[1]

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

        points = []

        for i, chunk in enumerate(chunks):

            points.append(
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload=asdict(chunk),
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    # ---------------------------------------------------------------
    # Single-query retrieval
    # ---------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = None,
    ):

        k = k or settings.TOP_K

        query_vector = embed_texts(
            [query]
        )[0].tolist()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
        )

        return [
            {
                "chunk": Chunk(
                    id=result.payload["id"],
                    text=result.payload["text"],
                    source=result.payload["source"],
                ),
                "score": float(result.score),
            }
            for result in results
        ]

    # ---------------------------------------------------------------
    # Multi-query retrieval
    # ---------------------------------------------------------------

    def multi_search(
        self,
        queries: List[str],
        k: int = None,
    ):
        """
        Retrieve evidence for multiple search queries.

        Each query performs an independent vector search.

        Results are merged by chunk ID so the same document
        chunk is not passed downstream multiple times.

        If a chunk appears for multiple queries, its strongest
        similarity score is retained.
        """

        if not queries:
            return []

        k = k or settings.TOP_K

        merged = {}

        for query in queries:

            query = query.strip()

            if not query:
                continue

            results = self.search(
                query,
                k=k,
            )

            for result in results:

                chunk = result["chunk"]

                chunk_id = chunk.id

                existing = merged.get(chunk_id)

                if (
                    existing is None
                    or result["score"]
                    > existing["score"]
                ):
                    merged[chunk_id] = result

        return sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )


# -------------------------------------------------------------------
# Singleton store
# -------------------------------------------------------------------

_store = None


def get_store() -> VectorStore:

    global _store

    if _store is None:
        _store = VectorStore()

    return _store