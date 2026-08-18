"""
GroundedRAG vector storage layer.

Responsibilities:
    - Generate embeddings
    - Manage the Qdrant collection
    - Incrementally upsert document chunks
    - Perform semantic retrieval
    - Perform multi-query retrieval
    - Merge duplicate evidence

Architecture note:
    New ingestion code should use upsert_chunks().
    build() is retained for backwards compatibility and delegates
    to incremental indexing instead of recreating the collection.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import hashlib

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.config import settings


# ============================================================
# LAZY-LOADED RESOURCES
# ============================================================

_embedder: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def get_embedder() -> SentenceTransformer:
    """
    Lazily load the local embedding model.
    """

    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            settings.EMBED_MODEL
        )

    return _embedder


def embed_texts(
    texts: List[str],
) -> np.ndarray:
    """
    Convert text into normalized embedding vectors.

    Returns:
        float32 NumPy array of shape:
        (number_of_texts, embedding_dimension)
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype="float32",
        )

    model = get_embedder()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        vectors,
        dtype="float32",
    )


def get_client() -> QdrantClient:
    """
    Lazily initialize the Qdrant client.
    """

    global _client

    if _client is None:
        if not settings.QDRANT_URL:
            raise RuntimeError(
                "QDRANT_URL is not configured."
            )

        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )

    return _client


# ============================================================
# DOCUMENT CHUNK
# ============================================================

@dataclass
class Chunk:
    """
    Canonical chunk representation used by retrieval and indexing.
    """

    id: str
    text: str
    source: str


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:
    """
    Qdrant-backed vector store.

    Supports:
        - collection creation
        - incremental chunk upserts
        - document deletion
        - semantic search
        - multi-query search
    """

    def __init__(
        self,
        collection: Optional[str] = None,
    ):
        self.collection = (
            collection
            or settings.QDRANT_COLLECTION
        )

        self.client = get_client()

    # ========================================================
    # COLLECTION MANAGEMENT
    # ========================================================

    def collection_exists(self) -> bool:
        """
        Check whether the configured Qdrant collection exists.
        """

        collections = self.client.get_collections()

        return any(
            item.name == self.collection
            for item in collections.collections
        )

    def ensure_collection(
        self,
        dimension: int,
    ) -> None:
        """
        Create the collection if it does not exist.

        IMPORTANT:
            Never recreate an existing collection.
            Incremental ingestion must preserve existing documents.
        """

        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

    # ========================================================
    # DETERMINISTIC POINT ID
    # ========================================================

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        """
        Generate a deterministic Qdrant-compatible integer ID.

        Python's built-in hash() is intentionally NOT used because
        Python hash values can change between processes.
        """

        digest = hashlib.sha256(
            chunk_id.encode("utf-8")
        ).hexdigest()

        # Qdrant integer point IDs support uint64.
        return int(
            digest[:16],
            16,
        )

    # ========================================================
    # INCREMENTAL INDEXING
    # ========================================================

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        extra_payload: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,
    ) -> int:
        """
        Incrementally embed and upsert chunks into Qdrant.

        Existing chunks with the same deterministic ID are replaced.

        Args:
            chunks:
                Chunks to index.

            extra_payload:
                Optional metadata keyed by chunk ID.

                Example:
                    {
                        "doc1::0": {
                            "document_id": "doc1",
                            "version": 2,
                        }
                    }

        Returns:
            Number of chunks successfully upserted.
        """

        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = embed_texts(texts)

        if vectors.size == 0:
            return 0

        dimension = vectors.shape[1]

        self.ensure_collection(
            dimension
        )

        points: List[PointStruct] = []

        for index, chunk in enumerate(
            chunks
        ):
            payload = asdict(chunk)

            if extra_payload:
                metadata = extra_payload.get(
                    chunk.id,
                    {},
                )

                if metadata:
                    payload.update(
                        metadata
                    )

            points.append(
                PointStruct(
                    id=self._point_id(
                        chunk.id
                    ),
                    vector=vectors[index].tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

        return len(points)

    # ========================================================
    # DOCUMENT-SCOPED INDEXING
    # ========================================================

    def upsert_document(
        self,
        document_id: str,
        source: str,
        chunks: List[str],
        version: int = 1,
        content_hash: Optional[str] = None,
    ) -> int:
        """
        Convert raw chunk strings into Chunk objects and index them.

        This is the preferred API for the ingestion service.
        """

        chunk_objects: List[Chunk] = []

        metadata: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for index, text in enumerate(
            chunks
        ):
            chunk_id = (
                f"{document_id}::"
                f"v{version}::"
                f"{index}"
            )

            chunk_objects.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=source,
                )
            )

            metadata[chunk_id] = {
                "document_id": document_id,
                "version": version,
                "chunk_index": index,
            }

            if content_hash:
                metadata[chunk_id][
                    "content_hash"
                ] = content_hash

        return self.upsert_chunks(
            chunk_objects,
            extra_payload=metadata,
        )

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every chunk belonging to a document.

        Uses a payload filter instead of rebuilding the collection.
        """

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
        )

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search.
        """

        k = (
            k
            or settings.TOP_K
        )

        query = query.strip()

        if not query:
            return []

        query_vector = embed_texts(
            [query]
        )[0].tolist()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
        )

        output = []

        for result in results:
            payload = result.payload or {}

            chunk_id = payload.get(
                "id",
                str(result.id),
            )

            text = payload.get(
                "text",
                "",
            )

            source = payload.get(
                "source",
                "",
            )

            output.append(
                {
                    "chunk": Chunk(
                        id=str(
                            chunk_id
                        ),
                        text=str(
                            text
                        ),
                        source=str(
                            source
                        ),
                    ),
                    "score": float(
                        result.score
                    ),
                }
            )

        return output

    # ========================================================
    # MULTI-QUERY SEARCH
    # ========================================================

    def multi_search(
        self,
        queries: List[str],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute independent searches for multiple retrieval queries.

        Duplicate chunks are merged and the strongest score retained.
        """

        if not queries:
            return []

        k = (
            k
            or settings.TOP_K
        )

        merged: Dict[
            str,
            Dict[str, Any],
        ] = {}

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

                existing = merged.get(
                    chunk_id
                )

                if (
                    existing is None
                    or result["score"]
                    > existing["score"]
                ):
                    merged[
                        chunk_id
                    ] = result

        return sorted(
            merged.values(),
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

    # ========================================================
    # LEGACY BUILD API
    # ========================================================

    def build(
        self,
        chunks: List[Chunk],
    ) -> int:
        """
        Backwards-compatible indexing API.

        Older code calls build().
        New code should call upsert_chunks().

        IMPORTANT:
            This no longer recreates the Qdrant collection.
        It performs an incremental upsert instead.
        """

        return self.upsert_chunks(
            chunks
        )


# ============================================================
# SINGLETON
# ============================================================

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """
    Return the application-wide VectorStore instance.
    """

    global _store

    if _store is None:
        _store = VectorStore()

    return _store