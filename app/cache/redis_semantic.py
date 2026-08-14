"""
Upstash-backed semantic cache for GroundedRAG.

Uses the same local embedding model as retrieval so that
semantically similar questions can reuse grounded responses.
"""

import hashlib
import json
import time

import numpy as np
from upstash_redis import Redis

from app.config import settings
from app.vector_store import embed_texts


redis = Redis.from_env()

CACHE_PREFIX = "groundedrag:cache:"
INDEX_KEY = "groundedrag:cache:index"


def _normalize(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _cache_id(query: str) -> str:
    normalized = _normalize(query)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def _cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    denominator = (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


class SemanticCache:

    def __init__(
        self,
        similarity_threshold: float | None = None,
        ttl: int | None = None,
    ):
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.CACHE_SIM_THRESHOLD
        )

        self.ttl = (
            ttl
            if ttl is not None
            else settings.CACHE_TTL_SECONDS
        )

    def ping(self) -> bool:
        return redis.ping() == "PONG"

    def get(self, query: str):
        """
        Find a semantically similar cached query.

        Returns:
            {
                "response": {...},
                "similarity": float,
                "cached_query": str
            }

        or None on cache miss.
        """

        normalized = _normalize(query)

        if not normalized:
            return None

        query_vector = embed_texts(
            [normalized]
        )[0]

        keys = redis.smembers(INDEX_KEY)

        if not keys:
            return None

        best_match = None
        best_similarity = 0.0

        for key in keys:
            raw = redis.get(key)

            if raw is None:
                continue

            try:
                # Upstash returns our JSON payload as a string.
                if isinstance(raw, str):
                    item = json.loads(raw)
                else:
                    item = raw

                cached_embedding = np.asarray(
                    item["embedding"],
                    dtype=np.float32,
                )

                similarity = _cosine_similarity(
                    query_vector,
                    cached_embedding,
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = item

            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        if (
            best_match is not None
            and best_similarity >= self.similarity_threshold
        ):
            response = best_match["response"]

            if isinstance(response, str):
                response = json.loads(response)

            return {
                "response": response,
                "similarity": best_similarity,
                "cached_query": best_match["query"],
            }

        return None

    def set(
        self,
        query: str,
        response: dict,
    ):
        """
        Store a response and its query embedding.
        """

        normalized = _normalize(query)

        if not normalized:
            return

        cache_id = _cache_id(normalized)
        key = f"{CACHE_PREFIX}{cache_id}"

        embedding = embed_texts(
            [normalized]
        )[0].tolist()

        payload = {
            "query": query,
            "embedding": embedding,
            "response": response,
            "created_at": time.time(),
        }

        # Explicit JSON serialization for Upstash Redis.
        redis.set(
            key,
            json.dumps(payload),
            ex=self.ttl,
        )

        redis.sadd(
            INDEX_KEY,
            key,
        )

    def clear(self):
        """
        Remove all GroundedRAG cache entries.
        """

        keys = redis.smembers(INDEX_KEY)

        for key in keys:
            redis.delete(key)

        redis.delete(INDEX_KEY)