from app.vector_store import get_store
from app.reranker import rerank
from app.config import settings


def retrieve_evidence(
    search_query: str,
    sub_queries: list[str] | None = None,
    top_k: int = 4,
):
    """
    Enterprise-style retrieval pipeline.

    Flow:

        Planner
           ↓
        Multi-query retrieval
           ↓
        Candidate merging
           ↓
        FlashRank reranking
           ↓
        Relevance filtering
           ↓
        Evidence returned to agent
    """

    store = get_store()

    queries = sub_queries or [search_query]

    # Retrieve a larger candidate pool.
    candidates = store.multi_search(
        queries,
        k=max(top_k * 3, 10),
    )

    if not candidates:
        return []

    # Semantic reranking.
    reranked = rerank(
        search_query,
        candidates,
        top_k=top_k,
    )

    # Remove weak evidence.
    threshold = getattr(
        settings,
        "RERANK_THRESHOLD",
        0.20,
    )

    filtered = [
        result
        for result in reranked
        if result.get("score", 0.0) >= threshold
    ]

    return filtered