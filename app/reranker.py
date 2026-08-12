from flashrank import Ranker, RerankRequest


_ranker = None


def get_ranker():
    global _ranker

    if _ranker is None:
        _ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="./models",
        )

    return _ranker


def rerank(query: str, documents: list, top_k: int = 4) -> list:
    """
    Rerank Qdrant's retrieved candidates using FlashRank.

    Qdrant:
        Fast vector similarity search

    FlashRank:
        More precise query-document relevance scoring

    Returns the highest-ranked documents.
    """

    if not documents:
        return []

    passages = [
        {
            "id": doc["chunk"].id,
            "text": doc["chunk"].text,
            "source": doc["chunk"].source,
        }
        for doc in documents
    ]

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    results = get_ranker().rerank(request)

    reranked = []

    for result in results[:top_k]:
        reranked.append(
            {
                "id": result["id"],
                "text": result["text"],
                "source": result["source"],
                "score": float(result["score"]),
            }
        )

    return reranked