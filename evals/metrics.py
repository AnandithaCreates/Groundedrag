"""
Three metrics, all hand-written so you can explain the exact formula
behind each number instead of citing a library.

faithfulness        -- does the answer's content actually resemble the
                        cited source chunks, or did the model drift?
answer_relevancy     -- does the answer resemble what a correct answer
                        should look like (vs. the golden expected_answer)?
retrieval_precision  -- of the chunks retrieved, what fraction came from
                        the document the question was actually about?
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.vector_store import embed_texts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def faithfulness(answer: str, sources: list) -> float:
    """Average, over each sentence in the answer, of its best cosine
    similarity to any retrieved source chunk. High score = the answer's
    language stays close to the actual source text rather than drifting
    into unsupported territory."""
    if not sources:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    sentence_vecs = embed_texts(sentences)
    source_vecs = embed_texts([s["text"] for s in sources])
    scores = []
    for svec in sentence_vecs:
        best = max(_cosine(svec, tvec) for tvec in source_vecs)
        scores.append(best)
    return float(np.mean(scores))


def answer_relevancy(answer: str, expected_answer: str) -> float:
    """Cosine similarity between the generated answer and the golden
    expected answer. Not a perfect proxy for correctness, but a cheap,
    explainable one -- deliberately simple over deliberately fancy."""
    if not answer or not expected_answer:
        return 0.0
    vecs = embed_texts([answer, expected_answer])
    return _cosine(vecs[0], vecs[1])


def retrieval_precision(sources: list, expected_source: str) -> float:
    """Fraction of retrieved chunks that came from the document the
    question was actually about."""
    if not sources:
        return 0.0
    hits = sum(1 for s in sources if s["source"] == expected_source)
    return hits / len(sources)
