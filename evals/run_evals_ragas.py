"""
    python evals/run_evals_ragas.py

Runs the same golden set through RAGAS's standard metrics (faithfulness,
answer_relevancy, context_precision) instead of the hand-coded versions
in metrics.py. Keep both scripts -- this one is the "I used the standard
library" answer, evals/run_evals.py is the "I understand the formula
underneath it" answer. Being able to run both, and explain why the
hand-coded numbers roughly track the RAGAS numbers, is a strong signal.

RAGAS has a genuinely fragile dependency chain (heavy langchain
sub-package coupling). If this script fails to even import in your
environment, that's a known RAGAS issue, not a bug in this project --
fall back to `python evals/run_evals.py`, which has zero fragile
dependencies, and mention the tradeoff if asked.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    IMPORT_ERROR = e

from app.config import settings
from app.graph import run_query

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ragas.json")


def build_dataset(golden: list) -> "Dataset":
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden:
        if item["expected_answer"] in ("REFUSE", "BLOCK"):
            continue  # RAGAS metrics assume an actual answer was given; skip refusal cases here
        result = run_query(item["question"])
        if result["status"] != "ok":
            continue
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["text"] for s in result["sources"]])
        rows["ground_truth"].append(item["expected_answer"])
    return Dataset.from_dict(rows)


def main():
    if not RAGAS_AVAILABLE:
        print(f"RAGAS could not be imported: {IMPORT_ERROR}")
        print("Falling back note: run `python evals/run_evals.py` instead -- ")
        print("it covers the same three metrics with zero fragile dependencies.")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    dataset = build_dataset(golden)
    if len(dataset) == 0:
        print("No answerable questions produced valid results -- check your Qdrant/Portkey setup.")
        return

    judge_llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)
    judge_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\nRAGAS scores:")
    print(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(result.to_pandas().to_dict(orient="records"), f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
