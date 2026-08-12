"""
    python evals/run_evals.py

Runs every question in golden_set.json through the real agent (no mocking),
scores it, prints a table, and writes evals/results.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import run_query
from evals.metrics import faithfulness, answer_relevancy, retrieval_precision

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def evaluate_one(item: dict) -> dict:
    result = run_query(item["question"])
    expects_refusal = item["expected_answer"] in ("REFUSE", "BLOCK")

    row = {
        "question": item["question"],
        "status": result["status"],
        "answer": result["answer"],
    }

    if expects_refusal:
        row["refusal_correct"] = result["status"] in ("refused", "blocked")
        row["faithfulness"] = None
        row["answer_relevancy"] = None
        row["retrieval_precision"] = None
    else:
        row["refusal_correct"] = result["status"] == "ok"
        row["faithfulness"] = round(faithfulness(result["answer"], result["sources"]), 3) \
            if result["status"] == "ok" else 0.0
        row["answer_relevancy"] = round(
            answer_relevancy(result["answer"], item["expected_answer"]), 3
        ) if result["status"] == "ok" else 0.0
        row["retrieval_precision"] = round(
            retrieval_precision(result["sources"], item["expected_source"]), 3
        ) if result["status"] == "ok" else 0.0

    return row


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    rows = [evaluate_one(item) for item in golden]

    print(f"\n{'QUESTION':<55}{'STATUS':<10}{'FAITH':<8}{'RELEV':<8}{'PREC':<8}{'REFUSAL OK'}")
    print("-" * 100)
    for r in rows:
        q = (r["question"][:52] + "...") if len(r["question"]) > 52 else r["question"]
        faith = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "-"
        relev = f"{r['answer_relevancy']:.2f}" if r["answer_relevancy"] is not None else "-"
        prec = f"{r['retrieval_precision']:.2f}" if r["retrieval_precision"] is not None else "-"
        print(f"{q:<55}{r['status']:<10}{faith:<8}{relev:<8}{prec:<8}{r['refusal_correct']}")

    scored = [r for r in rows if r["faithfulness"] is not None]
    if scored:
        avg_faith = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_relev = sum(r["answer_relevancy"] for r in scored) / len(scored)
        avg_prec = sum(r["retrieval_precision"] for r in scored) / len(scored)
        print("\nAggregate over answerable questions:")
        print(f"  avg faithfulness:        {avg_faith:.3f}")
        print(f"  avg answer relevancy:    {avg_relev:.3f}")
        print(f"  avg retrieval precision: {avg_prec:.3f}")

    refusal_acc = sum(1 for r in rows if r["refusal_correct"]) / len(rows)
    print(f"  refusal/block accuracy:  {refusal_acc:.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
