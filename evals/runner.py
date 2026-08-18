"""Sequential, resumable, quota-safe GroundedRAG evaluation runner."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("EVAL_MODE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.graph import run_query
from app.config import settings
from app.evaluation_throttle import get_evaluation_throttle
from app.main import QueryRequest, query
from evals.metrics import answer_relevancy, faithfulness, retrieval_precision

EVALS_DIR = Path(__file__).parent
DATASET_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_dataset() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must contain a list.")
    return data


def citations(answer: str, sources: list[dict]) -> dict:
    found = CITATION_PATTERN.findall(answer)
    ids = {str(source.get("id", "")) for source in sources}
    invalid = [item for item in found if item not in ids]
    return {"present": bool(found), "valid": bool(found) and not invalid, "invalid": invalid}


def response_result(result: dict, spec: dict, latency: float) -> dict:
    answer, sources = result.get("answer", ""), result.get("sources", [])
    metrics: dict[str, float] = {}
    if spec.get("expected_answer"):
        metrics["faithfulness_similarity"] = faithfulness(answer, sources)
        metrics["answer_relevancy_similarity"] = answer_relevancy(answer, spec["expected_answer"])
    if spec.get("expected_source"):
        metrics["retrieval_precision"] = retrieval_precision(sources, spec["expected_source"])
    expected_status = spec.get("expected_status", "ok")
    context_expected = spec.get("expected_contextualized_query_contains")
    planning = result.get("planning", {})
    context_match = None if context_expected is None else context_expected.lower() in planning.get("contextualized_query", "").lower()
    return {
        "agent_status": result.get("status", "unknown"),
        "expected_status": expected_status,
        "status_matches": result.get("status") == expected_status,
        "citation": citations(answer, sources),
        "contextualization_matches": context_match,
        "metrics": metrics,
        "latency_ms": round(latency * 1000, 1),
        "sources": [source.get("source") for source in sources],
        "grounding": result.get("grounding", {}),
        "planning": planning,
    }


def usage() -> dict:
    limiter = get_evaluation_throttle()
    events = limiter.drain_events() if limiter else []
    actual = [event["actual_tokens"] for event in events if event["actual_tokens"]]
    totals = {key: sum(item.get(key) or 0 for item in actual) for key in ("input", "output", "total")}
    return {
        "actual": totals if actual else "NOT_AVAILABLE",
        "estimated_total": sum(event["estimated_tokens"] for event in events),
        "calls": len(events),
        "throttle_wait_seconds": round(sum(event["wait_seconds"] for event in events), 3),
        "retries": sum(event["error_type"] is not None for event in events),
    }


def run_agent(spec: dict, thread_id: str) -> dict:
    started = time.perf_counter()
    result = run_query(spec["question"], thread_id)
    return response_result(result, spec, time.perf_counter() - started)


def run_case(case: dict) -> dict:
    get_evaluation_throttle().drain_events()
    started = time.perf_counter()
    try:
        if case["type"] == "multi_turn":
            turns = []
            thread_id = f"eval-{case['id']}"
            for index, turn in enumerate(case["turns"]):
                turn_result = run_agent(turn, thread_id)
                turns.append(turn_result)
            evaluated = turns[-1]
            passed = all(turn["status_matches"] for turn in turns) and evaluated["citation"]["valid"] and evaluated["contextualization_matches"] is not False
            payload = {"turns": turns, "agent_status": evaluated["agent_status"], "metrics": evaluated["metrics"], "citation": evaluated["citation"], "cache": "bypass"}
        elif case["type"] == "cache_behavior":
            first = query(QueryRequest(query=case["question"]))
            second = query(QueryRequest(query=case["question"]))
            evaluated = response_result(second, case, time.perf_counter() - started)
            cache_hit = bool(second.get("cache", {}).get("hit", False))
            passed = evaluated["status_matches"] and cache_hit and evaluated["citation"]["valid"]
            payload = {**evaluated, "cache": "hit" if cache_hit else "miss", "first_cache_hit": bool(first.get("cache", {}).get("hit", False))}
        else:
            evaluated = run_agent(case, f"eval-{case['id']}")
            expected_ok = case.get("expected_status", "ok") == "ok"
            passed = evaluated["status_matches"] and (not expected_ok or evaluated["citation"]["valid"])
            payload = {**evaluated, "cache": "bypass"}
        return {"case_id": case["id"], "category": case["type"], "status": "passed" if passed else "failed", "error": None, "latency_ms": round((time.perf_counter() - started) * 1000, 1), "tokens": usage(), **payload}
    except Exception as exc:
        return {"case_id": case["id"], "category": case["type"], "status": "error", "error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.perf_counter() - started) * 1000, 1), "cache": "bypass", "tokens": usage(), "metrics": {}}


def checkpoint(run: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(run, indent=2), encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def aggregate(results: list[dict], complete: bool) -> dict | str:
    if not complete:
        return "INSUFFICIENT_DATA"
    values = [result["latency_ms"] for result in results]
    values.sort()
    def percentile(p: float) -> float:
        return values[min(len(values) - 1, round((len(values) - 1) * p))]
    metric_names = sorted({name for result in results for name in result.get("metrics", {})})
    return {
        "pass_rate": round(sum(result["status"] == "passed" for result in results) / len(results), 3),
        "metrics": {name: round(sum(result["metrics"][name] for result in results if name in result["metrics"]) / sum(name in result["metrics"] for result in results), 3) for name in metric_names},
        "citation_id_validity": round(sum(result.get("citation", {}).get("valid", False) for result in results if result.get("agent_status") == "ok") / max(1, sum(result.get("agent_status") == "ok" for result in results)), 3),
        "latency_ms": {"p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99) if len(values) >= 100 else "INSUFFICIENT_DATA"},
        "retrieval_recall_at_k": "NOT_IMPLEMENTED",
        "mrr": "NOT_IMPLEMENTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Start a new run; do not reuse checkpointed cases.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many pending cases.")
    args = parser.parse_args()
    dataset = load_dataset()
    if args.fresh or not CHECKPOINT_PATH.exists():
        run = {"run_id": f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}", "started_at": now(), "commit": git_commit(), "model": settings.GROQ_MODEL, "provider": "groq", "gateway": "portkey", "total_cases": len(dataset), "results": []}
    else:
        run = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    limiter = get_evaluation_throttle()
    print(f"GroundedRAG Evaluation\nModel: {settings.GROQ_MODEL}\nGateway: Portkey -> Groq\nCases: {len(dataset)}\nConcurrency: 1\nMax RPM: {limiter.max_rpm}\nToken budget: {limiter.max_tpm} TPM\nThrottle: {limiter.effective_interval:g}s minimum interval")
    done = {result["case_id"] for result in run["results"]}
    pending = [case for case in dataset if case["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    for case in pending:
        print(f"\nRunning {case['id']} ({case['type']})", flush=True)
        result = run_case(case)
        print(f"  {result['status']} in {result['latency_ms']:.1f}ms; calls={result['tokens']['calls']}; throttle_wait={result['tokens']['throttle_wait_seconds']}s", flush=True)
        run["results"].append(result)
        checkpoint(run)
    completed = len(run["results"])
    complete = completed == len(dataset)
    run.update({"completed_cases": completed, "failed_cases": sum(result["status"] == "failed" for result in run["results"]), "blocked_cases": sum(result.get("agent_status") == "blocked" for result in run["results"]), "complete": complete, "completed_at": now() if complete else None, "aggregate": aggregate(run["results"], complete)})
    checkpoint(run)
    if complete:
        path = RESULTS_DIR / f"baseline_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"Complete report: {path}")
    else:
        print(f"INCOMPLETE: {completed}/{len(dataset)} cases. Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
