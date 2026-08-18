"""Evaluator-only safeguards for quota-limited LLM providers.

The limiter is inactive unless ``EVAL_MODE=1``. Production request handling
therefore retains its existing Portkey behavior.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from math import ceil
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


class EvaluationThrottle:
    """Serialize evaluation calls and conservatively reserve token budget."""

    def __init__(self) -> None:
        self.min_interval = float(os.getenv("EVAL_MIN_INTERVAL_SECONDS", "10"))
        self.max_rpm = _env_int("EVAL_MAX_REQUESTS_PER_MINUTE", 5)
        self.max_tpm = _env_int("EVAL_MAX_TOKENS_PER_MINUTE", 6000)
        self.max_retries = _env_int("EVAL_MAX_RETRIES", 2)
        self._lock = threading.Lock()
        self._requests: deque[tuple[float, int]] = deque()
        self._next_allowed = 0.0
        self._events: list[dict[str, Any]] = []

    @property
    def effective_interval(self) -> float:
        return max(self.min_interval, 60.0 / self.max_rpm)

    @staticmethod
    def estimate_tokens(prompt: str, max_tokens: int) -> int:
        # Deliberately conservative and labelled as an estimate. Portkey/Groq
        # usage metadata, when present, is recorded separately as actual.
        return ceil(len(prompt) / 3) + max_tokens

    def before_request(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        estimate = self.estimate_tokens(prompt, max_tokens)
        waited = 0.0
        with self._lock:
            while True:
                now = time.monotonic()
                while self._requests and now - self._requests[0][0] >= 60:
                    self._requests.popleft()
                used = sum(tokens for _, tokens in self._requests)
                wait_for_interval = max(0.0, self._next_allowed - now)
                wait_for_tokens = 0.0
                if self._requests and used + estimate > self.max_tpm:
                    wait_for_tokens = max(0.0, 60 - (now - self._requests[0][0]))
                wait_for = max(wait_for_interval, wait_for_tokens)
                if wait_for <= 0:
                    break
                time.sleep(wait_for)
                waited += wait_for
            now = time.monotonic()
            self._requests.append((now, estimate))
            self._next_allowed = now + self.effective_interval
        return {"estimated_tokens": estimate, "wait_seconds": round(waited, 3)}

    def record(self, reservation: dict[str, Any], response: Any = None, error: Exception | None = None) -> None:
        usage = getattr(response, "usage", None) if response is not None else None
        actual = None
        if usage is not None:
            actual = {
                "input": _usage_value(usage, "prompt_tokens"),
                "output": _usage_value(usage, "completion_tokens"),
                "total": _usage_value(usage, "total_tokens"),
            }
        self._events.append({
            **reservation,
            "actual_tokens": actual,
            "error_type": type(error).__name__ if error else None,
        })

    def drain_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events


def _usage_value(usage: Any, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return int(value) if isinstance(value, (int, float)) else None


_throttle: EvaluationThrottle | None = None


def get_evaluation_throttle() -> EvaluationThrottle | None:
    global _throttle
    if os.getenv("EVAL_MODE") != "1":
        return None
    if _throttle is None:
        _throttle = EvaluationThrottle()
    return _throttle


def transient_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else getattr(exc, "status_code", None)


def retry_delay(attempt: int) -> float:
    return (5 if attempt == 0 else 15) + random.uniform(0, 1)
