"""
All LLM calls in this project go through Portkey instead of hitting Groq
directly. Two concrete things this buys us, both real and demoable, not
just a resume line:

1. Caching: identical prompts within Portkey's free-tier "simple" cache
   window return instantly with $0 additional LLM cost, and you can show
   this in your demo (ask the same question twice, watch the latency drop).
2. A single choke point for retries/fallback config -- if you wanted to
   add a second provider as a fallback later, it's a config change here,
   not a rewrite of every call site.

Portkey exposes an OpenAI-compatible client, so this reads almost like a
normal OpenAI SDK call once you've supplied the virtual key.
"""
import time

from portkey_ai import Portkey

from app.config import settings
from app.evaluation_throttle import (
    get_evaluation_throttle,
    retry_delay,
    transient_status_code,
)

_client = None


def get_client() -> Portkey:
    global _client
    if _client is None:
        _client = Portkey(
            api_key=settings.PORTKEY_API_KEY,
            virtual_key=settings.PORTKEY_VIRTUAL_KEY,
        )
    return _client


def chat_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
    client = get_client()
    throttle = get_evaluation_throttle()

    for attempt in range((throttle.max_retries if throttle else 0) + 1):
        reservation = (
            throttle.before_request(prompt, max_tokens)
            if throttle else None
        )
        try:
            resp = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                # Enables Portkey's gateway-level cache. On a cache hit, Portkey
                # returns the stored response without forwarding to Groq at all.
                config={"cache": {"mode": settings.PORTKEY_CACHE_MODE}},
            )
            if throttle and reservation:
                throttle.record(reservation, response=resp)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if throttle and reservation:
                throttle.record(reservation, error=exc)
            transient = transient_status_code(exc) in {408, 429, 500, 502, 503, 504}
            if not throttle or not transient or attempt >= throttle.max_retries:
                raise
            time.sleep(retry_delay(attempt))
