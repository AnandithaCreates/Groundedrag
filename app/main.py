"""
GroundedRAG API

Architecture:

    New conversation
        ↓
    LangGraph agent
        ↓
    Postgres conversation state
        ↓
    Redis cache response

    Existing conversation
        ↓
    Skip global semantic cache
        ↓
    LangGraph + Postgres memory
        ↓
    Contextual retrieval
"""

import os
import traceback
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents.graph import run_query
from app.cache.redis_semantic import SemanticCache
from app.config import settings


app = FastAPI(
    title="GroundedRAG",
    version="2.0.0",
    description=(
        "Enterprise-style agentic RAG with guardrails, "
        "semantic caching, persistent memory, retrieval, "
        "reranking, and grounding verification."
    ),
)


STATIC_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "static",
)


cache = SemanticCache(
    similarity_threshold=settings.CACHE_SIM_THRESHOLD,
    ttl=settings.CACHE_TTL_SECONDS,
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    conversation_id: str | None = None


@app.post("/query")
def query(req: QueryRequest):

    query_text = req.query.strip()

    if not query_text:
        return {
            "answer": "Please enter a question.",
            "status": "blocked",
            "sources": [],
            "cache": {
                "hit": False,
            },
        }

    # --------------------------------------------------------
    # Conversation ID
    # --------------------------------------------------------

    is_new_conversation = (
        req.conversation_id is None
    )

    conversation_id = (
        req.conversation_id
        or str(uuid.uuid4())
    )

    # --------------------------------------------------------
    # CACHE POLICY
    #
    # IMPORTANT:
    # Never allow the global semantic cache to bypass a
    # stateful conversation.
    #
    # New conversation:
    #     Agent runs -> Postgres state -> Redis stores answer
    #
    # Existing conversation:
    #     Agent always runs -> Postgres context is preserved
    # --------------------------------------------------------

    if is_new_conversation:
        try:
            cached = cache.get(
                query_text
            )
        except Exception as exc:
            print(
                f"[CACHE GET ERROR] "
                f"{type(exc).__name__}: {exc}"
            )
            cached = None

        if cached is not None:

            cached_response = dict(
                cached["response"]
            )

            cached_response["cache"] = {
                "hit": True,
                "similarity": round(
                    float(
                        cached["similarity"]
                    ),
                    4,
                ),
                "cached_query": (
                    cached["cached_query"]
                ),
            }

            cached_response[
                "conversation_id"
            ] = conversation_id

            return cached_response

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    try:
        response = run_query(
            query_text,
            conversation_id,
        )

    except Exception as exc:

        print(
            f"[AGENT ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        return {
            "answer": (
                "The agent encountered an internal error."
            ),
            "status": "error",
            "sources": [],
            "cache": {
                "hit": False,
            },
            "conversation_id": conversation_id,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # --------------------------------------------------------
    # Cache successful grounded answers.
    #
    # This cache is useful for repeated standalone queries.
    # Stateful conversations continue through Postgres.
    # --------------------------------------------------------

    grounded = (
        response.get("status") == "ok"
        and response.get(
            "grounding",
            {},
        ).get(
            "passed",
            False,
        )
    )

    if grounded:
        try:
            cache.set(
                query_text,
                response,
            )
        except Exception as exc:
            print(
                f"[CACHE SET ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

    response["cache"] = {
        "hit": False,
    }

    response["conversation_id"] = (
        conversation_id
    )

    return response


@app.get("/health")
def health():

    try:
        redis_ok = cache.ping()
    except Exception:
        redis_ok = False

    return {
        "status": "ok",
        "services": {
            "redis": redis_ok,
            "agent": True,
            "qdrant": bool(
                settings.QDRANT_URL
            ),
            "postgres": bool(
                settings.POSTGRES_DSN
            ),
            "portkey": bool(
                settings.PORTKEY_API_KEY
            ),
        },
    }


app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


@app.get("/")
def index():
    return FileResponse(
        os.path.join(
            STATIC_DIR,
            "index.html"
        )
    )