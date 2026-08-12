"""
GroundedRAG Agentic Execution Graph

Pipeline:

                    ┌──────────────┐
                    │   PRECHECK   │
                    │  guardrails  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    PLAN      │
                    │ query intent │
                    │ + subqueries │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   RETRIEVE   │
                    │    Qdrant    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   RERANK     │
                    │  FlashRank   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   GENERATE   │
                    │ LLM + cites  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   CRITIQUE   │
                    │ grounded?    │
                    └──────┬───────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
              grounded           not grounded
                 │                   │
                 ▼                   ▼
               FINAL            regenerate
                                     │
                                     └──► max loops
                                          └──► refuse
"""

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.guardrails import post_check, pre_check
from app.llm_client import chat_completion
from app.planner import plan_query
from app.retrieval import retrieve_evidence


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    # Original user query
    query: str

    # Planner output
    query_type: str
    search_query: str
    sub_queries: List[str]
    requires_multiple_sources: bool
    planning_reason: str

    # Retrieved evidence
    sources: List[dict]

    # Generation
    answer: Optional[str]

    # Verification
    grounded: bool
    issues: List[str]

    # Agent loop
    refine_count: int

    # Execution status
    status: str
    refusal_reason: Optional[str]


# ============================================================
# NODE 1: PRECHECK
# ============================================================

def node_precheck(state: AgentState) -> AgentState:
    """
    Run the existing input guardrail before any retrieval or LLM
    generation happens.
    """

    result = pre_check(state["query"])

    if result.get("decision") == "block":
        state["status"] = "blocked"
        state["refusal_reason"] = result.get(
            "reason",
            "Query blocked by guardrail.",
        )

    return state


# ============================================================
# NODE 2: PLAN + RETRIEVE + RERANK
# ============================================================

def node_retrieve(state: AgentState) -> AgentState:
    """
    Agentic retrieval stage.

    Planner:
        Converts the user's question into a retrieval strategy.

    Retrieval:
        Searches Qdrant using one or multiple search queries.

    Reranking:
        FlashRank reorders the retrieved candidates according to
        query-document relevance.

    This keeps planning, retrieval and ranking separate from the
    generation stage.
    """

    if state.get("status") == "blocked":
        return state

    # --------------------------------------------------------
    # 1. PLAN
    # --------------------------------------------------------

    plan = plan_query(state["query"])

    state["query_type"] = plan.get(
        "query_type",
        "factual",
    )

    state["search_query"] = plan.get(
        "search_query",
        state["query"],
    )

    state["sub_queries"] = plan.get(
        "sub_queries",
        [state["query"]],
    )

    state["requires_multiple_sources"] = bool(
        plan.get(
            "requires_multiple_sources",
            len(state["sub_queries"]) > 1,
        )
    )

    state["planning_reason"] = plan.get(
        "reasoning",
        plan.get("reason", ""),
    )

    # Safety fallback
    if not state["sub_queries"]:
        state["sub_queries"] = [
            state["search_query"]
        ]

    # --------------------------------------------------------
    # 2. RETRIEVE + RERANK
    # --------------------------------------------------------

    results = retrieve_evidence(
        search_query=state["search_query"],
        sub_queries=state["sub_queries"],
        top_k=settings.TOP_K,
    )

    # --------------------------------------------------------
    # 3. NORMALIZE RESULTS
    # --------------------------------------------------------

    state["sources"] = [
        {
            "id": result["id"],
            "text": result["text"],
            "source": result["source"],
            "score": float(result["score"]),
        }
        for result in results
    ]

    # --------------------------------------------------------
    # 4. EVIDENCE CHECK
    # --------------------------------------------------------

    if not state["sources"]:
        state["status"] = "refused"
        state["refusal_reason"] = (
            "I couldn't find sufficiently relevant evidence "
            "in the indexed documents to answer this question."
        )

    return state


# ============================================================
# GENERATION PROMPT
# ============================================================

GENERATE_PROMPT = """
You are the answer generation component of an enterprise
document intelligence system.

Answer the user's question using ONLY the retrieved evidence.

Rules:

1. Do not use outside knowledge.
2. Every factual claim must have a citation immediately after it.
3. Citations must use the exact chunk ID supplied in SOURCES.
4. If the evidence does not support a claim, do not make the claim.
5. For comparative questions, clearly separate the evidence from
   each relevant document.
6. Prefer concise, evidence-dense answers.
7. If this is a regenerated answer, fix the issues identified
   by the previous grounding critique.

PREVIOUS CRITIQUE:
{issues}

SOURCES:
{sources}

QUESTION:
{question}
"""


# ============================================================
# NODE 3: GENERATE
# ============================================================

def node_generate(state: AgentState) -> AgentState:
    if state.get("status") in ("blocked", "refused"):
        return state

    source_block = "\n\n".join(
        f"[{source['id']}]: {source['text']}"
        for source in state["sources"]
    )

    issues = state.get("issues", [])

    if issues:
        critique_block = "\n".join(
            f"- {issue}" for issue in issues
        )
    else:
        critique_block = "No previous critique. Generate the initial answer."

    prompt = GENERATE_PROMPT.format(
        sources=source_block,
        question=state["query"],
        issues=critique_block,
    )

    state["answer"] = chat_completion(
        prompt,
        temperature=0.1,
        max_tokens=400,
    )

    return state


# ============================================================
# NODE 4: CRITIQUE / GROUNDING VERIFICATION
# ============================================================

def node_critique(state: AgentState) -> AgentState:
    """
    Verify whether the generated answer is actually supported
    by the retrieved evidence.
    """

    if state.get("status") in {
        "blocked",
        "refused",
    }:
        return state

    if not state.get("answer"):
        state["grounded"] = False
        state["issues"] = [
            "No answer was generated."
        ]
        return state

    result = post_check(
        state["answer"],
        state["sources"],
    )

    state["grounded"] = bool(
        result.get("grounded", False)
    )

    state["issues"] = result.get(
        "issues",
        [],
    )

    return state


# ============================================================
# ROUTER
# ============================================================

def route_after_critique(state: AgentState) -> str:
    """
    Decide whether the answer is acceptable, should be regenerated once,
    or should be refused.

    Maximum one regeneration per query.
    """

    if state.get("status") in ("blocked", "refused"):
        return "end"

    # Critique passed.
    if state.get("grounded", False):
        state["status"] = "ok"
        return "end"

    # Allow exactly one regeneration.
    if state.get("refine_count", 0) < 1:
        state["refine_count"] += 1
        return "retry"

    # Second failure -> refuse instead of hallucinating.
    state["status"] = "refused"
    state["refusal_reason"] = (
        "I could not produce an answer that was sufficiently "
        "grounded in the retrieved evidence."
    )

    return "end"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph():
    """
    Build and compile the LangGraph execution graph.
    """

    graph = StateGraph(AgentState)

    graph.add_node(
        "precheck",
        node_precheck,
    )

    graph.add_node(
        "retrieve",
        node_retrieve,
    )

    graph.add_node(
        "generate",
        node_generate,
    )

    graph.add_node(
        "critique",
        node_critique,
    )

    # Entry
    graph.set_entry_point("precheck")

    # Main pipeline
    graph.add_edge(
        "precheck",
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "generate",
    )

    graph.add_edge(
        "generate",
        "critique",
    )

    # Verification loop
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "retry": "generate",
            "end": END,
        },
    )

    return graph.compile()


# ============================================================
# SINGLETON AGENT
# ============================================================

_agent = None


def get_agent():
    """
    Lazily initialize the compiled graph.
    """

    global _agent

    if _agent is None:
        _agent = build_graph()

    return _agent


# ============================================================
# PUBLIC QUERY API
# ============================================================

def run_query(query: str) -> dict:
    """
    Execute the complete agentic RAG pipeline.

    Returns:
        answer
        status
        sources
        retrieval metadata
        grounding metadata
        planner metadata
        original query
    """

    query = query.strip()

    if not query:
        return {
            "answer": "Please provide a question.",
            "status": "blocked",
            "sources": [],
            "retrieval": {},
            "grounding": {},
            "planning": {},
            "query": query,
        }

    agent = get_agent()

    initial: AgentState = {
        "query": query,

        # Planner
        "query_type": "",
        "search_query": query,
        "sub_queries": [query],
        "requires_multiple_sources": False,
        "planning_reason": "",

        # Retrieval
        "sources": [],

        # Generation
        "answer": None,

        # Verification
        "grounded": False,
        "issues": [],

        # Loop
        "refine_count": 0,

        # Status
        "status": "ok",
        "refusal_reason": None,
    }

    final_state = agent.invoke(initial)

    sources = final_state.get(
        "sources",
        [],
    )

    # ========================================================
    # RETRIEVAL METADATA
    # ========================================================

    retrieval_info = {
        "chunks_found": len(sources),
        "chunks_used": len(sources),
        "threshold": settings.RELEVANCE_THRESHOLD,
        "top_score": max(
            (
                float(source["score"])
                for source in sources
            ),
            default=0.0,
        ),
        "top_k": settings.TOP_K,
    }

    # ========================================================
    # GROUNDING METADATA
    # ========================================================

    grounding_info = {
        "passed": bool(
            final_state.get(
                "grounded",
                False,
            )
        ),
        "issues": final_state.get(
            "issues",
            [],
        ),
        "refinements": final_state.get(
            "refine_count",
            0,
        ),
    }

    # ========================================================
    # PLANNER METADATA
    # ========================================================

    planning_info = {
        "query_type": final_state.get(
            "query_type",
            "",
        ),
        "search_query": final_state.get(
            "search_query",
            query,
        ),
        "sub_queries": final_state.get(
            "sub_queries",
            [query],
        ),
        "requires_multiple_sources": final_state.get(
            "requires_multiple_sources",
            False,
        ),
        "reasoning": final_state.get(
            "planning_reason",
            "",
        ),
    }

    # ========================================================
    # REFUSAL / BLOCKED RESPONSE
    # ========================================================

    if final_state["status"] != "ok":
        return {
            "answer": final_state.get(
                "refusal_reason",
                "The request could not be answered.",
            ),
            "status": final_state["status"],
            "sources": sources,
            "retrieval": retrieval_info,
            "grounding": grounding_info,
            "planning": planning_info,
            "query": query,
        }

    # ========================================================
    # SUCCESS
    # ========================================================

    return {
        "answer": final_state.get(
            "answer",
            "",
        ),
        "status": "ok",
        "sources": sources,
        "retrieval": retrieval_info,
        "grounding": grounding_info,
        "planning": planning_info,
        "query": query,
    }