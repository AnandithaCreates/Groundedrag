"""
GroundedRAG Query Planner

The planner decides how a user query should be handled before
retrieval. It does not generate the final answer.

Supported query types:
    factual
    procedural
    comparative
    analytical
    multi_hop
    conversational
"""

import json

from app.llm_client import chat_completion


PLANNER_PROMPT = """
You are the query intelligence module of an enterprise document
intelligence system.

Analyze ONLY the user question at the bottom of this prompt.

Return exactly ONE valid JSON object.
Do not return Markdown.
Do not use ```json fences.
Do not write anything before or after the JSON.

Choose exactly one query_type:

- factual
  A question asking for a specific fact, value, definition, or setting.

- procedural
  A question asking how to perform, configure, deploy, or implement something.

- comparative
  A question comparing two or more concepts, technologies,
  configurations, or approaches.

- analytical
  A question requiring interpretation, tradeoffs, consequences,
  or implications based on indexed documents.

- multi_hop
  A question requiring information from multiple concepts or documents.

- conversational
  A follow-up question that depends on previous conversation context.

Return this JSON structure:

{
  "query_type": "factual",
  "search_query": "optimized retrieval query",
  "sub_queries": [
    "retrieval query"
  ],
  "requires_multiple_sources": false,
  "reasoning": "short retrieval strategy explanation"
}

Rules for sub_queries:

- Factual: exactly 1 query.
- Procedural: 1 or 2 queries.
- Comparative: exactly 2 queries when two concepts are being compared.
- Analytical: at most 2 queries.
- Multi-hop: at most 2 queries.
- Conversational: use 1 retrieval query based on the current question
  and conversation context.
- Never generate more than 2 sub_queries.
- Do not invent concepts that are not relevant to the user question.

Example:

{
  "query_type": "comparative",
  "search_query": "Cloud Run deployment methods vs Terraform operational risks",
  "sub_queries": [
    "Cloud Run deployment methods",
    "Terraform operational risks"
  ],
  "requires_multiple_sources": true,
  "reasoning": "Retrieve evidence for both concepts before comparison."
}

USER QUESTION:

<<USER_QUERY>>
"""


VALID_QUERY_TYPES = {
    "factual",
    "procedural",
    "comparative",
    "analytical",
    "multi_hop",
    "conversational",
}


def _fallback(query: str, reason: str) -> dict:
    return {
        "query_type": "factual",
        "search_query": query,
        "sub_queries": [query],
        "requires_multiple_sources": False,
        "reasoning": f"Planner fallback: {reason}",
    }


def _extract_json(raw: str) -> dict:
    """
    Parse the model output even if it accidentally adds text or
    Markdown around the JSON object.
    """

    text = raw.strip()

    # Remove Markdown fences.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # First try direct JSON parsing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recover the first complete JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner returned no JSON object.")

    return json.loads(
        text[start : end + 1]
    )


def _validate(plan: dict, original_query: str) -> dict:
    query_type = plan.get(
        "query_type",
        "factual",
    )

    if query_type not in VALID_QUERY_TYPES:
        query_type = "factual"

    search_query = plan.get(
        "search_query",
        original_query,
    )

    if not isinstance(search_query, str):
        search_query = original_query

    search_query = search_query.strip()

    sub_queries = plan.get(
        "sub_queries",
        [],
    )

    if not isinstance(sub_queries, list):
        sub_queries = []

    sub_queries = [
        item.strip()
        for item in sub_queries
        if isinstance(item, str)
        and item.strip()
    ]

    # Hard limit. No fourth mystery query.
    sub_queries = sub_queries[:2]

    if not sub_queries:
        sub_queries = [search_query]

    requires_multiple_sources = bool(
        plan.get(
            "requires_multiple_sources",
            len(sub_queries) > 1,
        )
    )

    reasoning = plan.get(
        "reasoning",
        "",
    )

    if not isinstance(reasoning, str):
        reasoning = ""

    return {
        "query_type": query_type,
        "search_query": search_query,
        "sub_queries": sub_queries,
        "requires_multiple_sources": (
            requires_multiple_sources
        ),
        "reasoning": reasoning,
    }


def plan_query(query: str) -> dict:
    """
    Generate a structured retrieval plan.

    IMPORTANT:
    The prompt uses a custom <<USER_QUERY>> placeholder instead of
    Python str.format(), so JSON braces inside the prompt can never
    cause a KeyError.
    """

    query = query.strip()

    if not query:
        return _fallback(
            "",
            "empty query",
        )

    prompt = PLANNER_PROMPT.replace(
        "<<USER_QUERY>>",
        query,
    )

    try:
        response = chat_completion(
            prompt,
            temperature=0,
            max_tokens=180,
        )

        plan = _extract_json(response)

        return _validate(
            plan,
            query,
        )

    except Exception as exc:
        print(
            f"[PLANNER ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        return _fallback(
            query,
            type(exc).__name__,
        )