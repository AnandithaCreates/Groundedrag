import json

from app.llm_client import chat_completion


PLANNER_PROMPT = """
You are the query intelligence module of an enterprise document
intelligence system.

Analyze ONLY the user's question provided at the bottom.

Return ONLY one valid JSON object.
Do not write explanations before or after the JSON.
Do not use Markdown.
Do not use ```json fences.

Choose exactly ONE query_type:

- factual
  A question asking for a specific fact, value, definition, or setting.

- procedural
  A question asking how to perform, configure, deploy, or implement something.

- comparative
  A question comparing two or more concepts, technologies,
  configurations, or approaches.

- analytical
  A question requiring interpretation, reasoning, tradeoffs,
  consequences, or implications based on the indexed documents.

- multi_hop
  A question requiring information about multiple concepts or
  multiple documents before an answer can be constructed.

- conversational
  A follow-up question that depends on previous conversation context.

Generate these fields:

1. query_type
2. search_query
   A concise query optimized for document retrieval.

3. sub_queries
   Break the question into 1-4 focused retrieval queries.
   For a simple factual question, use one query.
   For comparative or multi-hop questions, use multiple queries.

4. requires_multiple_sources
   true if the answer should combine evidence from multiple
   concepts or documents, otherwise false.

5. reasoning
   A short explanation of the retrieval strategy.

Example:

{
  "query_type": "comparative",
  "search_query": "Cloud Run deployment methods and Terraform operational risks",
  "sub_queries": [
    "Cloud Run deployment methods",
    "Terraform operational risks"
  ],
  "requires_multiple_sources": true,
  "reasoning": "Retrieve evidence about both Cloud Run deployment methods and Terraform operational risks before comparing them."
}

USER QUESTION:
{query}
"""


def _fallback_plan(query: str, reason: str) -> dict:
    """
    Safe fallback when the planner LLM returns invalid JSON.
    """
    return {
        "query_type": "factual",
        "search_query": query,
        "sub_queries": [query],
        "requires_multiple_sources": False,
        "reasoning": f"Planner fallback: {reason}",
    }


def plan_query(query: str) -> dict:
    """
    Use the LLM to classify and decompose the user's query
    into retrieval-friendly queries.
    """

    prompt = PLANNER_PROMPT.format(query=query)

    try:
        raw = chat_completion(
            prompt,
            temperature=0,
            max_tokens=180,
        )

        # Remove accidental Markdown fences if the model adds them.
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            # Remove first line: ```json / ```
            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            # Remove final ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        # Parse JSON.
        plan = json.loads(cleaned)

        # Basic validation.
        allowed_types = {
            "factual",
            "procedural",
            "comparative",
            "analytical",
            "multi_hop",
            "conversational",
        }

        if plan.get("query_type") not in allowed_types:
            raise ValueError(
                f"Invalid query_type: {plan.get('query_type')}"
            )

        if not isinstance(plan.get("search_query"), str):
            raise ValueError("search_query must be a string")

        if not isinstance(plan.get("sub_queries"), list):
            raise ValueError("sub_queries must be a list")

        if not plan["sub_queries"]:
            plan["sub_queries"] = [query]

        if not isinstance(
            plan.get("requires_multiple_sources"),
            bool,
        ):
            raise ValueError(
                "requires_multiple_sources must be boolean"
            )

        if not isinstance(plan.get("reasoning"), str):
            plan["reasoning"] = "Planner generated retrieval strategy."

        return plan

    except json.JSONDecodeError as exc:
        print(f"[PLANNER ERROR] JSONDecodeError: {exc}")

        # Try extracting the JSON object if the model added
        # accidental text before/after it.
        try:
            start = raw.find("{")
            end = raw.rfind("}")

            if start != -1 and end != -1 and end > start:
                extracted = raw[start : end + 1]
                plan = json.loads(extracted)

                print("[PLANNER] Recovered JSON from model output.")

                return plan

        except Exception as recovery_error:
            print(
                f"[PLANNER RECOVERY ERROR] {recovery_error}"
            )

        return _fallback_plan(query, "JSONDecodeError")

    except Exception as exc:
        print(f"[PLANNER ERROR] {type(exc).__name__}: {exc}")
        return _fallback_plan(query, type(exc).__name__)