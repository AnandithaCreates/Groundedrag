"""
Contextual query rewriting for conversational RAG.

The contextualizer converts a follow-up question into a standalone
retrieval query using the recent conversation history.

It does NOT answer the question.
"""

import json

from app.llm_client import chat_completion


CONTEXTUALIZER_PROMPT = """
You are a query contextualizer for an enterprise RAG system.

Your task is to rewrite the user's CURRENT QUESTION into a standalone
retrieval query using the conversation history.

Rules:

1. Preserve the meaning of the current question.
2. Resolve references such as:
   - it
   - that
   - this
   - they
   - the above
   - that setting
3. Do not answer the question.
4. Do not invent facts.
5. If the current question is already standalone, return it unchanged.
6. Return ONLY valid JSON.
7. Keep the rewritten query concise.

Return:

{
  "standalone_query": "..."
}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION:
{question}
"""


def contextualize_query(
    question: str,
    history: list[dict],
) -> str:
    """
    Convert a conversational follow-up into a standalone query.
    """

    if not history:
        return question

    history_text = "\n".join(
        f"{item.get('role', 'unknown')}: "
        f"{item.get('content', '')}"
        for item in history[-6:]
    )

    # Avoid Python .format() JSON-brace problems.
    prompt = CONTEXTUALIZER_PROMPT.replace(
        "{history}",
        history_text,
    ).replace(
        "{question}",
        question,
    )

    try:
        raw = chat_completion(
            prompt,
            temperature=0,
            max_tokens=120,
        ).strip()

        # Remove accidental markdown fences.
        if raw.startswith("```"):
            lines = raw.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            raw = "\n".join(lines).strip()

        result = json.loads(raw)

        standalone = result.get(
            "standalone_query",
            question,
        )

        if (
            isinstance(standalone, str)
            and standalone.strip()
        ):
            return standalone.strip()

    except Exception as exc:
        print(
            f"[CONTEXTUALIZER ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

    # Safe fallback.
    return question