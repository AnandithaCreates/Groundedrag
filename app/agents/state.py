from typing import Any, Dict, List, Optional, TypedDict


class Evidence(TypedDict, total=False):
    id: str
    text: str
    source: str

    # Retrieval scores
    vector_score: float
    rerank_score: float

    # Optional metadata
    metadata: Dict[str, Any]


class Claim(TypedDict, total=False):
    text: str
    citations: List[str]
    supported: bool
    confidence: float


class AgentState(TypedDict):
    # ========================================================
    # Request
    # ========================================================

    query: str
    request_id: str

    # ========================================================
    # Conversation
    # ========================================================

    history: List[Dict[str, str]]
    contextualized_query: str

    # ========================================================
    # Query understanding / planning
    # ========================================================

    intent: str
    query_type: str
    search_query: str
    sub_queries: List[str]
    requires_multiple_sources: bool
    planning_reason: str

    # ========================================================
    # Retrieval
    # ========================================================

    retrieval_attempt: int
    retrieval_strategy: str
    sources: List[Evidence]

    # ========================================================
    # Evidence
    # ========================================================

    evidence: List[Evidence]
    evidence_score: float
    evidence_sufficient: bool

    # ========================================================
    # Generation
    # ========================================================

    answer: Optional[str]
    claims: List[Claim]

    # ========================================================
    # Verification
    # ========================================================

    grounded: bool
    citation_valid: bool
    verification_issues: List[str]
    issues: List[str]
    confidence: float

    # ========================================================
    # Control flow
    # ========================================================

    refine_count: int
    max_retrieval_attempts: int

    # ========================================================
    # Final state
    # ========================================================

    status: str
    refusal_reason: Optional[str]