from typing import TypedDict, List, Dict, Optional, Any


class Evidence(TypedDict):
    id: str
    text: str
    source: str
    score: float
    rerank_score: float
    metadata: Dict[str, Any]


class Claim(TypedDict):
    text: str
    citations: List[str]
    supported: bool
    confidence: float


class AgentState(TypedDict):
    # Request
    query: str
    request_id: str

    # Query understanding
    intent: str
    query_type: str
    rewritten_query: Optional[str]

    # Retrieval
    retrieval_attempt: int
    retrieval_strategy: str
    sources: List[Evidence]

    # Evidence
    evidence: List[Evidence]
    evidence_score: float
    evidence_sufficient: bool

    # Generation
    answer: Optional[str]
    claims: List[Claim]

    # Verification
    grounded: bool
    citation_valid: bool
    verification_issues: List[str]
    confidence: float

    # Control flow
    refine_count: int
    max_retrieval_attempts: int

    # Final state
    status: str
    refusal_reason: Optional[str]