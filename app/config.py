import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM, routed through Portkey rather than called directly -- gives us
    # a gateway (retries/fallback config) and request-level caching for free.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    PORTKEY_API_KEY: str = os.getenv("PORTKEY_API_KEY", "")
    # Portkey's "virtual key" is how it stores your Groq credential server-side
    # so your app never has to hold the raw provider key. Create one at
    # app.portkey.ai -> Virtual Keys -> Add -> Groq.
    PORTKEY_VIRTUAL_KEY: str = os.getenv("PORTKEY_VIRTUAL_KEY", "")
    # "simple" (exact-match) caching is included on Portkey's free Developer
    # tier. "semantic" caching is a paid-tier feature -- documented tradeoff,
    # not an oversight: we accept exact-match-only caching to stay free.
    PORTKEY_CACHE_MODE: str = os.getenv("PORTKEY_CACHE_MODE", "simple")

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Qdrant Cloud free cluster: 1GB RAM / 4GB disk, no card required.
    # Free clusters auto-suspend after 1 week idle -- reactivate from the
    # Qdrant console before a demo if it's been quiet.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "groundedrag_chunks")

    INDEX_DIR: str = os.getenv("INDEX_DIR", "./index_store")  # kept only for local fallback/testing

    TOP_K: int = int(os.getenv("TOP_K", "4"))
    MAX_REFINE_LOOPS: int = int(os.getenv("MAX_REFINE_LOOPS", "2"))

    # Below this similarity score, a chunk is considered irrelevant to the query.
    # Tuned for cosine similarity on MiniLM embeddings; adjust if you swap embedding models.
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
    RERANK_THRESHOLD: float = 0.20

    # Cache: if a new query is more similar than this to a cached one, reuse the cached answer.
    CACHE_SIM_THRESHOLD: float = float(os.getenv("CACHE_SIM_THRESHOLD", "0.92"))


settings = Settings()
