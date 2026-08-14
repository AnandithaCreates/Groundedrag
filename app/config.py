import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # =========================================================
    # LLM / GROQ / PORTKEY
    # =========================================================

    GROQ_API_KEY: str = os.getenv(
        "GROQ_API_KEY",
        "",
    )

    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    PORTKEY_API_KEY: str = os.getenv(
        "PORTKEY_API_KEY",
        "",
    )

    PORTKEY_VIRTUAL_KEY: str = os.getenv(
        "PORTKEY_VIRTUAL_KEY",
        "",
    )

    PORTKEY_CACHE_MODE: str = os.getenv(
        "PORTKEY_CACHE_MODE",
        "simple",
    )

    # =========================================================
    # EMBEDDING MODEL
    # =========================================================

    EMBED_MODEL: str = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # =========================================================
    # QDRANT
    # =========================================================

    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "",
    )

    QDRANT_API_KEY: str = os.getenv(
        "QDRANT_API_KEY",
        "",
    )

    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION",
        "groundedrag_chunks",
    )

    # =========================================================
    # INDEXING
    # =========================================================

    INDEX_DIR: str = os.getenv(
        "INDEX_DIR",
        "./index_store",
    )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    TOP_K: int = int(
        os.getenv(
            "TOP_K",
            "4",
        )
    )

    RELEVANCE_THRESHOLD: float = float(
        os.getenv(
            "RELEVANCE_THRESHOLD",
            "0.35",
        )
    )

    RERANK_THRESHOLD: float = float(
        os.getenv(
            "RERANK_THRESHOLD",
            "0.20",
        )
    )

    # =========================================================
    # AGENT REFINEMENT
    # =========================================================

    # Keep this at 1 while using the free Groq tier so one bad
    # generation does not trigger a cascade of LLM calls.
    MAX_REFINE_LOOPS: int = int(
        os.getenv(
            "MAX_REFINE_LOOPS",
            "1",
        )
    )

    # =========================================================
    # CACHE
    # =========================================================

    CACHE_SIM_THRESHOLD: float = float(
        os.getenv(
            "CACHE_SIM_THRESHOLD",
            "0.92",
        )
    )

    REDIS_URL: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    # Cache lifetime in seconds.
    CACHE_TTL_SECONDS: int = int(
        os.getenv(
            "CACHE_TTL_SECONDS",
            "3600",
        )
    )

    # =========================================================
    # PERSISTENT CONVERSATION MEMORY
    # =========================================================

    POSTGRES_DSN: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/groundedrag",
    )

    # Maximum conversation messages loaded into agent context.
    MEMORY_LIMIT: int = int(
        os.getenv(
            "MEMORY_LIMIT",
            "20",
        )
    )


settings = Settings()