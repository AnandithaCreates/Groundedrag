import psycopg
from psycopg.rows import dict_row


class ConversationMemory:
    """
    Persistent conversation storage.

    This is intentionally simple for local development.
    Later it can be replaced by LangGraph PostgresSaver
    without changing the API layer.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._ensure_tables()

    def _connect(self):
        return psycopg.connect(
            self.dsn,
            row_factory=dict_row,
        )

    def _ensure_tables(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (id)
                    VALUES (%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (conversation_id,),
                )

                cur.execute(
                    """
                    INSERT INTO messages (
                        conversation_id,
                        role,
                        content
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        conversation_id,
                        role,
                        content,
                    ),
                )

    def get_history(
        self,
        conversation_id: str,
        limit: int = 20,
    ):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (
                        conversation_id,
                        limit,
                    ),
                )

                rows = cur.fetchall()

        return list(reversed(rows))