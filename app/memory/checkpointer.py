from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import settings


_pool = None
_checkpointer = None


def get_checkpointer():
    global _pool
    global _checkpointer

    if _checkpointer is None:
        _pool = ConnectionPool(
            conninfo=settings.POSTGRES_DSN,
            min_size=1,
            max_size=5,
            max_idle=300,
            max_lifetime=1800,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
            },
        )

        _pool.wait()

        _checkpointer = PostgresSaver(
            conn=_pool,
        )

        _checkpointer.setup()

    return _checkpointer