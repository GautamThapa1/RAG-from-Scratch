from contextlib import contextmanager

import psycopg2

from app.config import config


class DatabaseManager:
    """Manages PostgreSQL connections with automatic commit/rollback."""

    def __init__(self):
        self._params = dict(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
        )

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(**self._params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def health_check(self) -> str:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version();")
            return cur.fetchone()[0]


# Module-level singleton
db = DatabaseManager()