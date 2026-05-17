from contextlib import contextmanager

import psycopg2

from app.config import config


class Database:
    def __init__(self):
        self.params = dict(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
        )

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(**self.params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


db = Database()