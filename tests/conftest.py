from unittest.mock import MagicMock

import pytest


# Mock the database connection globally
@pytest.fixture(autouse=True) # don't need to explictly declare
def mock_db(monkeypatch):
    """Replace db.get_connection with a mock that returns a fake cursor."""
    from app.database import db

    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.__enter__.return_value = mock_conn

    # with db.get_connection() as conn: to mock_conn
    monkeypatch.setattr(db, "get_connection", MagicMock(return_value=mock_conn))
    return mock_cursor
