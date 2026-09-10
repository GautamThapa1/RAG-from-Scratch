from app.config import config


def test_config_defaults():
    """Check that defaults are used when env vars are missing."""
    assert config.DB_NAME == "rag_db"
    assert config.CHUNK_SIZE == 500
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.LLM_PROVIDER in ("local", "groq")

def test_config_env_overrides(monkeypatch):
    """Checking if the env is reflected."""
    monkeypatch.setenv("DB_NAME", "test_db") # changing temporarily to see
    # Reload config module to pick up new env
    import importlib

    import app.config
    importlib.reload(app.config)
    from app.config import config as new_config
    assert new_config.DB_NAME == "test_db"