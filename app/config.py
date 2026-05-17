"""Central configuration — change values here, nowhere else."""

from pathlib import Path


class Config:
    # ── Database ──────────────────────────────────────────────────────────────
    DB_NAME     = "rag_db"
    DB_USER     = "postgres"
    DB_PASSWORD = "rag_password_123"
    DB_HOST     = "localhost"

    # ── File upload ───────────────────────────────────────────────────────────
    UPLOAD_DIR = "uploads"

    # ── Chunking (character-based, word-boundary snapping) ────────────────────
    CHUNK_SIZE    = 500   
    CHUNK_OVERLAP = 50   

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 64   

    # ── LLM (llama.cpp) ───────────────────────────────────────────────────────
    BASE_DIR    = Path(__file__).resolve().parent.parent
    LLAMA_MODEL = BASE_DIR / "models" / "Llama-3.2-3B-Instruct-Q6_K.gguf"
    LLM_CTX     = 4096
    LLM_TEMP    = 0.3

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K = 15   # semantic + keyword for each
    RRF_K = 60   # RRF
    TOP_N = 5    # after fusion +  rerank)

    # ── Re-ranker ─────────────────────────────────────────────────────────────
    # Set to None to disable.  Runs on CPU.
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


config = Config()