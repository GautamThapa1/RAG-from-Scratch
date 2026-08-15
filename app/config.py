import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # database
    DB_NAME     = "rag_db"
    DB_USER     = "postgres"
    DB_PASSWORD = "rag_password_123"
    DB_HOST     = "localhost"

    # uploads
    UPLOAD_DIR = "uploads"

    # chunking
    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 50

    # models
    EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 64

    BASE_DIR    = Path(__file__).resolve().parent.parent
    LLAMA_MODEL = BASE_DIR / "models" / "Llama-3.2-3B-Instruct-Q6_K.gguf"
    LLM_CTX     = 4096
    LLM_TEMP    = 0.3

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # retrieval
    TOP_K = 15   # candidates per search leg
    RRF_K = 60
    TOP_N = 5    # final chunks sent to the LLM

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


config = Config()