class Config:
    # Database
    DB_NAME     = "rag_db"
    DB_USER     = "postgres"
    DB_PASSWORD = "rag_password_123"
    DB_HOST     = "localhost"

    # File upload
    UPLOAD_DIR = "uploads"

    # Chunking
    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 50

    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Ollama
    OLLAMA_MODEL = "mistral"
    OLLAMA_URL   = "http://localhost:11434"

    # Retrieval
    TOP_K = 5


config = Config()