CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    filename     TEXT NOT NULL,
    title        TEXT,
    total_chunks INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id           SERIAL PRIMARY KEY,
    document_id  INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER,
    content      TEXT NOT NULL,
    embedding    vector(384),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    page_number  INTEGER,
    content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);