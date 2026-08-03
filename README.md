# 🦙 Local Hybrid RAG System (Llama 3.2 + PGVector)

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)

A fully local Retrieval-Augmented Generation app for asking questions over your own PDFs. No OpenAI key, no LangChain, no LlamaIndex — everything here is written from scratch in plain Python so I actually understand and control what's happening at every step of the pipeline.

Search is hybrid: vector similarity (pgvector) and Postgres full-text search run side by side, get merged with Reciprocal Rank Fusion, then get re-scored by a cross-encoder before the top chunks ever reach the LLM. Generation happens locally with Llama 3.2 3B via `llama-cpp-python`.

---

## 📑 Table of Contents
- [🎥 Demo](Demo)
- [🤔 Why no framework?](#-why-no-framework)
- [⚙️ How it works](#%EF%B8%8F-how-it-works)
- [🛠️ Stack](#%EF%B8%8F-stack)
- [🚀 Getting it running](#-getting-it-running)
- [📡 API](#-api)
- [🎛️ Tuning knobs](#%EF%B8%8F-tuning-knobs-appconfigpy)
- [📄 License](#-license)

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/tpV_Xm-i8LE/0.jpg)](https://youtu.be/tpV_Xm-i8LE)

> Full walkthrough on YouTube

---

## 🤔 Why no framework?

I never used LangChain here — that was the point from day one. My goal wasn't to ship something fast, it was to actually learn how RAG works under the hood: chunking, embeddings, hybrid search, RRF, reranking, all of it. Frameworks come and go, but if you understand the fundamentals you can rebuild this in whatever stack is popular five years from now. Hybrid search + RRF + reranking is maybe 150 lines of actual logic — writing it directly meant I could tune it, log it, and reason about failures without fighting a framework's internals. So this project talks to `sentence-transformers`, `pgvector`, and `llama-cpp-python` directly. That's it.

---

## ⚙️ How it works

1. PDF gets uploaded and split into ~500 character chunks (50 char overlap, snapped to word boundaries) using PyMuPDF.
2. Each chunk gets embedded with `all-MiniLM-L6-v2` (384 dims) and stored in Postgres alongside a generated `tsvector` column for keyword search.
3. On a query, two searches run in parallel — cosine similarity over embeddings, and Postgres full-text search — each returning their own top-k.
4. Results get merged with Reciprocal Rank Fusion, since raw scores from the two methods aren't directly comparable but rank position is.
5. The merged candidates get reranked with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) to squeeze out the ones that actually answer the question.
6. Top N chunks get stuffed into a prompt and handed to a local Llama 3.2 3B GGUF model for the final answer, with page citations attached.

```
PDF → chunk (PyMuPDF) → embed (MiniLM) → Postgres (pgvector + tsvector)
                                              │
                                     query comes in
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                                │
              vector search (top-k)                          keyword search (top-k)
                      │                                                │
                      └───────────────────────┬───────────────────────┘
                                     Reciprocal Rank Fusion
                                              │
                                  cross-encoder rerank (top-n)
                                              │
                                   Llama 3.2 3B (local, GGUF)
                                              │
                                        React frontend
```

---

## 🛠️ Stack

**Backend** — FastAPI, Postgres + pgvector, `sentence-transformers` (embedding + reranking), `llama-cpp-python`, PyMuPDF for extraction.

**Frontend** — React 19 + Vite, Axios, plain CSS (no component library — didn't need one for this).

No LangChain, no LlamaIndex, no vector DB abstraction layer, no orchestration framework of any kind. Every retrieval decision in this repo is something I wrote and can point to.

---

## 🚀 Getting it running

### 📋 You'll need

- Python 3.12+
- `uv` (or plain pip) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node 18+
- Postgres with the `pgvector` extension available

### 1. Clone it

```bash
git clone https://github.com/YOUR_USERNAME/rag_system.git
cd rag_system
```

### 2. Set up the database

```bash
sudo -u postgres psql -c "CREATE DATABASE rag_db;"
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'rag_password_123';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE rag_db TO postgres;"
```

Then create the schema:

```bash
psql -h localhost -U postgres -d rag_db -c "
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    total_chunks INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    embedding vector(384),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv ON document_chunks USING gin (content_tsv);
"
```

Credentials are configurable in `app/config.py` if you don't want to use the defaults above.

### 3. Grab the model

```bash
mkdir -p models
```

Download `Llama-3.2-3B-Instruct-Q6_K.gguf` from [bartowski's GGUF repo on HuggingFace](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) and drop it in `models/`.

### 4. Run the backend

```bash
make sync
make dev
```

or manually:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Runs at `http://localhost:8000`.

### 5. Run the frontend

```bash
make react
```

or manually:

```bash
cd frontend/rag-app
npm install
npm run dev
```

Runs at `http://localhost:5173`.

---

## 📡 API

Swagger docs live at `http://localhost:8000/docs` once the backend is up. Quick reference:

| Endpoint | Method | What it does |
|---|---|---|
| `/upload` | POST | Upload a PDF — gets chunked, embedded, and indexed |
| `/ask` | POST | Ask a question, get an answer with sources |
| `/documents` | GET | List everything that's been ingested |
| `/documents/{id}` | DELETE | Remove one document and its chunks |
| `/documents` | DELETE | Wipe everything |
| `/health` | GET | Health check + document count |

---

## 🎛️ Tuning knobs (`app/config.py`)

| Setting | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embedding model |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2 | Cross-encoder reranker |
| `TOP_K` | 15 | Candidates pulled per search method |
| `TOP_N` | 5 | Chunks that actually make it into the prompt |
| `LLM_CTX` | 4096 | Llama context window |
| `LLM_TEMP` | 0.3 | Generation temperature |

---

## 📄 License

Contributions, issues, and feature requests are welcome!

Distributed under the **MIT License**.
