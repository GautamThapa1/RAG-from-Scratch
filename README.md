# 🦙 Hybrid RAG System (Local Llama, Groq + PGVector)

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)

A Retrieval-Augmented Generation app for asking questions over your own PDFs. Answer generation can use the local Llama 3.2 model or Groq. No LangChain, no LlamaIndex — everything here is written from scratch in plain Python so I actually understand and control what's happening at every step of the pipeline.

Search is hybrid: vector similarity (pgvector) and Postgres full-text search run side by side, get merged with Reciprocal Rank Fusion, then get re-scored by a cross-encoder before the top chunks reach the selected LLM.

---

## 📑 Table of Contents
- [🎥 Demo](#-demo)
- [🤔 Why no framework?](#-why-no-framework)
- [⚙️ How it works](#️-how-it-works)
- [🛠️ Stack](#️-stack)
- [🚀 Getting it running](#-getting-it-running)
- [📡 API](#-api)
- [🎛️ Tuning knobs](#️-tuning-knobs-appconfigpy)
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
6. Top N chunks get stuffed into a prompt and handed to either the local Llama 3.2 3B GGUF model or Groq for the final answer, with page citations attached.

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
                                    Local Llama 3.2 or Groq
                                              │
                                        React frontend
```

---

## 🛠️ Stack

**Backend** — FastAPI, Postgres + pgvector, `sentence-transformers` (embedding + reranking), `llama-cpp-python` for local generation, Groq API support, and PyMuPDF for extraction.

**Frontend** — React 19 + Vite, Axios, plain CSS (no component library — didn't need one for this).

No LangChain, no LlamaIndex, no vector DB abstraction layer, no orchestration framework of any kind. Every retrieval decision in this repo is something I wrote and can point to.

---

## 🚀 Getting it running

### 📋 You'll need

- Python 3.12+
- `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node 18+
- Either Docker and Docker Compose, or a local Postgres instance with the `pgvector` extension

> **Note:** Docker only runs the Groq-backed configuration — `compose.yaml` forces `LLM_PROVIDER=groq` regardless of `.env`. Use the manual `make dev` setup (step 3) if you want to run the local Llama model.

### 1. Clone it

```bash
git clone https://github.com/GautamThapa1/RAG-from-Scratch.git
cd rag_system
```

### 2. Configure environment variables

Copy the tracked environment template, then edit `.env` with your database credentials and, when needed, your Groq API key. Never commit a real API key.

```bash
cp .env.example .env
```

```dotenv
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=rag_password_123
DB_HOST=localhost

# Choose "local" or "groq" when running with make dev.
LLM_PROVIDER=local

# Required only when LLM_PROVIDER=groq or when using Docker.
GROQ_API_KEY=your_groq_api_key
# Optional; defaults to openai/gpt-oss-20b.
GROQ_LLM_MODEL=openai/gpt-oss-20b
```

Get a Groq API key from [Groq Console](https://console.groq.com/keys). If you use `LLM_PROVIDER=local`, the key is not used.

### 3. Run locally with `make dev` (no Docker)

Docker is optional. To run the local model without Docker, install Postgres with the `pgvector` extension and complete the manual database setup in step 4 before running `make dev`. The provider is read from `.env`, so you can use either local Llama or Groq.

#### Local Llama model

Set:

```dotenv
LLM_PROVIDER=local
```

Install the local LLM dependency and download the GGUF model:

```bash
make sync
mkdir -p models
```

Download `Llama-3.2-3B-Instruct-Q6_K.gguf` from [bartowski's GGUF repo on HuggingFace](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) and place it in `models/`. Then start the API:

```bash
make dev
```

#### Groq model

Set these values instead:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_LLM_MODEL=openai/gpt-oss-20b
```

Then start the API. `make sync` is still required for the other Python dependencies, but the local GGUF model is not required when using Groq:

```bash
make sync
make dev
```

The local API runs at `http://localhost:8000`.

### 4. Set up the database manually (for non-Docker runs)

Use this step when running the API with `make dev`. Docker users can skip it because Compose starts and initializes the pgvector database automatically.

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

Credentials are configured in `.env`.

### 5. Run with Docker (optional, Groq only)

Docker starts both the pgvector database and the API server. The Docker server always uses Groq, even if `.env` contains `LLM_PROVIDER=local`, because `compose.yaml` overrides it with `LLM_PROVIDER=groq`.

Set a valid `GROQ_API_KEY` in `.env`, then run:

```bash
docker compose up --build
```

The API runs at `http://localhost:8000`. Stop the services with:

```bash
docker compose down
```

The local GGUF model is not used or required by the Docker deployment.

### 6. Run the frontend

The frontend expects the backend at `http://localhost:8000` (see [frontend/rag-app/src/services/api.js](frontend/rag-app/src/services/api.js) — update this if your API runs elsewhere).

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
| `/eval` | POST | Evaluate a document's RAG answers |

### Evaluation pipeline

The `/eval` endpoint evaluates the quality of answers generated from one indexed document. It:

1. Generates up to one natural-language question for each sampled document chunk, so `max_questions=20` can produce up to 20 questions rather than only one total.
2. Runs each question through the normal retrieval and answer-generation pipeline.
3. Uses a Groq model to score answer faithfulness and relevancy.
4. Checks whether retrieval found the source chunk used to generate each question.
5. Reports aggregate scores, context recall, and average latency.

Run an evaluation with:

```bash
curl -X POST http://localhost:8000/eval \
    -H "Content-Type: application/json" \
    -d '{"document_id": 1, "max_questions": 20}'
```

`document_id` is required and must refer to an uploaded document. `max_questions` is optional and limits the number of sampled chunks; it defaults to 30. Chunks shorter than 40 characters and questions whose generated response is not valid JSON are skipped, so the final number of evaluation questions can be lower than this limit.

Evaluation requires `GROQ_API_KEY`, even if the normal `/ask` endpoint is configured to use the local Llama model. By default, evaluation uses the `GROQ_LLM_MODEL` configured in `.env` to generate questions and judge faithfulness and relevancy, so it can make several API calls per question. You can override the evaluation models with `GROQ_EVAL_GEN_MODEL` and `GROQ_EVAL_JUDGE_MODEL`.

The response contains a `summary` and detailed per-question `results`:

```json
{
    "summary": {
        "n_questions": 20,
        "faithfulness_avg": 0.9,
        "relevancy_avg": 0.95,
        "context_recall": 0.85,
        "avg_latency_ms": 742.3
    },
    "results": []
}
```

Each result includes the generated `question`, answer, faithfulness and relevancy scores with reasons, whether the source chunk was retrieved (`context_hit`), the number of sources, and request latency.

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
