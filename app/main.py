import asyncio
import os
import shutil
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import Agent
from app.config import config
from app.database import db
from app.embedder import Embedder
from app.eval import run_eval, summarize
from app.processor import PDFProcessor
from app.rag import DocumentManager, Ingester

if config.LLM_PROVIDER == "groq":
    from app.llm_groq import GroqLLMClient
    llm = GroqLLMClient()
else:
    try:
        from app.llm import LLMClient
        llm = LLMClient()
    except ImportError:
        raise RuntimeError(
            "LLM_PROVIDER=local requires llama-cpp-python. "
            "Install it with: uv sync --extra local-llm"
        )
    
embedder  = Embedder()
processor = PDFProcessor()
ingester  = Ingester(processor, embedder)
doc_mgr   = DocumentManager()
agent     = Agent(llm, embedder)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = config.TOP_K
    top_n: int | None = config.TOP_N


class AskResponse(BaseModel):
    question:           str
    answer:             str
    sources:            list[dict]
    processing_time_ms: float

class EvalRequest(BaseModel):
    document_id: int
    max_questions: int | None = 30


class EvalResponse(BaseModel):
    summary: dict
    results: list[dict]


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(config.UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id, num_chunks = ingester.ingest(file_path, safe_name)
    return {"document_id": doc_id, "chunks": num_chunks, "filename": safe_name}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start  = time.time()

    # runs agent.run in background, agent.run(question, top_k, top_n) 
    # but in background
    result = await asyncio.to_thread(
        agent.run,
        req.question,
        req.top_k,
        req.top_n,
    )

    return AskResponse(
        question=req.question,
        answer=result.answer,
        sources=result.sources,
        processing_time_ms=round((time.time() - start) * 1000, 1),
    )


@app.get("/documents")
def get_documents():
    return [
        {"id": d[0], "filename": d[1], "chunks": d[2], "created_at": d[3]}
        for d in doc_mgr.list()
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    if not doc_mgr.delete(doc_id):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"message": f"Document {doc_id} deleted"}


@app.delete("/documents")
def delete_all_documents():
    doc_mgr.delete_all()
    return {"message": "All documents deleted"}


@app.get("/health")
def health():
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            count = cur.fetchone()[0]
        return {"status": "healthy", "documents": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/eval", response_model=EvalResponse)
async def eval_document(req: EvalRequest):
    # confirm doc exists before burning Groq calls on an empty chunk set
    docs = {d[0] for d in doc_mgr.list()}
    if req.document_id not in docs:
        raise HTTPException(status_code=404, detail=f"Document {req.document_id} not found")

    results = await asyncio.to_thread(run_eval, req.document_id, agent, req.max_questions)

    return EvalResponse(
        summary=summarize(results),
        results=[r.__dict__ for r in results],
    )