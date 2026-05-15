"""FastAPI application for RAG."""

import os
import shutil
import time
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import config
from app.database import db
from app.embedder import Embedder
from app.llm import LLMClient
from app.processor import PDFProcessor
from app.rag import DocumentManager, Ingester, Retriever

# ── Singletons (loaded once at startup) ──────────────────────────────────────
_embedder  = Embedder()
_processor = PDFProcessor()
ingester   = Ingester(_processor, _embedder)
retriever  = Retriever(_embedder)
doc_mgr    = DocumentManager()
llm        = LLMClient()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG for college PDFs")
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = config.TOP_K


class AskResponse(BaseModel):
    question:           str
    answer:             str
    sources:            List[dict]
    processing_time_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    doc_id, num_chunks = ingester.ingest(file_path, file.filename)
    return {"document_id": doc_id, "chunks": num_chunks, "filename": file.filename}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    start  = time.time()
    chunks = retriever.search(request.question, top_k=request.top_k)
    answer = llm.generate(request.question, chunks)

    sources = [
        {
            "content":          c["content"][:300],
            "similarity_score": round(c["similarity_score"], 3),
            "document_id":      c["document_id"],
        }
        for c in chunks
        if c["similarity_score"] > 0.3
    ]

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.get("/documents")
async def get_documents():
    return [
        {"id": d[0], "filename": d[1], "chunks": d[2], "created_at": d[3]}
        for d in doc_mgr.list()
    ]


@app.delete("/documents/{document_id}")
async def remove_document(document_id: int):
    try:
        if not doc_mgr.delete(document_id):
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return {"message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")


@app.get("/health")
async def health():
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
        return {
            "status":         "healthy",
            "document_count": doc_count,
            "upload_dir":     config.UPLOAD_DIR,
            "ollama_model":   config.OLLAMA_MODEL,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}