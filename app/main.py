import os
import shutil
import time
import asyncio
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import config
from app.database import db
from app.embedder import Embedder
from app.llm import LLMClient
from app.processor import PDFProcessor
from app.rag import DocumentManager, HybridSearch, Ingester

from fastapi.middleware.cors import CORSMiddleware

embedder  = Embedder()
processor = PDFProcessor()
ingester  = Ingester(processor, embedder)
doc_mgr   = DocumentManager()
llm       = LLMClient()

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG API")

# allow react and fastapi to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = config.TOP_K
    top_n: Optional[int] = config.TOP_N


class AskResponse(BaseModel):
    question:           str
    answer:             str
    sources:            list[dict]
    processing_time_ms: float


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id, num_chunks = ingester.ingest(file_path, file.filename)
    return {"document_id": doc_id, "chunks": num_chunks, "filename": file.filename}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    start  = time.time()
    chunks = await asyncio.to_thread(HybridSearch(req.question, embedder, req.top_k).search, req.top_n)
    answer = await asyncio.to_thread(llm.generate, req.question, chunks)

    sources = [
        {
            "content":     c["content"][:300],
            "score":       c.get("rerank_score", c.get("rrf_score")),
            "document_id": c["document_id"],
            "page_number": c["page_number"],
        }
        for c in chunks
    ]

    return AskResponse(
        question=req.question,
        answer=answer,
        sources=sources,
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