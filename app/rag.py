"""
Core RAG components:
  - Ingester       → PDF → chunks → embeddings → DB
  - Retriever      → query → similar chunks from DB
  - DocumentManager → CRUD on stored documents
"""

from app.config import config
from app.database import db
from app.embedder import Embedder
from app.processor import PDFProcessor


def _vec_str(vec: list[float]) -> str:
    """Convert a float list to pgvector literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(map(str, vec)) + "]"


# ── Ingester ──────────────────────────────────────────────────────────────────

class Ingester:
    def __init__(self, processor: PDFProcessor, embedder: Embedder):
        self.processor = processor
        self.embedder = embedder

    def ingest(self, file_path: str, filename: str) -> tuple[int, int]:
        print(f"Extracting & chunking '{filename}'...")
        chunks = self.processor.process(file_path)
        print(f"  {len(chunks)} chunks — generating embeddings...")
        embeddings = self.embedder.generate_batch(chunks)

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO documents (filename, title, total_chunks)
                   VALUES (%s, %s, %s) RETURNING id""",
                (filename, filename, len(chunks)),
            )
            doc_id = cur.fetchone()[0]

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """INSERT INTO document_chunks
                         (document_id, chunk_index, content, embedding)
                       VALUES (%s, %s, %s, %s::vector)""",
                    (doc_id, i, chunk, _vec_str(emb)),
                )

        print(f"✅ Ingested '{filename}' — doc_id={doc_id}, {len(chunks)} chunks")
        return doc_id, len(chunks)


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = config.TOP_K):
        self.embedder = embedder
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        emb_str = _vec_str(self.embedder.generate(query))

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT dc.content, dc.document_id, dc.chunk_index,
                          dc.embedding <=> %s::vector AS distance
                   FROM document_chunks dc
                   ORDER BY distance
                   LIMIT %s""",
                (emb_str, k),          # ← fix: was passing `top_k` (None) instead of `k`
            )
            rows = cur.fetchall()

        return [
            {
                "content":          row[0],
                "document_id":      row[1],
                "chunk_index":      row[2],
                "similarity_score": 1 - row[3],
            }
            for row in rows
        ]


# ── DocumentManager ───────────────────────────────────────────────────────────

class DocumentManager:
    def list(self) -> list[tuple]:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, filename, total_chunks, created_at FROM documents ORDER BY created_at DESC"
            )
            return cur.fetchall()

    def delete(self, document_id: int) -> bool:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT filename FROM documents WHERE id = %s", (document_id,))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        return True

    def delete_all(self) -> None:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_chunks")
            cur.execute("DELETE FROM documents")