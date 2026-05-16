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

        # Extract just the text for embedding
        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.embedder.generate_batch(chunk_texts)

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO documents (filename, title, total_chunks)
                VALUES (%s, %s, %s) RETURNING id""",
                (filename, filename, len(chunks)),
            )
            doc_id = cur.fetchone()[0]

            for i, chunk in enumerate(chunks):
                cur.execute(
                    """INSERT INTO document_chunks
                        (document_id, chunk_index, content, embedding, page_number)
                    VALUES (%s, %s, %s, %s::vector, %s)""",
                    (doc_id, i, chunk["content"], _vec_str(embeddings[i]), chunk["page_number"]),
                )

        print(f"✅ Ingested '{filename}' — doc_id={doc_id}, {len(chunks)} chunks")
        return doc_id, len(chunks)

# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridSearch:
    def __init__(self, query:str, embedder: Embedder, k=config.TOP_K):
        self.query = query
        self.embedder = embedder
        self.k = k
    
    def semantic_search(self) -> list[tuple]:

        embedded_query = _vec_str(self.embedder.generate(self.query))

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content,
                    document_id,
                    chunk_index,
                    page_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER by similarity DESC
                LIMIT %s""",
                (embedded_query, self.k),
            )
            rows = cur.fetchall()
            return rows
    
    def keyword_search(self):

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 
                    content,
                    document_id,
                    chunk_index,
                    page_number,
                    ts_rank(content_tsv, plainto_tsquery('english', %s)) AS text_score
                FROM document_chunks
                WHERE 
                    content_tsv @@ plainto_tsquery('english', %s)
                ORDER by text_score DESC
                LIMIT %s""",
                (self.query, self.query, self.k),
            )
            rows = cur.fetchall()
            return rows
    
    def rrf_fusion(self)->list[dict]:

        semantic = self.semantic_search()
        keyword = self.keyword_search()

        scores = {}
        k_rrf = config.RRF_K

        # for semantic search
        for rank, row in enumerate(semantic,start=1):
            key = (row[1], row[2])
            scores[key] = scores.get(key, 0) + 1 / (k_rrf + rank)
            # scores.get(key, 0) checks if we seen already, core of RRF fusion
        
        # for keyword search
        for rank, row in enumerate(keyword, start=1):
            key = (row[1], row[2])
            scores[key] = scores.get(key, 0) + 1 / (k_rrf + rank)
        
        best = sorted(scores, key=scores.get, reverse=True)[:self.k]
        
        # this is a lookup table, returns all values and match with document_id
        # and chunk_index
        lookup = {(r[1], r[2]): r for r in semantic + keyword}

        return[
            {
                "content":          lookup[k][0],
                "document_id":      lookup[k][1],
                "chunk_index":      lookup[k][2],
                "page_number":      lookup[k][3],
                "similarity_score": round(scores[k], 3),
            }
            for k in best
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