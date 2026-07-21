import math

from app.config import config
from app.database import db
from app.embedder import Embedder
from app.processor import PDFProcessor


def vec_str(vec: list[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"

#just loading the re-ranker from config
def load_reranker():
    if not config.RERANKER_MODEL:
        return None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(config.RERANKER_MODEL)
        print(f"Re-ranker loaded: {config.RERANKER_MODEL}")
        return model
    except Exception as e:
        print(f"Re-ranker unavailable: {e}")
        return None

reranker = load_reranker()


class Ingester:
    def __init__(self, processor: PDFProcessor, embedder: Embedder):
        self.processor = processor
        self.embedder  = embedder

    def ingest(self, file_path: str, filename: str) -> tuple[int, int]:
        print(f"Processing '{filename}'...")
        chunks     = self.processor.process(file_path)
        embeddings = self.embedder.embed_batch([c["content"] for c in chunks])
        print(f"  {len(chunks)} chunks ready")

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO documents (filename, title, total_chunks) VALUES (%s, %s, %s) RETURNING id",
                (filename, filename, len(chunks)),
            )
            doc_id = cur.fetchone()[0]

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    "INSERT INTO document_chunks (document_id, chunk_index, content, embedding, page_number) "
                    "VALUES (%s, %s, %s, %s::vector, %s)",
                    (doc_id, i, chunk["content"], vec_str(emb), chunk["page_number"]),
                )

        print(f"Done — doc_id={doc_id}, {len(chunks)} chunks")
        return doc_id, len(chunks)


class HybridSearch:
    def __init__(self, query: str, embedder: Embedder, top_k: int = config.TOP_K):
        self.query    = query
        self.embedder = embedder
        self.top_k    = top_k

    def semantic_search(self) -> list[tuple]:
        vec = vec_str(self.embedder.embed(self.query))
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content, document_id, chunk_index, page_number,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY score ASC
                LIMIT %s
                """,
                (vec, self.top_k),
            )
            return cur.fetchall()

    def keyword_search(self) -> list[tuple]:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content, document_id, chunk_index, page_number,
                       ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM document_chunks
                WHERE content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (self.query, self.query, self.top_k),
            )
            return cur.fetchall()

# sample output for RRF
# [
#     {
#         "content": "The quick brown fox jumps over the lazy dog.",
#         "document_id": 1,
#         "chunk_index": 0,
#         "page_number": 2,
#         "rrf_score": 0.0325
#     },
#     {
#         "content": "RAG combines retrieval and generation.",
#         "document_id": 3,
#         "chunk_index": 1,
#         "page_number": 3,
#         "rrf_score": 0.0323
#     },
#     {
#         "content": "Machine learning is a subset of AI.",
#         "document_id": 2,
#         "chunk_index": 3,
#         "page_number": 1,
#         "rrf_score": 0.0161
#     }
# ]
    def rrf_fuse(self, semantic: list[tuple], keyword: list[tuple]) -> list[dict]:
        scores = {}
        rows = {}

        def update_scores(results):
            for rank, row in enumerate(results, start=1):
                key = (row[1], row[2])

                scores[key] = scores.get(key, 0) + 1 / (config.RRF_K + rank)

                if key not in rows:
                    rows[key] = row

        update_scores(semantic)
        update_scores(keyword)

        top_keys = sorted(scores, key=scores.get, reverse=True)[:self.top_k]

        return [
            {
                "content": rows[key][0],
                "document_id": rows[key][1],
                "chunk_index": rows[key][2],
                "page_number": rows[key][3],
                "rrf_score": round(scores[key], 4),
            }
            for key in top_keys
        ]
    
    def rerank(self, candidates: list[dict]) -> list[dict]:
        # reranker instance is at the top as global
        if reranker is None or not candidates:
            return candidates

        pairs  = [(self.query, c["content"]) for c in candidates]
        logits = reranker.predict(pairs) # Output: [-2.1, 3.4, -4.2, 2.8]

        for chunk, logit in zip(candidates, logits):
            # sigmoid turns raw logits into a 0-1 score that's easy to read
            # σ(x) = 1 / (1 + e^(-x))
            chunk["rerank_score"] = round(1.0 / (1.0 + math.exp(-float(logit))), 4)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    def search(self, top_n: int = config.TOP_N) -> list[dict]:
        candidates = self.rrf_fuse(self.semantic_search(), self.keyword_search())
        return self.rerank(candidates)[:top_n]


class DocumentManager:
    def list(self) -> list[tuple]:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, filename, total_chunks, created_at FROM documents ORDER BY created_at DESC")
            return cur.fetchall()

    def delete(self, doc_id: int) -> bool:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return True

    def delete_all(self):
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_chunks")
            cur.execute("DELETE FROM documents")