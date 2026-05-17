"""
Core RAG components
  Ingester        → PDF → chunks → embeddings → DB
  HybridSearch    → query → RRF-fused results (+ optional re-rank)
  DocumentManager → CRUD on stored documents
"""

from app.config import config
from app.database import db
from app.embedder import Embedder
from app.processor import PDFProcessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec_str(vec: list[float]) -> str:
    """Float list → pgvector literal '[0.1,0.2,...]'."""
    return "[" + ",".join(map(str, vec)) + "]"


def _load_reranker():
    """Return a CrossEncoder, or None if RERANKER_MODEL is unset."""
    if not config.RERANKER_MODEL:
        return None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(config.RERANKER_MODEL)
        print(f"Re-ranker loaded: {config.RERANKER_MODEL}")
        return model
    except Exception as e:
        print(f"[warn] Re-ranker unavailable ({e}); continuing without it.")
        return None


# Module-level singleton so we pay the load cost once.
_reranker = _load_reranker()


# ── Ingester ──────────────────────────────────────────────────────────────────

class Ingester:
    def __init__(self, processor: PDFProcessor, embedder: Embedder):
        self.processor = processor
        self.embedder  = embedder

    def ingest(self, file_path: str, filename: str) -> tuple[int, int]:
        print(f"Extracting & chunking '{filename}'…")
        chunks = self.processor.process(file_path)
        print(f"  {len(chunks)} chunks — generating embeddings…")

        texts      = [c["content"] for c in chunks]
        embeddings = self.embedder.generate_batch(texts)

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO documents (filename, title, total_chunks) "
                "VALUES (%s, %s, %s) RETURNING id",
                (filename, filename, len(chunks)),
            )
            doc_id = cur.fetchone()[0]

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    "INSERT INTO document_chunks "
                    "  (document_id, chunk_index, content, embedding, page_number) "
                    "VALUES (%s, %s, %s, %s::vector, %s)",
                    (doc_id, i, chunk["content"], _vec_str(emb), chunk["page_number"]),
                )

        print(f"✅ Ingested '{filename}' — doc_id={doc_id}, {len(chunks)} chunks")
        return doc_id, len(chunks)


# ── HybridSearch ──────────────────────────────────────────────────────────────

class HybridSearch:
    """
    Combines semantic (vector) and keyword (BM25-via-tsvector) search with
    Reciprocal Rank Fusion, then optionally re-ranks the fused candidates
    with a cross-encoder before returning the final TOP_N chunks.
    """

    def __init__(self, query: str, embedder: Embedder, top_k: int = config.TOP_K):
        self.query    = query
        self.embedder = embedder
        self.top_k    = top_k

    # ── Search legs ───────────────────────────────────────────────────────────

    def _semantic(self) -> list[tuple]:
        vec = _vec_str(self.embedder.generate(self.query))
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content, document_id, chunk_index, page_number,
                       1 - (embedding <=> %s::vector) AS score
                FROM   document_chunks
                ORDER  BY score DESC
                LIMIT  %s
                """,
                (vec, self.top_k),
            )
            return cur.fetchall()

    def _keyword(self) -> list[tuple]:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT content, document_id, chunk_index, page_number,
                       ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM   document_chunks
                WHERE  content_tsv @@ plainto_tsquery('english', %s)
                ORDER  BY score DESC
                LIMIT  %s
                """,
                (self.query, self.query, self.top_k),
            )
            return cur.fetchall()

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _rrf_fuse(self, semantic: list[tuple], keyword: list[tuple]) -> list[dict]:
        """
        Merge two ranked lists with RRF.  Each row is
        (content, document_id, chunk_index, page_number, score).
        The key is (document_id, chunk_index) so duplicates are merged, not doubled.
        """
        rrf_scores: dict[tuple, float] = {}
        # Store the row for each unique chunk (first-seen wins — content is identical)
        rows: dict[tuple, tuple] = {}

        for rank, row in enumerate(semantic, start=1):
            key = (row[1], row[2])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (config.RRF_K + rank)
            rows.setdefault(key, row)

        for rank, row in enumerate(keyword, start=1):
            key = (row[1], row[2])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (config.RRF_K + rank)
            rows.setdefault(key, row)

        # Sort by fused score, keep top_k candidates
        sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[: self.top_k]

        return [
            {
                "content":          rows[k][0],
                "document_id":      rows[k][1],
                "chunk_index":      rows[k][2],
                "page_number":      rows[k][3],
                "similarity_score": round(rrf_scores[k], 4),
            }
            for k in sorted_keys
        ]

    # ── Re-rank ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def _rerank(self, candidates: list[dict]) -> list[dict]:
        """
        Score with a cross-encoder. Raw logits → sigmoid [0,1] as rerank_score.
        RRF score is preserved as rrf_score for debugging.
        """
        if _reranker is None or not candidates:
            return candidates

        pairs  = [(self.query, c["content"]) for c in candidates]
        logits = _reranker.predict(pairs)

        for chunk, logit in zip(candidates, logits):
            chunk["rrf_score"]    = chunk.pop("similarity_score")
            chunk["rerank_score"] = round(self._sigmoid(float(logit)), 4)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def search(self, top_n: int = config.TOP_N) -> list[dict]:
        """
        Return at most *top_n* chunks, best-first.
        Pipeline: semantic + keyword → RRF fusion → (optional) re-rank → slice.
        """
        candidates = self._rrf_fuse(self._semantic(), self._keyword())
        ranked     = self._rerank(candidates)
        return ranked[:top_n]


# ── DocumentManager ───────────────────────────────────────────────────────────

class DocumentManager:
    def list(self) -> list[tuple]:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, filename, total_chunks, created_at "
                "FROM documents ORDER BY created_at DESC"
            )
            return cur.fetchall()

    def delete(self, document_id: int) -> bool:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM documents WHERE id = %s", (document_id,))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            cur.execute("DELETE FROM documents      WHERE id = %s",            (document_id,))
        return True

    def delete_all(self) -> None:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_chunks")
            cur.execute("DELETE FROM documents")