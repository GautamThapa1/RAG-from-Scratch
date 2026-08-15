from pydantic import BaseModel

from app.rag import HybridSearch


class AgentResult(BaseModel):
    answer: str
    sources: list[dict]

class Agent:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder

    def run(self, question, top_k, top_n):

        chunks = HybridSearch(
            question,
            self.embedder,
            top_k
        ).search(top_n)

        answer = self.llm.generate(question, chunks)

        # formatting
        return AgentResult(
            answer=answer,
            sources=[
                {
                    "content": c["content"][:300],
                    "full_content": c["content"],
                    "score": c.get("rerank_score", c.get("rrf_score")),
                    "document_id": c["document_id"],
                    "chunk_index": c["chunk_index"],
                    "page_number": c["page_number"],
                }
                for c in chunks
            ],
        )