"""Ollama/Mistral client for answer generation."""

import requests

from app.config import config


class LLMClient:
    def __init__(self, model: str = config.OLLAMA_MODEL, url: str = config.OLLAMA_URL):
        self.model = model
        self.url   = url

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            return "No relevant context found to answer the question."

        context = "\n\n---\n".join(
            f"Source {i + 1}:\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
- If the context contains the answer, provide a clear, concise answer.
- If the context does NOT contain the answer, say "I cannot find this information in the provided documents."
- Do not make up information.

Answer:"""

        resp = requests.post(
            f"{self.url}/api/generate",
            json={
                "model":   self.model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.3},
            },
            timeout=60,
        )

        if resp.status_code == 200:
            return resp.json().get("response", "No response generated.")
        return f"Error calling Ollama: {resp.status_code}"