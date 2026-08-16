import os

import requests

from app.prompts import SYSTEM_PROMPT, USER_PROMPT

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("GROQ_LLM_MODEL", "llama-3.1-8b-instant")


class GroqLLMClient:
    def generate(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context found to answer the question."

        context = "\n\n---\n".join(
            f"[{i+1}] {c['content']}" for i, c in enumerate(chunks)
        )

        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
                ],
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]