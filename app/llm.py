from llama_cpp import Llama

from app.config import config

###########

SYSTEM_PROMPT = """
You are a question-answering assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- If the context does not contain enough information, say so.
- Do not make up facts.
- Combine information from multiple context sections when appropriate.
- Cite supporting context using chunk numbers like [1] or [2].
- Keep answers clear and concise.
"""

###########

USER_PROMPT = """
Retrieved Context:

{context}

Question:
{question}

Answer using only the retrieved context:
"""



class LLMClient:
    def __init__(self):
        self.llm = Llama(
            model_path=str(config.LLAMA_MODEL),
            n_ctx=config.LLM_CTX,
            n_gpu_layers=-1,
            verbose=False,
        )

    def generate(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context found to answer the question."

        context = "\n\n---\n".join(
            f"[{i+1}] {c['content']}" for i, c in enumerate(chunks)
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT.format(context=context, question=question)},
            ],
            temperature=config.LLM_TEMP,
        )
        return response["choices"][0]["message"]["content"]

