from llama_cpp import Llama

from app.config import config

SYSTEM_PROMPT = "You are a strict RAG assistant. Answer only from the provided context."

USER_PROMPT = """\
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

- If the question asks what the document is about, summarise from the chunks.
- If you cannot find the answer, say so. Do not make things up.

Answer:"""


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