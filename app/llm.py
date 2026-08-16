from llama_cpp import Llama

from app.config import config
from app.prompts import SYSTEM_PROMPT, USER_PROMPT


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

