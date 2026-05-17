import torch
from sentence_transformers import SentenceTransformer

from app.config import config


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
        print(f"Embedder on: {self.model.device}")

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=len(texts) > config.EMBEDDING_BATCH_SIZE,
        )
        return [v.tolist() for v in vecs]