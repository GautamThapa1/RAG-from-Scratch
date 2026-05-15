"""Sentence-transformer wrapper for embedding generation."""

from sentence_transformers import SentenceTransformer
import torch
from app.config import config


class Embedder:
    """Generates embeddings using a sentence-transformer model."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name)
        print(f"Embedder loaded on: {self.model.device}")

    def generate(self, text: str) -> list[float]:
        """Embed a single string. Returns 384-dim vector."""
        return self.model.encode(text).tolist()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings efficiently."""
        return [emb.tolist() for emb in self.model.encode(texts)]