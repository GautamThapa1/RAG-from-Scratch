"""Sentence-transformer wrapper for embedding generation."""

import torch
from sentence_transformers import SentenceTransformer

from app.config import config


class Embedder:
    """Generates embeddings using a sentence-transformer model."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = SentenceTransformer(model_name, device=device)
        self._batch = config.EMBEDDING_BATCH_SIZE
        print(f"Embedder loaded on: {self.model.device}")

    def generate(self, text: str) -> list[float]:
        """Embed a single string → 384-dim vector."""
        return self.model.encode(text, show_progress_bar=False).tolist()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in batches → list of vectors."""
        return [
            v.tolist()
            for v in self.model.encode(
                texts,
                batch_size=self._batch,
                show_progress_bar=len(texts) > self._batch,
            )
        ]