"""Handles PDF text extraction and text chunking."""

import fitz  # PyMuPDF

from app.config import config


class PDFProcessor:
    """Extracts text from PDFs and splits it into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ── Text extraction ──────────────────────────────────────────────────────

    def extract_text(self, file_path: str) -> str:
        try:
            with fitz.open(file_path) as doc:
                return "\n".join(page.get_text() for page in doc)
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {e}") from e

    # ── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        step = self.chunk_size - self.overlap
        chunks = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # ── Combined pipeline ────────────────────────────────────────────────────

    def process(self, file_path: str) -> list[str]:
        """Extract text from PDF and return chunks."""
        text = self.extract_text(file_path)
        return self.chunk_text(text)