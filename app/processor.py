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


    # ── text extracting, chunking with page no ────────────────────────────────────────────────────

    def process(self, file_path: str) -> list[dict]:
        """Return list [chunks, page_number]."""
        chunks = []
        step = self.chunk_size - self.overlap
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                for start in range(0, len(text), step):
                    chunk_text = text[start: start + self.chunk_size].strip()
                    if chunk_text:
                        chunks.append({"content": chunk_text, "page_number": page_num})
        return chunks
        