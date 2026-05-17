"""PDF text extraction and character-based chunking (word-boundary safe)."""

import fitz  # PyMuPDF

from app.config import config


class PDFProcessor:
    """Extracts text from PDFs and splits it into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap:    int = config.CHUNK_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap    = overlap

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """
        Slide a window of `chunk_size` chars across *text*, snapping BOTH
        edges to word boundaries so chunks never start or end mid-word.

        The key fix: next `start` is derived from the actual snapped `end`
        minus the overlap — not from the original unsnapped step — so drift
        never accumulates across chunks.
        """
        chunks = []
        start  = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Snap right edge back to the nearest space
                snap = text.rfind(" ", start, end)
                if snap > start:
                    end = snap

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Next chunk begins (end - overlap) chars in, snapped forward to a
            # word boundary so the LEFT edge is also always clean.
            next_start = end - self.overlap
            snap_fwd   = text.find(" ", next_start)
            start      = (snap_fwd + 1) if 0 < snap_fwd < end else next_start

        return chunks

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> list[dict]:
        """Return list of {'content': str, 'page_number': int}."""
        results = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                for chunk in self._chunk_text(text):
                    results.append({"content": chunk, "page_number": page_num})
        return results