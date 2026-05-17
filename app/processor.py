import fitz

from app.config import config


class PDFProcessor:
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.overlap    = config.CHUNK_OVERLAP

    def _chunk(self, text: str) -> list[str]:
        chunks = []
        start  = 0

        while start < len(text):
            end = start + self.chunk_size

            # snap right edge to a word boundary
            if end < len(text):
                snap = text.rfind(" ", start, end)
                if snap > start:
                    end = snap

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # next start: go back by overlap, then snap forward to a word boundary
            # so the left edge is also always clean (fixes mid-word drift)
            next_start = end - self.overlap
            snap_fwd   = text.find(" ", next_start)
            start      = (snap_fwd + 1) if 0 < snap_fwd < end else next_start

        return chunks

    def process(self, file_path: str) -> list[dict]:
        results = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                for chunk in self._chunk(text):
                    results.append({"content": chunk, "page_number": page_num})
        return results