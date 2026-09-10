from app.config import config
from app.processor import PDFProcessor


def test_chunk_basic():
    config.CHUNK_SIZE = 20
    config.CHUNK_OVERLAP = 5

    proc = PDFProcessor()
    text = "This is a test for chunking algorithm."
    chunks = proc._chunk(text)
    # Expect at least 2 chunks
    assert len(chunks) > 1
    # Check that no chunks exceed chunk_size
    assert all(len(c) <= 20 for c in chunks)
    # checking overlap
    assert len(chunks[1]) > 0


def test_process_pdf(tmp_path, mocker):
    # faking fitz.open realpdf
    mock_doc = mocker.MagicMock()
    mock_page = mocker.MagicMock()
    mock_page.get_text.return_value = "Sample page text"
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__enter__.return_value = mock_doc
    mocker.patch("fitz.open", return_value=mock_doc)

    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 5

    proc = PDFProcessor()
    file_path = tmp_path / "dummy.pdf"
    file_path.write_bytes(b"") # write 0 bytes to it
    results = proc.process(str(file_path))
    assert len(results) == 1
    assert results[0]["content"] == "Sample page text"
    assert results[0]["page_number"] == 1