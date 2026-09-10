from app.embedder import Embedder
from app.processor import PDFProcessor
from app.rag import HybridSearch, Ingester


def test_hybrid_search_semantic(mock_db, mocker):
    # mock embedder
    mock_embedder = mocker.MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    # mock database cursor to return fake rows
    fake_rows = [
        ("content1", 1, 0, 1, 0.9),
        ("content2", 2, 1, 1, 0.8),
    ]
    mock_db.fetchall.return_value = fake_rows

    hs = HybridSearch("test query", mock_embedder, top_k=2)
    results = hs.semantic_search()
    assert len(results) == 2
    # check that embed was called
    mock_embedder.embed.assert_called_once_with("test query")
    # check that SQL used the vector (we can't inspect easily, but ok)

def test_hybrid_search_rrf(mocker):
    # Test RRF fusion without db
    hs = HybridSearch("dummy", mocker.MagicMock(), top_k=2)
    semantic = [("a", 1, 0, 1, 0.9), ("b", 2, 0, 1, 0.8)]
    keyword = [("b", 2, 0, 1, 0.7), ("c", 3, 0, 1, 0.6)]
    fused = hs.rrf_fuse(semantic, keyword)
    # b should be top because it appears in both lists
    assert fused[0]["document_id"] == 2
    assert fused[0]["content"] == "b"
    # check scores sorted
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

def test_ingester(mock_db, mocker, tmp_path):
    # mock processor and embedder
    mock_processor = mocker.MagicMock(spec=PDFProcessor)
    mock_processor.process.return_value = [
        {"content": "chunk1", "page_number": 1},
        {"content": "chunk2", "page_number": 2},
    ]
    mock_embedder = mocker.MagicMock(spec=Embedder)
    mock_embedder.embed_batch.return_value = [[0.1]*384, [0.2]*384]

    # mock db cursor to return a doc_id
    mock_db.fetchone.return_value = (42,)

    ingester = Ingester(mock_processor, mock_embedder)
    file_path = tmp_path / "dummy.pdf"
    file_path.write_bytes(b"")
    doc_id, num_chunks = ingester.ingest(str(file_path), "test.pdf")

    assert doc_id == 42
    assert num_chunks == 2
    # check that insert statements were executed
    assert mock_db.execute.call_count >= 3  # insert document + 2 chunks