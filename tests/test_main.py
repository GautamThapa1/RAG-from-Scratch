from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health(mock_db):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_upload_pdf(mocker, mock_db):
    # mock ingester.ingest to avoid real processing
    mocker.patch("app.main.ingester.ingest", return_value=(1, 5))
    # mock file storage? send a fake file
    files = {"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == 1
    assert data["chunks"] == 5

def test_ask_endpoint(mocker):
    # mock agent.run to return a fake result
    fake_result = mocker.MagicMock()
    fake_result.answer = "Fake answer"
    fake_result.sources = [{"content": "fake source", "document_id": 1, "chunk_index": 0, "page_number": 1, "score": 0.9}]
    mocker.patch("app.main.agent.run", return_value=fake_result)

    payload = {"question": "What is AI?", "top_k": 10, "top_n": 3}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Fake answer"
    assert len(data["sources"]) == 1
    assert "processing_time_ms" in data

