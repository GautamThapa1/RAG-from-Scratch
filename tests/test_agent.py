from app.agent import Agent, AgentResult


def test_agent_run(mocker):
    # Mock the llm to return a fixed answer
    mock_llm = mocker.MagicMock()
    mock_llm.generate.return_value = "This is a fake answer."

    # Mocker embedder
    mock_embedder = mocker.MagicMock()

    # Mock HybridSearch.search to return fake chunks
    fake_chunks = [
        {
            "content": "Fake context",
            "document_id": 1,
            "chunk_index": 0,
            "page_number": 1,
            "rerank_score": 0.95,
        }
    ]
    mocker.patch("app.agent.HybridSearch.search", return_value=fake_chunks)

    agent = Agent(mock_llm, mock_embedder)
    result = agent.run("What is RAG?", top_k=10, top_n=5)

    assert isinstance(result, AgentResult)
    assert result.answer == "This is a fake answer."
    assert len(result.sources) == 1
    assert result.sources[0]["content"] == "Fake context"
    # Check that llm.generate was called with the right arguments
    mock_llm.generate.assert_called_once_with("What is RAG?", fake_chunks)
