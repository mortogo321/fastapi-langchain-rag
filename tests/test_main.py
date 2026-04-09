from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Test the health check endpoint returns expected fields."""
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["vector_store"] == "chromadb"
    assert data["embedding_model"] == "text-embedding-ada-002"


@pytest.mark.asyncio
async def test_query_endpoint(client: AsyncClient, mock_rag_service: AsyncMock) -> None:
    """Test the query endpoint returns an answer with sources."""
    response = await client.post(
        "/api/query",
        json={"question": "What is RAG?", "k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert data["sources"] == ["test_doc.pdf (page 1)"]
    mock_rag_service.query.assert_awaited_once_with(question="What is RAG?", k=3)


@pytest.mark.asyncio
async def test_ingest_endpoint(
    client: AsyncClient,
    mock_vector_store: MagicMock,
) -> None:
    """Test the ingest endpoint processes uploaded files."""
    mock_documents = [
        MagicMock(page_content="chunk 1", metadata={"source": "test.txt"}),
        MagicMock(page_content="chunk 2", metadata={"source": "test.txt"}),
    ]

    with (
        patch("app.main.load_document", new_callable=AsyncMock) as mock_load,
        patch("app.main.split_documents") as mock_split,
    ):
        mock_load.return_value = mock_documents
        mock_split.return_value = mock_documents

        response = await client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "Successfully ingested" in data["message"]
    assert data["chunks_created"] == 2
    mock_vector_store.add_documents.assert_called_once()
