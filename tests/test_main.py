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


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_type(client: AsyncClient) -> None:
    """Unsupported extensions must be rejected with 400."""
    response = await client.post(
        "/api/ingest",
        files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ingest_rejects_empty_file(client: AsyncClient) -> None:
    """Empty uploads must be rejected with 400."""
    response = await client.post(
        "/api/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ingest_rejects_oversize_file(client: AsyncClient) -> None:
    """Files over MAX_UPLOAD_BYTES must be rejected with 413."""
    with patch("app.main.get_settings") as mock_get:
        from app.config import Settings

        mock_get.return_value = Settings(
            OPENAI_API_KEY="sk-fake-test-key-1234567890",
            CHROMA_DB_PATH="./test_chroma_db",
            MAX_UPLOAD_BYTES=1024,
        )
        response = await client.post(
            "/api/ingest",
            files={"file": ("big.txt", b"x" * 1025, "text/plain")},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_ingest_sanitizes_path_traversal(
    client: AsyncClient,
    mock_vector_store: MagicMock,
) -> None:
    """Path-traversal filenames must be sanitized to basename in metadata."""
    mock_documents = [
        MagicMock(page_content="chunk", metadata={"source": "x"}),
    ]
    with (
        patch("app.main.load_document", new_callable=AsyncMock) as mock_load,
        patch("app.main.split_documents") as mock_split,
    ):
        mock_load.return_value = mock_documents
        mock_split.return_value = mock_documents
        response = await client.post(
            "/api/ingest",
            files={"file": ("../../etc/passwd.txt", b"content here", "text/plain")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "../" not in data["message"]
    assert "passwd.txt" in data["message"]


@pytest.mark.asyncio
async def test_query_validation_rejects_empty_question(client: AsyncClient) -> None:
    """Empty questions must fail request validation with 422."""
    response = await client.post("/api/query", json={"question": "", "k": 4})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_validation_rejects_out_of_range_k(client: AsyncClient) -> None:
    """k outside 1..20 must fail request validation with 422."""
    response = await client.post("/api/query", json={"question": "hi", "k": 99})
    assert response.status_code == 422
