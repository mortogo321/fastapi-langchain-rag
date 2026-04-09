from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings with a fake OpenAI API key."""
    return Settings(
        OPENAI_API_KEY="sk-fake-test-key-1234567890",
        CHROMA_DB_PATH="./test_chroma_db",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=100,
        MODEL_NAME="gpt-4o-mini",
    )


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store service."""
    store = MagicMock(spec=VectorStoreService)
    store.add_documents.return_value = ["id-1", "id-2", "id-3"]
    store.similarity_search.return_value = []
    store.retriever = MagicMock()
    return store


@pytest.fixture
def mock_rag_service() -> AsyncMock:
    """Create a mock RAG service."""
    service = AsyncMock(spec=RAGService)
    service.query.return_value = {
        "answer": "This is a test answer.",
        "sources": ["test_doc.pdf (page 1)"],
    }
    return service


@pytest.fixture
async def client(mock_rag_service: AsyncMock, mock_vector_store: MagicMock) -> AsyncClient:
    """Create an async test client with mocked services."""
    from app.main import app

    app.state.settings = Settings(
        OPENAI_API_KEY="sk-fake-test-key-1234567890",
        CHROMA_DB_PATH="./test_chroma_db",
    )
    app.state.vector_store = mock_vector_store
    app.state.rag_service = mock_rag_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
