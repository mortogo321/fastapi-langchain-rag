"""Tests for the LCEL-style RAGService (langchain>=1.x, no RetrievalQA)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.config import Settings
from app.services.rag_chain import RAGService


def _settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="sk-fake-test-key-1234567890",
        CHROMA_DB_PATH="./test_chroma_db",
        MODEL_NAME="gpt-4o-mini",
    )


def _service_with_mocks(
    docs: list[Document],
    answer: str = "Grounded answer.",
) -> tuple[RAGService, MagicMock]:
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = docs
    with patch("app.services.rag_chain.ChatOpenAI") as mock_llm_cls:
        service = RAGService(_settings(), mock_store)
        mock_llm = mock_llm_cls.return_value
        mock_msg = MagicMock()
        mock_msg.content = answer
        mock_llm.ainvoke = AsyncMock(return_value=mock_msg)
        # rebind for mypy: service._llm is the mock instance
        service._llm = mock_llm  # type: ignore[assignment]
        return service, mock_store


async def test_query_returns_answer_and_deduped_sources() -> None:
    docs = [
        Document(
            page_content="Chunk A", metadata={"source": "a.pdf", "source_detail": "a.pdf (page 1)"}
        ),
        Document(
            page_content="Chunk B", metadata={"source": "a.pdf", "source_detail": "a.pdf (page 1)"}
        ),
        Document(page_content="Chunk C", metadata={"source": "b.txt", "source_detail": "b.txt"}),
    ]
    service, mock_store = _service_with_mocks(docs)
    result: dict[str, Any] = await service.query("What is in the docs?", k=3)

    assert result["answer"] == "Grounded answer."
    assert result["sources"] == ["a.pdf (page 1)", "b.txt"]
    mock_store.similarity_search.assert_called_once_with("What is in the docs?", k=3)


async def test_query_empty_corpus() -> None:
    service, _ = _service_with_mocks([], answer="I don't have enough information.")
    result = await service.query("Anything?", k=4)

    assert isinstance(result["answer"], str)
    assert result["sources"] == []


async def test_query_forwards_k_to_vector_store() -> None:
    docs = [Document(page_content="x", metadata={"source": "d.txt"})]
    service, mock_store = _service_with_mocks(docs)
    await service.query("q", k=7)
    mock_store.similarity_search.assert_called_once_with("q", k=7)
