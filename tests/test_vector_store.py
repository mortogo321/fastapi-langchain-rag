from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


@patch("app.services.vector_store.Chroma")
@patch("app.services.vector_store.OpenAIEmbeddings")
def test_add_documents(mock_embeddings_cls, mock_chroma_cls) -> None:
    """Test that documents are added to the vector store."""
    from app.config import Settings
    from app.services.vector_store import VectorStoreService

    mock_store_instance = MagicMock()
    mock_store_instance.add_documents.return_value = ["id-1", "id-2"]
    mock_chroma_cls.return_value = mock_store_instance

    settings = Settings(
        OPENAI_API_KEY="sk-fake-test-key-1234567890",
        CHROMA_DB_PATH="./test_chroma_db",
    )
    service = VectorStoreService(settings)

    docs = [
        Document(page_content="First document", metadata={"source": "a.txt"}),
        Document(page_content="Second document", metadata={"source": "b.txt"}),
    ]
    ids = service.add_documents(docs)

    assert ids == ["id-1", "id-2"]
    mock_store_instance.add_documents.assert_called_once_with(docs)


@patch("app.services.vector_store.Chroma")
@patch("app.services.vector_store.OpenAIEmbeddings")
def test_similarity_search(mock_embeddings_cls, mock_chroma_cls) -> None:
    """Test that similarity search returns expected documents."""
    from app.config import Settings
    from app.services.vector_store import VectorStoreService

    expected_docs = [
        Document(page_content="Relevant content", metadata={"source": "doc.txt"}),
    ]
    mock_store_instance = MagicMock()
    mock_store_instance.similarity_search.return_value = expected_docs
    mock_chroma_cls.return_value = mock_store_instance

    settings = Settings(
        OPENAI_API_KEY="sk-fake-test-key-1234567890",
        CHROMA_DB_PATH="./test_chroma_db",
    )
    service = VectorStoreService(settings)

    results = service.similarity_search("What is AI?", k=2)

    assert results == expected_docs
    mock_store_instance.similarity_search.assert_called_once_with("What is AI?", k=2)
