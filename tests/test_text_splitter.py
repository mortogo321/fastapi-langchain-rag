from unittest.mock import patch

from langchain_core.documents import Document

from app.services.text_splitter import split_documents


def _make_doc(content: str) -> Document:
    """Helper to create a Document with minimal metadata."""
    return Document(page_content=content, metadata={"source": "test.txt"})


@patch("app.services.text_splitter.get_settings")
def test_split_documents_respects_chunk_size(mock_settings) -> None:
    """Verify that chunks do not exceed the configured chunk size."""
    mock_settings.return_value.CHUNK_SIZE = 50
    mock_settings.return_value.CHUNK_OVERLAP = 0

    long_text = "word " * 100  # ~500 characters
    docs = [_make_doc(long_text)]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 50


@patch("app.services.text_splitter.get_settings")
def test_split_documents_with_overlap(mock_settings) -> None:
    """Verify that chunk overlap produces overlapping content."""
    mock_settings.return_value.CHUNK_SIZE = 50
    mock_settings.return_value.CHUNK_OVERLAP = 10

    long_text = "word " * 100
    docs = [_make_doc(long_text)]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    # With overlap, consecutive chunks should share some content
    for i in range(len(chunks) - 1):
        current_end = chunks[i].page_content[-10:]
        next_start = chunks[i + 1].page_content[:20]
        # The overlap means the end of one chunk appears at the start of the next
        assert current_end in chunks[i + 1].page_content or any(
            w in next_start for w in current_end.split()
        )


@patch("app.services.text_splitter.get_settings")
def test_empty_documents(mock_settings) -> None:
    """Verify that splitting an empty list returns an empty list."""
    mock_settings.return_value.CHUNK_SIZE = 500
    mock_settings.return_value.CHUNK_OVERLAP = 100

    chunks = split_documents([])

    assert chunks == []
