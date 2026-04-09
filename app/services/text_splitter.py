from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks using recursive character splitting.

    Uses the configured chunk size and overlap from application settings.
    The splitter recursively tries to split on paragraph breaks, newlines,
    spaces, and finally individual characters to produce semantically
    coherent chunks.

    Args:
        documents: List of LangChain Document objects to split.

    Returns:
        A list of smaller Document chunks with preserved metadata.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    return chunks
