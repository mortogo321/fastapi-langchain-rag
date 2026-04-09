import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}


async def load_document(file_path: str) -> list[Document]:
    """
    Load a document from the given file path based on its extension.

    Supports PDF (.pdf) and plain text (.txt) files.
    Uses the appropriate LangChain loader for each file type.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        A list of LangChain Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise ValueError(
            f"Unsupported file extension '{extension}'. Supported: {supported}"
        )

    loader_class = SUPPORTED_EXTENSIONS[extension]
    loader = loader_class(str(path))

    # PyPDFLoader and TextLoader both support lazy_load
    documents = loader.load()

    # Ensure source metadata is set to the filename (not full path)
    for doc in documents:
        doc.metadata["source"] = path.name
        if "page" in doc.metadata:
            doc.metadata["source_detail"] = f"{path.name} (page {doc.metadata['page'] + 1})"
        else:
            doc.metadata["source_detail"] = path.name

    return documents
