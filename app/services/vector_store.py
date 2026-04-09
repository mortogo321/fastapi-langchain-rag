from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import Settings


class VectorStoreService:
    """
    Service for managing document vectors in ChromaDB.

    Handles embedding generation via OpenAI and persistence
    through ChromaDB's local storage.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self._store = Chroma(
            collection_name="rag_documents",
            embedding_function=self._embeddings,
            persist_directory=settings.CHROMA_DB_PATH,
        )

    @property
    def retriever(self):
        """Expose the underlying vector store as a LangChain retriever."""
        return self._store.as_retriever()

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Embed and store documents in the vector database.

        Args:
            documents: List of LangChain Document objects to store.

        Returns:
            List of document IDs assigned by ChromaDB.
        """
        ids = self._store.add_documents(documents)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """
        Perform similarity search against the vector store.

        Args:
            query: The search query string.
            k: Number of most similar documents to return.

        Returns:
            List of the k most similar Document objects.
        """
        results = self._store.similarity_search(query, k=k)
        return results
