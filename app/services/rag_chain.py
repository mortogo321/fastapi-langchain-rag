from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.vector_store import VectorStoreService

RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If the context does not contain enough information to answer the question, say "I don't have \
enough information in the provided documents to answer this question."
Do not make up information that is not supported by the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """
    Retrieval-Augmented Generation service.

    LCEL-style pipeline (langchain>=1.x): retrieve top-k chunks via the
    vector store, render them into a grounded prompt, and call the LLM.
    Replaces the removed ``langchain.chains.RetrievalQA`` API.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store_service: VectorStoreService,
    ) -> None:
        self._settings = settings
        self._vector_store_service = vector_store_service
        self._llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self._prompt: PromptTemplate = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

    def _format_sources(self, docs: list[Document]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            fallback = str(doc.metadata.get("source", "Unknown"))
            source_ref = str(doc.metadata.get("source_detail", fallback))
            if source_ref not in seen:
                sources.append(source_ref)
                seen.add(source_ref)
        return sources

    async def query(self, question: str, k: int = 4) -> dict[str, Any]:
        """
        Query the RAG pipeline with a question.

        Retrieves the top-k most relevant document chunks and generates
        an answer using the LLM.

        Args:
            question: The user's question.
            k: Number of document chunks to retrieve.

        Returns:
            A dict with 'answer' (str) and 'sources' (list[str]).
        """
        docs: list[Document] = self._vector_store_service.similarity_search(question, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_text: str = self._prompt.format(context=context, question=question)
        response = await self._llm.ainvoke(prompt_text)
        content = response.content
        answer = content if isinstance(content, str) else str(content)
        return {
            "answer": answer,
            "sources": self._format_sources(docs),
        }
