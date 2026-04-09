from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.vector_store import VectorStoreService

RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
Do not make up information that is not supported by the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Combines a vector store retriever with an LLM to answer questions
    grounded in the ingested document corpus.
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
        self._prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

    def _build_chain(self, k: int) -> RetrievalQA:
        """
        Build a RetrievalQA chain with the specified number of retrieved docs.

        Args:
            k: Number of documents to retrieve for context.

        Returns:
            A configured RetrievalQA chain instance.
        """
        retriever = self._vector_store_service.retriever
        retriever.search_kwargs = {"k": k}

        chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self._prompt},
        )
        return chain

    async def query(self, question: str, k: int = 4) -> dict:
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
        chain = self._build_chain(k=k)
        result = await chain.ainvoke({"query": question})

        source_docs = result.get("source_documents", [])
        sources: list[str] = []
        seen: set[str] = set()
        for doc in source_docs:
            source_ref = doc.metadata.get("source_detail", doc.metadata.get("source", "Unknown"))
            if source_ref not in seen:
                sources.append(source_ref)
                seen.add(source_ref)

        return {
            "answer": result["result"],
            "sources": sources,
        }
