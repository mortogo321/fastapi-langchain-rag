"""Chainlit chat interface for the RAG pipeline."""

import chainlit as cl

from app.config import get_settings
from app.services.rag_chain import RAGService
from app.services.vector_store import VectorStoreService


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Initialize RAG services when a new chat session starts.

    Creates the vector store service and RAG service, storing them
    in the Chainlit user session for use during message handling.
    """
    settings = get_settings()
    vector_store_service = VectorStoreService(settings)
    rag_service = RAGService(settings, vector_store_service)

    cl.user_session.set("rag_service", rag_service)

    await cl.Message(
        content=(
            "Welcome to the **RAG Knowledge Assistant**!\n\n"
            "I can answer questions based on documents that have been ingested "
            "into the knowledge base via the API.\n\n"
            "**How to use:**\n"
            "1. First, upload documents through the API endpoint `POST /api/ingest`\n"
            "2. Then ask me any question about the ingested documents\n\n"
            "Go ahead and ask a question!"
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle incoming user messages by querying the RAG pipeline.

    Retrieves relevant document chunks from ChromaDB, generates
    an answer via the LLM, and returns the response with source
    references as Chainlit Text elements.
    """
    rag_service: RAGService = cl.user_session.get("rag_service")

    if rag_service is None:
        await cl.Message(content="Session not initialized. Please refresh the page.").send()
        return

    # Show a thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    try:
        result = await rag_service.query(question=message.content, k=4)
        answer = result["answer"]
        sources = result["sources"]

        # Build source references as Text elements
        elements: list[cl.Text] = []
        if sources:
            source_text = "\n".join(f"- {src}" for src in sources)
            elements.append(
                cl.Text(
                    name="Sources",
                    content=source_text,
                    display="inline",
                )
            )

        # Format the response
        response_content = answer
        if sources:
            response_content += "\n\n---\n**Sources:**\n"
            response_content += "\n".join(f"- {src}" for src in sources)

        msg.content = response_content
        msg.elements = elements
        await msg.update()

    except Exception as exc:
        msg.content = f"An error occurred while processing your question: {exc}"
        await msg.update()
