import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.models.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.services.document_loader import load_document
from app.services.rag_chain import RAGService
from app.services.text_splitter import split_documents
from app.services.vector_store import VectorStoreService

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Async lifespan context manager for FastAPI.

    Initializes shared services (vector store, RAG chain) on startup
    and performs cleanup on shutdown.
    """
    settings = get_settings()

    # Ensure ChromaDB directory exists
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

    # Initialize services and attach to app state
    vector_store_service = VectorStoreService(settings)
    rag_service = RAGService(settings, vector_store_service)

    app.state.settings = settings
    app.state.vector_store = vector_store_service
    app.state.rag_service = rag_service

    yield

    # Cleanup (if needed in the future)


app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation API powered by LangChain, ChromaDB, and OpenAI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the API and its dependencies."""
    return HealthResponse(
        status="healthy",
        vector_store="chromadb",
        embedding_model="text-embedding-ada-002",
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile) -> IngestResponse:
    """
    Ingest a document into the RAG pipeline.

    Accepts PDF (.pdf) or plain text (.txt) files. The document is:
    1. Saved to a temporary location
    2. Loaded and parsed into Document objects
    3. Split into chunks using recursive character splitting
    4. Embedded and stored in ChromaDB

    The temporary file is cleaned up after processing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    tmp_path: str | None = None
    try:
        # Save uploaded file to a temp location preserving extension
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=None,
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Load -> Split -> Store
        documents = await load_document(tmp_path)
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the uploaded file.",
            )

        chunks = split_documents(documents)

        # Override source metadata with original filename
        for chunk in chunks:
            chunk.metadata["source"] = file.filename
            if "page" in chunk.metadata:
                chunk.metadata["source_detail"] = f"{file.filename} (page {chunk.metadata['page'] + 1})"
            else:
                chunk.metadata["source_detail"] = file.filename

        vector_store: VectorStoreService = app.state.vector_store
        vector_store.add_documents(chunks)

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_created=len(chunks),
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during ingestion: {exc}",
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the ingested document corpus.

    Performs similarity search against the vector store, retrieves the
    top-k most relevant chunks, and generates an answer using the LLM.
    """
    try:
        rag_service: RAGService = app.state.rag_service
        result = await rag_service.query(
            question=request.question,
            k=request.k,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {exc}",
        )
