import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.request_logger import RequestLoggerMiddleware
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
    version="1.1.0",
    lifespan=lifespan,
)

# NOTE: wildcard origins cannot be combined with credentials — keep
# allow_credentials=False while allowing all origins for local demo use.
# Set explicit origins via reverse proxy in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
# Rate limit is enforced from settings at request time via app.state;
# default 60/min keeps the middleware stateless at import time.
app.add_middleware(RateLimiterMiddleware, requests_per_minute=60)


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

    # Sanitize filename to prevent path traversal in metadata/logs.
    safe_filename = os.path.basename(file.filename)
    extension = os.path.splitext(safe_filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {allowed}",
        )

    tmp_path: str | None = None
    try:
        content = await file.read()
        settings = get_settings()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(f"File too large: {len(content)} bytes > {settings.MAX_UPLOAD_BYTES} max."),
            )
        # Save uploaded file to a temp location preserving extension
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=None,
        ) as tmp_file:
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
            chunk.metadata["source"] = safe_filename
            if "page" in chunk.metadata:
                page_num = int(chunk.metadata["page"]) + 1
                chunk.metadata["source_detail"] = f"{safe_filename} (page {page_num})"
            else:
                chunk.metadata["source_detail"] = safe_filename

        vector_store: VectorStoreService = app.state.vector_store
        vector_store.add_documents(chunks)

        return IngestResponse(
            message=f"Successfully ingested {safe_filename}",
            chunks_created=len(chunks),
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during ingestion: {exc}",
        ) from exc
    finally:
        # Clean up temp file
        if tmp_path and await anyio.Path(tmp_path).exists():
            await anyio.Path(tmp_path).unlink()


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
        answer = result["answer"]
        sources = result["sources"]
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ValueError("Invalid RAG service response shape.")
        return QueryResponse(
            answer=answer,
            sources=[str(s) for s in sources],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {exc}",
        ) from exc
