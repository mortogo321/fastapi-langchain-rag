# FastAPI + LangChain RAG Pipeline with Chainlit Chat UI

[![CI](https://github.com/mortogo321/fastapi-langchain-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/mortogo321/fastapi-langchain-rag/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.x](https://img.shields.io/badge/langchain-1.x-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready Retrieval-Augmented Generation (RAG) system combining FastAPI for the API layer, LangChain 1.x (LCEL-style, no legacy RetrievalQA) for the orchestration pipeline, ChromaDB for vector storage, and Chainlit for an interactive chat interface.

## Architecture

```
                         FastAPI + LangChain RAG Pipeline
 ============================================================================

  Upload           Chunk            Embed            Store
  ------          -------          -------          ---------
 | PDF  |  --->  | Split |  --->  | OpenAI |  ---> | ChromaDB |
 | TXT  |        | Recursive      | Embeddings     | Vector   |
  ------         | CharText       |  ada-002 |      | Store   |
                 | Splitter |      -------          ---------
                  -------                               |
                                                        |
                                                        v
  Answer           LLM             Retrieve          Query
  --------       --------         ----------        --------
 | Cited  | <-- | GPT-4o | <---  | Top-K    | <-- | User   |
 | Response|    | mini   |       | Similarity|    | Question|
  --------      --------        | Search    |     --------
                                 ----------

 ============================================================================
  Interfaces:
    - REST API  (FastAPI)    :8000  ->  /api/ingest, /api/query, /api/health
    - Chat UI   (Chainlit)   :8080  ->  Interactive RAG chat
```

## Why RAG Over Fine-Tuning?

| Aspect              | RAG                                      | Fine-Tuning                          |
|---------------------|------------------------------------------|--------------------------------------|
| **Data freshness**  | Real-time; ingest new docs anytime       | Requires retraining on new data      |
| **Cost**            | Pay per query (embedding + LLM)          | High upfront training cost           |
| **Hallucination**   | Grounded in retrieved context            | Can still hallucinate confidently    |
| **Transparency**    | Sources are traceable and citable        | Black-box knowledge                  |
| **Setup time**      | Minutes to prototype                     | Hours/days to train                  |
| **Domain pivot**    | Swap document corpus instantly           | Retrain from scratch                 |

RAG is the right choice when you need **up-to-date, verifiable, source-cited answers** from your own documents without the overhead of model training.

## Quick Start

### Prerequisites

- Python 3.13 (3.12+ supported, `>=3.12,<3.14`)
- OpenAI API key
- Docker (optional, for container run)

### Local Development

```bash
# Clone and enter the project
git clone git@github.com:mortogo321/fastapi-langchain-rag.git
cd fastapi-langchain-rag

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Quality gates
make check  # ruff + mypy strict + 17 pytest

# Run FastAPI server
uvicorn app.main:app --reload --port 8000

# Run Chainlit UI (in a separate terminal)
chainlit run chainlit_app.py --port 8080
```

### Docker

```bash
# Build and run all services
docker compose up --build

# Or run individually
docker compose up api       # FastAPI on :8000
docker compose up chainlit  # Chainlit on :8080
```

## API Documentation

Once the FastAPI server is running, interactive docs are available at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

#### `POST /api/ingest`

Upload a PDF or text file to be chunked, embedded, and stored in the vector database.

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "message": "Successfully ingested document.pdf",
  "chunks_created": 42
}
```

#### `POST /api/query`

Query the knowledge base with a natural language question.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "k": 4}'
```

**Response:**
```json
{
  "answer": "The document primarily discusses...",
  "sources": ["document.pdf (page 3)", "document.pdf (page 7)"]
}
```

#### `GET /api/health`

Health check endpoint.

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store": "chromadb",
  "embedding_model": "text-embedding-ada-002"
}
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app (lifespan, CORS, middleware, upload guards)
│   ├── config.py               # Pydantic-settings config (cached)
│   ├── middleware/             # Error handler, rate limiter, request logger
│   │   ├── __init__.py
│   │   ├── error_handler.py
│   │   ├── rate_limiter.py
│   │   └── request_logger.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Request/response Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── document_loader.py  # PDF and text file loading
│       ├── text_splitter.py    # Recursive character text splitting
│       ├── vector_store.py     # ChromaDB vector store operations
│       └── rag_chain.py        # LCEL RAG service (retrieve → prompt → LLM)
├── tests/                      # pytest suite (17 tests: API, splitter, store, RAG, security)
├── chainlit_app.py             # Chainlit interactive chat UI
├── chainlit.md                 # Chainlit welcome message
├── Dockerfile                  # Pinned Python 3.13 multi-stage, non-root, HEALTHCHECK
├── docker-compose.yml          # API + Chainlit orchestration
├── pyproject.toml              # PEP 621 packaging + ruff + mypy strict + pytest
├── requirements.txt            # Pinned runtime deps (mirrors pyproject)
├── requirements-dev.txt        # Pinned dev deps
├── Makefile                    # lint / test / typecheck / build shortcuts
├── .dockerignore
├── .env.example                # Environment variable template
└── .github/workflows/ci.yml    # Split lint + test + docker CI
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

make check   # ruff + mypy strict + pytest
make build   # docker build
```

## Configuration

All settings are managed via environment variables (see `.env.example`):

| Variable                 | Default         | Description                        |
|--------------------------|-----------------|------------------------------------|
| `OPENAI_API_KEY`         | (required)      | OpenAI API key for embeddings/LLM  |
| `CHROMA_DB_PATH`         | `./chroma_db`   | ChromaDB persistence directory     |
| `CHUNK_SIZE`             | `1000`          | Characters per text chunk          |
| `CHUNK_OVERLAP`          | `200`           | Overlap between consecutive chunks |
| `MODEL_NAME`             | `gpt-4o-mini`   | OpenAI model for answer generation |
| `MAX_UPLOAD_BYTES`       | `10485760`      | Max ingest upload size (10 MB)     |
| `RATE_LIMIT_PER_MINUTE`  | `60`            | Per-IP request budget              |

## Security notes

- Filenames are sanitized to basename (path-traversal safe); only `.pdf`/`.txt` accepted.
- Uploads rejected when empty or over `MAX_UPLOAD_BYTES` (413).
- CORS uses wildcard origins with `allow_credentials=False` (valid combo); tighten origins behind a reverse proxy in production.
- In-memory per-IP rate limiter (60/min default), structured request logging, and JSON error handler are wired in `app/main.py`.
- Non-root `appuser` Docker runtime with `HEALTHCHECK` on `/api/health`.

## License

MIT — see [LICENSE](LICENSE).
