# FastAPI + LangChain RAG Pipeline with Chainlit Chat UI

A production-ready Retrieval-Augmented Generation (RAG) system combining FastAPI for the API layer, LangChain for the orchestration pipeline, ChromaDB for vector storage, and Chainlit for an interactive chat interface.

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

- Python 3.12+
- OpenAI API key

### Local Development

```bash
# Clone and enter the project
git clone git@github.com:mortogo321/fastapi-langchain-rag.git
cd fastapi-langchain-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

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
│   ├── main.py                 # FastAPI application with async lifespan
│   ├── config.py               # Pydantic settings configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Request/response Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── document_loader.py  # PDF and text file loading
│       ├── text_splitter.py    # Recursive character text splitting
│       ├── vector_store.py     # ChromaDB vector store operations
│       └── rag_chain.py        # LangChain RetrievalQA chain
├── chainlit_app.py             # Chainlit interactive chat UI
├── chainlit.md                 # Chainlit welcome message
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # API + Chainlit orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore
```

## Configuration

All settings are managed via environment variables (see `.env.example`):

| Variable           | Default         | Description                        |
|--------------------|-----------------|------------------------------------|
| `OPENAI_API_KEY`   | (required)      | OpenAI API key for embeddings/LLM  |
| `CHROMA_DB_PATH`   | `./chroma_db`   | ChromaDB persistence directory     |
| `CHUNK_SIZE`       | `1000`          | Characters per text chunk          |
| `CHUNK_OVERLAP`    | `200`           | Overlap between consecutive chunks |
| `MODEL_NAME`       | `gpt-4o-mini`   | OpenAI model for answer generation |

## License

MIT
