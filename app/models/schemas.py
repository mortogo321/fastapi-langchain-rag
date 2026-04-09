from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for RAG query requests."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask against the ingested documents.",
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of relevant document chunks to retrieve.",
    )


class QueryResponse(BaseModel):
    """Schema for RAG query responses."""

    answer: str = Field(
        ...,
        description="The generated answer based on retrieved context.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of source references used to generate the answer.",
    )


class IngestResponse(BaseModel):
    """Schema for document ingestion responses."""

    message: str = Field(
        ...,
        description="Status message describing the ingestion result.",
    )
    chunks_created: int = Field(
        ...,
        ge=0,
        description="Number of text chunks created and stored in the vector database.",
    )


class HealthResponse(BaseModel):
    """Schema for health check responses."""

    status: str
    vector_store: str
    embedding_model: str
