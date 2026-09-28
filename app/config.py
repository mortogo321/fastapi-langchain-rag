from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for embeddings/LLM")
    CHROMA_DB_PATH: str = "./chroma_db"
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=8000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=2000)
    MODEL_NAME: str = "gpt-4o-mini"
    MAX_UPLOAD_BYTES: int = Field(default=10 * 1024 * 1024, ge=1024)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1, le=1000)


@lru_cache
def get_settings() -> Settings:
    """Cached settings factory (lru_cache avoids re-parsing env per request)."""
    return Settings()
