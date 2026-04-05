"""Application configuration from environment variables."""

# pylint: disable=no-member
# (Pydantic v2 settings fields are runtime-populated; pylint infers FieldInfo.)

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    storage_dir: Path = Field(default=Path("storage"), alias="STORAGE_DIR")

    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI, alias="EMBEDDING_PROVIDER"
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )

    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    chunk_size: int = Field(default=500, ge=50, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, ge=0, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, ge=1, le=50, alias="TOP_K")
    relevance_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0, alias="RELEVANCE_THRESHOLD"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    cors_allow_origins: str = Field(
        default="*",
        alias="CORS_ALLOW_ORIGINS",
        description="Comma-separated origins or * for all",
    )

    openai_timeout_seconds: float = Field(
        default=120.0, ge=5.0, alias="OPENAI_TIMEOUT_SECONDS"
    )

    max_ingest_bytes: int = Field(
        default=20_000_000,
        ge=1,
        alias="MAX_INGEST_BYTES",
        description="Maximum upload size for /ingest (bytes)",
    )

    health_check_llm: bool = Field(
        default=False,
        alias="HEALTH_CHECK_LLM",
        description="When true, /health probes LLM reachability (short timeout)",
    )

    api_key: str | None = Field(
        default=None,
        alias="RAG_API_KEY",
        description="When set, require Authorization: Bearer or X-API-Key on API routes",
    )

    embedding_dimension_override: int | None = Field(
        default=None,
        ge=1,
        alias="EMBEDDING_DIMENSION",
        description="Explicit embedding dimension; auto-detected if omitted",
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return self

    @model_validator(mode="after")
    def validate_openai_key(self) -> "Settings":
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                )
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return self

    @property
    def embedding_dimension(self) -> int:
        """Return explicit override or guess from model name."""
        if self.embedding_dimension_override is not None:
            return self.embedding_dimension_override
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            if "large" in self.openai_embedding_model.lower():
                return 3072
            return 1536
        return 384  # all-MiniLM-L6-v2

    def cors_origins_list(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [p.strip() for p in raw.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
