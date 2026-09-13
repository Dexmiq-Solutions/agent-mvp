from functools import lru_cache
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Configuration
    APP_NAME: str = "Agent MVP"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Integration Placeholders (Optional in Phase 1)

    # Supabase (PostgreSQL & Storage)
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    SUPABASE_STORAGE_BUCKET: str = "documents"
    DATABASE_URL: str | None = None

    # Database Connection Pool Settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False

    @property
    def supabase_storage_key(self) -> str | None:
        """Return the preferred key for Supabase Storage operations.
        
        Prefers the service role secret key for backend server operations, falling back to SUPABASE_KEY.
        """
        return self.SUPABASE_SERVICE_ROLE_KEY or self.SUPABASE_KEY

    @property
    def async_database_url(self) -> str | None:
        """Return the database URL formatted for asynchronous SQLAlchemy access with asyncpg.
        
        Normalizes standard 'postgresql://' or 'postgres://' connection schemes to 'postgresql+asyncpg://'.
        """
        if not self.DATABASE_URL:
            return None
        url = self.DATABASE_URL.strip()
        if url.startswith("postgres://"):
            return "postgresql+asyncpg://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url



    # Qdrant Vector Store & Indexing
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "document_chunks"
    QDRANT_TIMEOUT: int = 30
    QDRANT_VECTOR_SIZE: int = 1024
    QDRANT_BATCH_SIZE: int = 64
    QDRANT_DISTANCE: str = "Cosine"
    QDRANT_MAX_RETRIES: int = 3
    QDRANT_RETRY_DELAY: float = 0.5
    QDRANT_RETRY_BACKOFF: float = 2.0

    # Embeddings (Voyage)
    VOYAGE_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "voyage-4"

    # Redis Cache (Shared embedding result cache)
    REDIS_URL: str | None = None
    EMBEDDING_CACHE_ENABLED: bool = True

    # LLM Provider
    OPENAI_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o"

    # Chunking Configuration
    CHUNK_MAX_SIZE: int = 1000
    CHUNK_MIN_SIZE: int = 50
    CHUNK_OVERLAP: int = 0

    # Contextual Enrichment Configuration
    CONTEXTUAL_ENRICHMENT_ENABLED: bool = False
    CONTEXTUAL_ENRICHMENT_STRATEGY: str = "structured"

    # Retrieval & Query Configuration
    MAX_QUERY_LENGTH: int = 2000
    QUERY_TRANSFORMATION_ENABLED: bool = True
    QUERY_TRANSFORMATION_STRATEGY: str = "llm_rewrite"
    QUERY_TRANSFORMATION_MODEL: str | None = None
    QUERY_TRANSFORMATION_TIMEOUT: float = 5.0
    QUERY_TRANSFORMATION_MAX_RETRIES: int = 1
    QUERY_TRANSFORMATION_MAX_FALLBACK_ATTEMPTS: int = 1
    QUERY_TRANSFORMATION_BASE_URL: str | None = None
    VECTOR_SEARCH_TOP_K: int = 10
    VECTOR_SEARCH_SCORE_THRESHOLD: float | None = None
    SPARSE_INDEXING_ENABLED: bool = True
    SPARSE_ENCODER_STRATEGY: str = "technical_hash"
    SPARSE_ENCODER_VERSION: str = "1.0"
    SPARSE_VECTOR_NAME: str = "sparse"
    KEYWORD_SEARCH_TOP_K: int = 10
    KEYWORD_SEARCH_SCORE_THRESHOLD: float | None = None
    FUSION_RRF_K: int = 60
    FUSION_TOP_K: int = 10
    METADATA_FILTERING_ENABLED: bool = True
    METADATA_FILTERING_STRICT_MODE: bool = False

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
        if isinstance(v, str):
            normalized = v.upper().strip()
            if normalized == "WARN":
                normalized = "WARNING"
            if normalized not in valid_levels:
                raise ValueError(
                    f"Invalid LOG_LEVEL '{v}'. Must be one of: {', '.join(sorted(valid_levels))}"
                )
            return normalized
        raise ValueError(f"LOG_LEVEL must be a string, got {type(v).__name__}")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance (Single Source of Truth)."""
    return Settings()
