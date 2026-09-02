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

    # --------------------------------------------------------------------------
    # Application Configuration
    # --------------------------------------------------------------------------
    APP_NAME: str = "Agent MVP"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --------------------------------------------------------------------------
    # Future Integration Placeholders (Optional in Phase 1)
    # --------------------------------------------------------------------------
    # Supabase (PostgreSQL & Storage)
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    DATABASE_URL: str | None = None

    # Qdrant Vector Store
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # Embeddings (Voyage AI)
    VOYAGE_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "voyage-3-large"

    # LLM Provider
    OPENAI_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o"

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
