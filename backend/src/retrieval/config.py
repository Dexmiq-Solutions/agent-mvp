"""Configuration options for query preprocessing and query transformation."""

from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class QueryPreprocessingConfig:
    """Configuration settings for deterministic query preprocessing.

    Attributes:
        max_query_length: Maximum allowed query character length before validation failure.
        unicode_form: Standard Unicode normalization form (default: "NFC").
    """

    max_query_length: int = 2000
    unicode_form: str = "NFC"

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "QueryPreprocessingConfig":
        """Construct configuration from application Settings."""
        resolved = settings or get_settings()
        return cls(
            max_query_length=getattr(resolved, "MAX_QUERY_LENGTH", 2000),
        )


@dataclass(frozen=True)
class QueryTransformationConfig:
    """Configuration settings for adaptive query transformation.

    Consumer dataclass mapping directly from authoritative application Settings.
    """

    enabled: bool = True
    strategy: str = "llm_rewrite"
    model: str = "gpt-4o"
    max_query_length: int = 2000
    timeout_seconds: float = 5.0
    max_retries: int = 1
    retry_delay: float = 0.5
    max_fallback_attempts: int = 1
    preserve_identifiers: bool = True
    temperature: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "QueryTransformationConfig":
        """Construct configuration from application Settings (Single Source of Truth)."""
        resolved = settings or get_settings()
        model_name = resolved.QUERY_TRANSFORMATION_MODEL or resolved.LLM_MODEL
        return cls(
            enabled=resolved.QUERY_TRANSFORMATION_ENABLED,
            strategy=resolved.QUERY_TRANSFORMATION_STRATEGY,
            model=model_name,
            max_query_length=resolved.MAX_QUERY_LENGTH,
            timeout_seconds=resolved.QUERY_TRANSFORMATION_TIMEOUT,
            max_retries=resolved.QUERY_TRANSFORMATION_MAX_RETRIES,
            max_fallback_attempts=resolved.QUERY_TRANSFORMATION_MAX_FALLBACK_ATTEMPTS,
            api_key=resolved.OPENAI_API_KEY,
            base_url=resolved.QUERY_TRANSFORMATION_BASE_URL,
        )

