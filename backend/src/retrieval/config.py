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


@dataclass(frozen=True)
class VectorSearchConfig:
    """Configuration settings for dense vector similarity search.

    Attributes:
        top_k: Number of highest-scoring candidates to retrieve per query representation.
        score_threshold: Optional minimum similarity score threshold.
    """

    top_k: int = 10
    score_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate vector search configuration parameters."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {self.top_k}.")
        if self.score_threshold is not None:
            if not isinstance(self.score_threshold, (int, float)) or not (
                self.score_threshold == self.score_threshold and abs(self.score_threshold) != float("inf")
            ):
                raise ValueError(
                    f"score_threshold must be a finite float or None, got {self.score_threshold}."
                )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "VectorSearchConfig":
        """Construct configuration from application Settings (Single Source of Truth)."""
        resolved = settings or get_settings()
        return cls(
            top_k=getattr(resolved, "VECTOR_SEARCH_TOP_K", 10),
            score_threshold=getattr(resolved, "VECTOR_SEARCH_SCORE_THRESHOLD", None),
        )


@dataclass(frozen=True)
class KeywordSearchConfig:
    """Configuration settings for lexical/keyword sparse search.

    Attributes:
        top_k: Number of highest-scoring candidates to retrieve per query representation.
        score_threshold: Optional minimum similarity score threshold forwarded to vector store.
        sparse_vector_name: Target sparse vector configuration name in Qdrant collection.
        encoder_strategy: Strategy name identifying the sparse representation mechanism.
        encoder_version: Version identifier for the sparse representation mechanism.
    """

    top_k: int = 10
    score_threshold: Optional[float] = None
    sparse_vector_name: str = "sparse"
    encoder_strategy: str = "technical_hash"
    encoder_version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate keyword search configuration parameters."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {self.top_k}.")
        if self.score_threshold is not None:
            if not isinstance(self.score_threshold, (int, float)) or not (
                self.score_threshold == self.score_threshold and abs(self.score_threshold) != float("inf")
            ):
                raise ValueError(
                    f"score_threshold must be a finite float or None, got {self.score_threshold}."
                )
        if not self.sparse_vector_name or not self.sparse_vector_name.strip():
            raise ValueError("sparse_vector_name must be a non-empty string.")
        if not self.encoder_strategy or not self.encoder_strategy.strip():
            raise ValueError("encoder_strategy must be a non-empty string.")
        if not self.encoder_version or not self.encoder_version.strip():
            raise ValueError("encoder_version must be a non-empty string.")

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "KeywordSearchConfig":
        """Construct configuration from application Settings (Single Source of Truth)."""
        resolved = settings or get_settings()
        return cls(
            top_k=getattr(resolved, "KEYWORD_SEARCH_TOP_K", 10),
            score_threshold=getattr(resolved, "KEYWORD_SEARCH_SCORE_THRESHOLD", None),
            sparse_vector_name=getattr(resolved, "SPARSE_VECTOR_NAME", "sparse"),
            encoder_strategy=getattr(resolved, "SPARSE_ENCODER_STRATEGY", "technical_hash"),
            encoder_version=getattr(resolved, "SPARSE_ENCODER_VERSION", "1.0"),
        )


@dataclass(frozen=True)
class FusionConfig:
    """Configuration settings for hybrid retrieval fusion.

    Attributes:
        rrf_k: Reciprocal Rank Fusion smoothing constant k (default: 60).
        top_k: Optional maximum number of fused candidates to return (default: 10).
    """

    rrf_k: int = 60
    top_k: Optional[int] = 10

    def __post_init__(self) -> None:
        """Validate fusion configuration parameters."""
        if not isinstance(self.rrf_k, int) or self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {self.rrf_k}.")
        if self.top_k is not None and (not isinstance(self.top_k, int) or self.top_k <= 0):
            raise ValueError(f"top_k must be a positive integer or None, got {self.top_k}.")

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "FusionConfig":
        """Construct configuration from application Settings (Single Source of Truth)."""
        resolved = settings or get_settings()
        return cls(
            rrf_k=getattr(resolved, "FUSION_RRF_K", 60),
            top_k=getattr(resolved, "FUSION_TOP_K", 10),
        )

