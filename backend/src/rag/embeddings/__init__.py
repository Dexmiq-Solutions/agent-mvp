"""Embedding package providing provider abstraction, Voyage AI, and Cohere implementations."""

from typing import Optional

from core.config import Settings, get_settings
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.cache import (
    RedisEmbeddingCache,
    generate_embedding_cache_key,
    get_redis_embedding_cache,
    reset_redis_embedding_cache,
)
from rag.embeddings.client import get_async_voyage_client, reset_async_voyage_client
from rag.embeddings.cohere import CohereEmbeddingProvider
from rag.embeddings.cohere_client import get_async_cohere_client, reset_async_cohere_client
from rag.embeddings.models import EmbeddingBatchResult, EmbeddingResult
from rag.embeddings.validation import validate_embedding_batch, validate_embedding_vector
from rag.embeddings.voyage import VoyageEmbeddingProvider
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseValidationError,
)

_default_embedding_provider: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider(
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    provider: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Get or create an embedding provider instance based on application configuration.
    
    If no custom model or provider override is provided, returns a cached singleton instance
    configured with application settings.
    
    Args:
        model: Optional embedding model name override.
        settings: Optional Settings override.
        provider: Optional provider name override ('cohere' or 'voyage').
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider instance.
    """
    global _default_embedding_provider
    app_settings = settings or get_settings()
    provider_name = (provider or getattr(app_settings, "EMBEDDING_PROVIDER", "cohere")).lower()

    if model is not None or provider is not None:
        if provider_name == "voyage":
            return VoyageEmbeddingProvider(model=model, settings=app_settings)
        return CohereEmbeddingProvider(model=model, settings=app_settings)

    if _default_embedding_provider is None:
        if provider_name == "voyage":
            _default_embedding_provider = VoyageEmbeddingProvider(settings=app_settings)
        else:
            _default_embedding_provider = CohereEmbeddingProvider(settings=app_settings)
    return _default_embedding_provider


def reset_embedding_provider() -> None:
    """Reset the cached default embedding provider instance. Useful for tests."""
    global _default_embedding_provider
    _default_embedding_provider = None


__all__ = [
    # Interfaces and Models
    "BaseEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "CohereEmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingBatchResult",
    # Cache
    "RedisEmbeddingCache",
    "generate_embedding_cache_key",
    "get_redis_embedding_cache",
    "reset_redis_embedding_cache",
    # Validation
    "validate_embedding_vector",
    "validate_embedding_batch",
    # Provider Factories
    "get_embedding_provider",
    "reset_embedding_provider",
    "get_async_voyage_client",
    "reset_async_voyage_client",
    "get_async_cohere_client",
    "reset_async_cohere_client",
    # Exceptions
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
    "EmbeddingInputValidationError",
    "EmbeddingRequestError",
    "EmbeddingResponseValidationError",
]
