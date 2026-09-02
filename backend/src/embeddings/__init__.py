"""Embedding package providing provider abstraction and Voyage AI implementation."""

from typing import Optional

from app.core.config import Settings
from embeddings.base import BaseEmbeddingProvider
from embeddings.client import get_async_voyage_client, reset_async_voyage_client
from embeddings.models import EmbeddingBatchResult, EmbeddingResult
from embeddings.voyage import VoyageEmbeddingProvider
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
)

_default_embedding_provider: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider(
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseEmbeddingProvider:
    """Get or create an embedding provider instance.
    
    If no custom model override is provided, returns a cached singleton instance configured
    with application settings.
    
    Args:
        model: Optional embedding model name override.
        settings: Optional Settings override.
        
    Returns:
        BaseEmbeddingProvider: Configured embedding provider instance (VoyageEmbeddingProvider).
    """
    global _default_embedding_provider
    if model is not None:
        return VoyageEmbeddingProvider(model=model, settings=settings)

    if _default_embedding_provider is None:
        _default_embedding_provider = VoyageEmbeddingProvider(settings=settings)
    return _default_embedding_provider


def reset_embedding_provider() -> None:
    """Reset the cached default embedding provider instance. Useful for tests."""
    global _default_embedding_provider
    _default_embedding_provider = None


__all__ = [
    # Interfaces and Models
    "BaseEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingBatchResult",
    # Provider Factories
    "get_embedding_provider",
    "reset_embedding_provider",
    "get_async_voyage_client",
    "reset_async_voyage_client",
    # Exceptions
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
    "EmbeddingInputValidationError",
    "EmbeddingRequestError",
]
