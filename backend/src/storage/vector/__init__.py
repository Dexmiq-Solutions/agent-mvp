"""Vector storage package providing vector store abstraction and Qdrant implementation."""

from typing import Optional

from core.config import Settings
from exceptions.vector import (
    CollectionConfigurationError,
    CollectionNotFoundError,
    VectorDeletionError,
    VectorInputValidationError,
    VectorSearchError,
    VectorStoreAuthenticationError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from storage.vector.base import BaseVectorStore
from storage.vector.client import (
    close_async_qdrant_client,
    get_async_qdrant_client,
    reset_async_qdrant_client,
)
from storage.vector.models import VectorPayload, VectorRecord, VectorSearchResult
from storage.vector.qdrant import QdrantVectorStore

_default_vector_store: Optional[BaseVectorStore] = None


def get_vector_store(
    collection_name: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseVectorStore:
    """Get or create a vector store instance.
    
    If no custom collection override is provided, returns a cached singleton instance
    configured with application settings.
    
    Args:
        collection_name: Optional collection name override.
        settings: Optional Settings override.
        
    Returns:
        BaseVectorStore: Configured vector store instance (QdrantVectorStore).
    """
    global _default_vector_store
    if collection_name is not None:
        return QdrantVectorStore(collection_name=collection_name, settings=settings)

    if _default_vector_store is None:
        _default_vector_store = QdrantVectorStore(settings=settings)
    return _default_vector_store


def reset_vector_store() -> None:
    """Reset the cached default vector store instance. Useful for tests."""
    global _default_vector_store
    _default_vector_store = None


__all__ = [
    # Interfaces and Models
    "BaseVectorStore",
    "QdrantVectorStore",
    "VectorPayload",
    "VectorRecord",
    "VectorSearchResult",
    # Factory & Client Management
    "get_vector_store",
    "reset_vector_store",
    "get_async_qdrant_client",
    "reset_async_qdrant_client",
    "close_async_qdrant_client",
    # Exceptions
    "VectorStoreError",
    "VectorStoreConfigurationError",
    "VectorStoreConnectionError",
    "VectorStoreAuthenticationError",
    "CollectionNotFoundError",
    "CollectionConfigurationError",
    "VectorUpsertError",
    "VectorSearchError",
    "VectorDeletionError",
    "VectorInputValidationError",
]
