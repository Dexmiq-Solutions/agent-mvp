"""Application domain exceptions."""

from exceptions.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseSessionError,
)
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
)
from exceptions.storage import (
    BucketNotFoundError,
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageBucketError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageUploadError,
)
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

__all__ = [
    # Storage Exceptions
    "StorageError",
    "StorageConfigurationError",
    "StorageConnectionError",
    "StorageAuthenticationError",
    "BucketNotFoundError",
    "ObjectNotFoundError",
    "StorageUploadError",
    "StorageDownloadError",
    "StorageDeleteError",
    "StorageBucketError",
    # Database Exceptions
    "DatabaseError",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseSessionError",
    # Embedding Exceptions
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
    "EmbeddingInputValidationError",
    "EmbeddingRequestError",
    # Vector Store Exceptions
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
