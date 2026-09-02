"""Vector-storage-specific domain exceptions."""


class VectorStoreError(Exception):
    """Base exception for all vector-storage-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector store configuration or credentials are missing or invalid."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when unable to establish a connection to the vector database."""


class VectorStoreAuthenticationError(VectorStoreError):
    """Raised when authentication or authorization with the vector database fails."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when the specified vector collection is not found."""


class CollectionConfigurationError(VectorStoreError):
    """Raised when vector collection configuration (e.g. dimensions, metric) is invalid or mismatched."""


class VectorUpsertError(VectorStoreError):
    """Raised when an operation to store or update vector records fails."""


class VectorSearchError(VectorStoreError):
    """Raised when vector similarity search fails."""


class VectorDeletionError(VectorStoreError):
    """Raised when an operation to delete vector records fails."""


class VectorInputValidationError(VectorStoreError):
    """Raised when input vectors, IDs, payloads, or filters are invalid."""
