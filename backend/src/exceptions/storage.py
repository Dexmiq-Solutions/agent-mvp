"""Storage-specific domain exceptions."""


class StorageError(Exception):
    """Base exception for all storage-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class StorageConfigurationError(StorageError):
    """Raised when storage configuration is missing or invalid."""


class StorageConnectionError(StorageError):
    """Raised when unable to connect to the storage service."""


class StorageAuthenticationError(StorageError):
    """Raised when authentication or authorization with the storage service fails."""


class BucketNotFoundError(StorageError):
    """Raised when the specified storage bucket is not found or inaccessible."""


class ObjectNotFoundError(StorageError):
    """Raised when the requested object is not found in storage."""


class StorageUploadError(StorageError):
    """Raised when uploading an object to storage fails."""


class StorageDownloadError(StorageError):
    """Raised when downloading an object from storage fails."""


class StorageDeleteError(StorageError):
    """Raised when deleting an object from storage fails."""


class StorageBucketError(StorageError):
    """Raised when an operation on a storage bucket fails."""
