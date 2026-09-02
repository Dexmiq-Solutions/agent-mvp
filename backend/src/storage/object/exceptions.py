"""Storage domain exceptions re-exported for convenience."""

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

__all__ = [
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
]
