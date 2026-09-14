"""Object storage module for managing original source files."""

from typing import Optional

from core.config import Settings
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
from storage.object.base import BaseObjectStorage
from storage.object.client import get_async_supabase_client, reset_async_supabase_client
from storage.object.models import StorageObjectMetadata
from storage.object.supabase import SupabaseObjectStorage

_default_storage_instance: Optional[BaseObjectStorage] = None


def get_object_storage(
    bucket_name: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseObjectStorage:
    """Get or create an object storage provider instance.
    
    If no custom bucket name is provided, returns a cached singleton instance configured
    with the application settings.
    
    Args:
        bucket_name: Optional custom bucket name override.
        settings: Optional Settings override.
        
    Returns:
        BaseObjectStorage implementation (SupabaseObjectStorage).
    """
    global _default_storage_instance
    if bucket_name is not None:
        return SupabaseObjectStorage(bucket_name=bucket_name, settings=settings)

    if _default_storage_instance is None:
        _default_storage_instance = SupabaseObjectStorage(settings=settings)
    return _default_storage_instance


def reset_object_storage() -> None:
    """Reset the cached default object storage instance. Useful for tests."""
    global _default_storage_instance
    _default_storage_instance = None


__all__ = [
    "BaseObjectStorage",
    "SupabaseObjectStorage",
    "StorageObjectMetadata",
    "get_object_storage",
    "reset_object_storage",
    "get_async_supabase_client",
    "reset_async_supabase_client",
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
