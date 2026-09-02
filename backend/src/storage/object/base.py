"""Abstract base class for object storage operations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from storage.object.models import StorageObjectMetadata


class BaseObjectStorage(ABC):
    """Abstract interface for object storage providers.
    
    Provides standardized methods for storing, retrieving, and managing original source files
    in object storage without coupling the application layer to specific vendor SDKs.
    """

    @property
    @abstractmethod
    def bucket_name(self) -> str:
        """Return the name of the active storage bucket."""

    @abstractmethod
    async def download(self, path: str) -> bytes:
        """Download an object's contents from storage as bytes.
        
        Args:
            path: Relative storage path of the file to download.
            
        Returns:
            Raw bytes content of the object.
            
        Raises:
            ObjectNotFoundError: If the object does not exist.
            StorageDownloadError: If the download operation fails.
        """

    @abstractmethod
    async def upload(
        self,
        path: str,
        data: bytes | BinaryIO | str | Path,
        content_type: str | None = None,
        upsert: bool = False,
    ) -> StorageObjectMetadata:
        """Upload a file or byte stream to object storage.
        
        Args:
            path: Target storage path for the object.
            data: File contents as bytes, BinaryIO stream, or a local file path.
            content_type: Optional MIME type. If not provided, inferred from file extension.
            upsert: Whether to overwrite if the object already exists. Defaults to False.
            
        Returns:
            StorageObjectMetadata containing details of the uploaded object.
            
        Raises:
            StorageUploadError: If the upload operation fails.
        """

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete an object from storage.
        
        Args:
            path: Storage path of the object to delete.
            
        Returns:
            True if the object was deleted, False if it did not exist.
            
        Raises:
            StorageDeleteError: If the delete operation fails.
        """

    @abstractmethod
    async def delete_many(self, paths: list[str]) -> list[str]:
        """Delete multiple objects from storage.
        
        Args:
            paths: List of storage paths to delete.
            
        Returns:
            List of successfully deleted object paths.
            
        Raises:
            StorageDeleteError: If the delete operation fails.
        """

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[StorageObjectMetadata]:
        """List objects in the storage bucket.
        
        Args:
            prefix: Optional path prefix to filter objects.
            limit: Maximum number of objects to return.
            offset: Pagination offset.
            
        Returns:
            List of StorageObjectMetadata instances.
            
        Raises:
            StorageError: If the listing operation fails.
        """

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if an object exists at the specified storage path.
        
        Args:
            path: Storage path to check.
            
        Returns:
            True if the object exists, False otherwise.
        """

    @abstractmethod
    async def get_metadata(self, path: str) -> StorageObjectMetadata:
        """Retrieve metadata for an object in storage.
        
        Args:
            path: Storage path of the object.
            
        Returns:
            StorageObjectMetadata for the object.
            
        Raises:
            ObjectNotFoundError: If the object does not exist.
            StorageError: If retrieving metadata fails.
        """

    @abstractmethod
    async def ensure_bucket_exists(self) -> bool:
        """Verify that the configured bucket exists and is accessible, creating it idempotently if needed.
        
        Returns:
            True if the bucket exists or was created successfully.
            
        Raises:
            BucketNotFoundError: If the bucket cannot be accessed or created.
            StorageAuthenticationError: If access is unauthorized.
            StorageConnectionError: If connection fails.
        """

    @abstractmethod
    async def create_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Generate a temporary pre-signed URL for accessing a private object.
        
        Args:
            path: Storage path of the object.
            expires_in: Expiration time in seconds (default: 3600 / 1 hour).
            
        Returns:
            Pre-signed URL string.
            
        Raises:
            ObjectNotFoundError: If the object does not exist.
            StorageError: If URL generation fails.
        """
