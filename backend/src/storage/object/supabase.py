"""Supabase Storage implementation of the ObjectStorage interface."""

import mimetypes
from pathlib import Path
from typing import Any, BinaryIO, Optional

from storage3.exceptions import StorageApiError, StorageException
from supabase import AsyncClient

from core.config import Settings, get_settings
from observability.logging import get_logger
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
from storage.object.client import get_async_supabase_client
from storage.object.models import StorageObjectMetadata

logger = get_logger(__name__)


class SupabaseObjectStorage(BaseObjectStorage):
    """Supabase Storage implementation for managing source documents.
    
    Provides asynchronous file operations against a configured Supabase Storage bucket
    using the official asynchronous Supabase Python SDK.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Supabase Object Storage client.
        
        Args:
            bucket_name: Target bucket name. Defaults to settings.SUPABASE_STORAGE_BUCKET.
            client: Pre-configured AsyncClient instance. If None, resolves from client provider.
            settings: Settings instance. Defaults to application settings.
        """
        self._settings = settings or get_settings()
        self._bucket_name = bucket_name or self._settings.SUPABASE_STORAGE_BUCKET
        if not self._bucket_name:
            raise StorageConfigurationError(
                "Storage bucket name must be provided either directly or via SUPABASE_STORAGE_BUCKET."
            )
        self._client = client

    @property
    def bucket_name(self) -> str:
        """Return the active bucket name."""
        return self._bucket_name

    # --------------------------------------------------------------------------
    # Synchronous Helpers (Pure Local Computation)
    # --------------------------------------------------------------------------

    def _normalize_path(self, path: str) -> str:
        """Normalize and validate a storage path synchronously.
        
        Strips leading/trailing whitespace and slashes, and converts backslashes to forward slashes.
        """
        if not path or not path.strip():
            raise StorageError("Storage path cannot be empty.")
        clean_path = path.strip().replace("\\", "/").strip("/")
        if not clean_path:
            raise StorageError("Invalid storage path.")
        return clean_path

    def _guess_content_type(self, path: str, default: str = "application/octet-stream") -> str:
        """Guess MIME content type from file path extension synchronously."""
        guessed, _ = mimetypes.guess_type(path)
        return guessed or default

    def _parse_metadata_dict(
        self,
        item: dict[str, Any],
        parent_prefix: str = "",
    ) -> StorageObjectMetadata:
        """Convert a Supabase Storage item dictionary to StorageObjectMetadata synchronously."""
        name = item.get("name", "")
        item_metadata = item.get("metadata") or {}
        
        # Construct relative object path
        clean_prefix = parent_prefix.strip("/")
        object_path = f"{clean_prefix}/{name}" if clean_prefix else name

        size_bytes = item_metadata.get("size")
        if size_bytes is not None:
            try:
                size_bytes = int(size_bytes)
            except (ValueError, TypeError):
                size_bytes = None

        return StorageObjectMetadata(
            name=name,
            path=object_path,
            bucket=self._bucket_name,
            size_bytes=size_bytes,
            content_type=item_metadata.get("mimetype") or self._guess_content_type(name),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            etag=item_metadata.get("eTag"),
            extra_metadata=item_metadata,
        )

    # --------------------------------------------------------------------------
    # Asynchronous I/O Client Access
    # --------------------------------------------------------------------------

    async def _get_client(self) -> AsyncClient:
        """Resolve the active asynchronous Supabase client."""
        if self._client is not None:
            return self._client
        return await get_async_supabase_client(self._settings)

    async def _get_bucket_proxy(self):
        """Get the AsyncBucketProxy for the configured bucket."""
        client = await self._get_client()
        return client.storage.from_(self._bucket_name)

    # --------------------------------------------------------------------------
    # Storage Operations (Asynchronous External I/O)
    # --------------------------------------------------------------------------

    async def download(self, path: str) -> bytes:
        """Download an object's contents from Supabase Storage as bytes."""
        clean_path = self._normalize_path(path)
        try:
            bucket_proxy = await self._get_bucket_proxy()
            content = await bucket_proxy.download(clean_path)
            logger.debug("Successfully downloaded '%s' from bucket '%s'", clean_path, self._bucket_name)
            return content
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            code = str(getattr(exc, "code", "")).lower()
            msg = str(getattr(exc, "message", "")).lower()

            if status in (404, "404") or "404" in code or "not found" in msg:
                logger.warning("Object '%s' not found in bucket '%s'", clean_path, self._bucket_name)
                raise ObjectNotFoundError(
                    f"Object '{clean_path}' not found in bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc
            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                logger.error("Authentication failed accessing Supabase Storage for '%s'", clean_path)
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            logger.error("Failed to download '%s' from bucket '%s': %s", clean_path, self._bucket_name, exc)
            raise StorageDownloadError(
                f"Failed to download object '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error downloading '%s': %s", clean_path, exc)
            raise StorageDownloadError(
                f"Unexpected error downloading '{clean_path}': {exc}",
                original_error=exc,
            ) from exc

    async def upload(
        self,
        path: str,
        data: bytes | BinaryIO | str | Path,
        content_type: str | None = None,
        upsert: bool = False,
    ) -> StorageObjectMetadata:
        """Upload an object to Supabase Storage."""
        clean_path = self._normalize_path(path)
        mime = content_type or self._guess_content_type(clean_path)

        # Prepare payload
        payload: bytes
        if isinstance(data, bytes):
            payload = data
        elif isinstance(data, (str, Path)):
            file_path = Path(data)
            if not file_path.exists() or not file_path.is_file():
                raise StorageUploadError(f"Local file '{data}' does not exist or is not a file.")
            payload = file_path.read_bytes()
        elif hasattr(data, "read"):
            payload = data.read()
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
        else:
            raise StorageUploadError(f"Unsupported data type for upload: {type(data).__name__}")

        file_options = {
            "content-type": mime,
            "upsert": "true" if upsert else "false",
        }

        try:
            bucket_proxy = await self._get_bucket_proxy()
            await bucket_proxy.upload(
                path=clean_path,
                file=payload,
                file_options=file_options,
            )
            logger.debug(
                "Successfully uploaded '%s' (%d bytes, %s) to bucket '%s'",
                clean_path,
                len(payload),
                mime,
                self._bucket_name,
            )
            filename = clean_path.split("/")[-1]
            return StorageObjectMetadata(
                name=filename,
                path=clean_path,
                bucket=self._bucket_name,
                size_bytes=len(payload),
                content_type=mime,
            )
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            code = str(getattr(exc, "code", "")).lower()
            msg = str(getattr(exc, "message", "")).lower()

            if status in (404, "404") or "bucket not found" in msg or "bucket" in code:
                logger.error("Bucket '%s' not found while uploading '%s'", self._bucket_name, clean_path)
                raise BucketNotFoundError(
                    f"Bucket '{self._bucket_name}' not found.",
                    original_error=exc,
                ) from exc
            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                logger.error("Unauthorized upload to bucket '%s'", self._bucket_name)
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            logger.error("Failed to upload '%s' to bucket '%s': %s", clean_path, self._bucket_name, exc)
            raise StorageUploadError(
                f"Failed to upload object '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error uploading '%s': %s", clean_path, exc)
            raise StorageUploadError(
                f"Unexpected error uploading '{clean_path}': {exc}",
                original_error=exc,
            ) from exc

    async def delete(self, path: str) -> bool:
        """Delete an object from Supabase Storage."""
        clean_path = self._normalize_path(path)
        try:
            bucket_proxy = await self._get_bucket_proxy()
            response = await bucket_proxy.remove([clean_path])
            logger.debug("Deleted '%s' from bucket '%s'", clean_path, self._bucket_name)
            return bool(response)
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            msg = str(getattr(exc, "message", "")).lower()

            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            logger.error("Failed to delete '%s' from bucket '%s': %s", clean_path, self._bucket_name, exc)
            raise StorageDeleteError(
                f"Failed to delete object '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error deleting '%s': %s", clean_path, exc)
            raise StorageDeleteError(
                f"Unexpected error deleting '{clean_path}': {exc}",
                original_error=exc,
            ) from exc

    async def delete_many(self, paths: list[str]) -> list[str]:
        """Delete multiple objects from Supabase Storage."""
        if not paths:
            return []
        clean_paths = [self._normalize_path(p) for p in paths]
        try:
            bucket_proxy = await self._get_bucket_proxy()
            response = await bucket_proxy.remove(clean_paths)
            logger.debug("Deleted %d objects from bucket '%s'", len(clean_paths), self._bucket_name)
            # Response contains list of deleted items
            if isinstance(response, list):
                return [item.get("name") or p for item, p in zip(response, clean_paths, strict=False)]
            return clean_paths
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            msg = str(getattr(exc, "message", "")).lower()

            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            logger.error("Failed to delete objects from bucket '%s': %s", self._bucket_name, exc)
            raise StorageDeleteError(
                f"Failed to delete objects from bucket '{self._bucket_name}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error deleting objects: %s", exc)
            raise StorageDeleteError(
                f"Unexpected error deleting objects: {exc}",
                original_error=exc,
            ) from exc

    async def list_objects(
        self,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[StorageObjectMetadata]:
        """List objects in the storage bucket matching an optional prefix."""
        clean_prefix = prefix.strip().replace("\\", "/").strip("/") if prefix else ""
        try:
            bucket_proxy = await self._get_bucket_proxy()
            options = {
                "limit": limit,
                "offset": offset,
                "sortBy": {"column": "name", "order": "asc"},
            }
            results = await bucket_proxy.list(path=clean_prefix or None, options=options)
            logger.debug("Listed %d objects under prefix '%s'", len(results), clean_prefix)
            return [
                self._parse_metadata_dict(item, parent_prefix=clean_prefix)
                for item in results
                if isinstance(item, dict) and item.get("name")
            ]
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            msg = str(getattr(exc, "message", "")).lower()

            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc
            if status in (404, "404") or "bucket not found" in msg:
                raise BucketNotFoundError(
                    f"Bucket '{self._bucket_name}' not found.",
                    original_error=exc,
                ) from exc

            logger.error("Failed to list objects in bucket '%s': %s", self._bucket_name, exc)
            raise StorageError(
                f"Failed to list objects in bucket '{self._bucket_name}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error listing objects in bucket '%s': %s", self._bucket_name, exc)
            raise StorageError(
                f"Unexpected error listing objects: {exc}",
                original_error=exc,
            ) from exc

    async def exists(self, path: str) -> bool:
        """Check if an object exists in Supabase Storage."""
        clean_path = self._normalize_path(path)
        try:
            bucket_proxy = await self._get_bucket_proxy()
            return await bucket_proxy.exists(clean_path)
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            code = str(getattr(exc, "code", "")).lower()
            msg = str(getattr(exc, "message", "")).lower()
            if status in (404, "404") or "404" in code or "not found" in msg:
                return False
            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc
            return False
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception:
            return False

    async def get_metadata(self, path: str) -> StorageObjectMetadata:
        """Retrieve metadata for a specific stored object."""
        clean_path = self._normalize_path(path)
        try:
            bucket_proxy = await self._get_bucket_proxy()
            info = await bucket_proxy.info(clean_path)
            filename = clean_path.split("/")[-1]
            parent_path = "/".join(clean_path.split("/")[:-1])
            info["name"] = filename
            return self._parse_metadata_dict(info, parent_prefix=parent_path)
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            code = str(getattr(exc, "code", "")).lower()
            msg = str(getattr(exc, "message", "")).lower()

            if status in (404, "404") or "404" in code or "not found" in msg:
                raise ObjectNotFoundError(
                    f"Object '{clean_path}' not found in bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc
            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            raise StorageError(
                f"Failed to get metadata for '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            raise StorageError(
                f"Unexpected error getting metadata for '{clean_path}': {exc}",
                original_error=exc,
            ) from exc

    async def ensure_bucket_exists(self) -> bool:
        """Verify that the configured bucket exists and is accessible, creating it idempotently if needed."""
        client = await self._get_client()
        try:
            logger.debug("Checking existence of storage bucket: '%s'", self._bucket_name)
            await client.storage.get_bucket(self._bucket_name)
            logger.info("Storage bucket '%s' is accessible", self._bucket_name)
            return True
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            code = str(getattr(exc, "code", "")).lower()
            msg = str(getattr(exc, "message", "")).lower()

            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                logger.error("Unauthorized access checking bucket '%s'", self._bucket_name)
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'. Check credentials.",
                    original_error=exc,
                ) from exc

            # Bucket not found - attempt creation
            if status in (404, "404") or "404" in code or "not found" in msg:
                try:
                    logger.info("Bucket '%s' does not exist; attempting to create it", self._bucket_name)
                    await client.storage.create_bucket(
                        id=self._bucket_name,
                        name=self._bucket_name,
                        options={"public": False},
                    )
                    logger.info("Successfully created private storage bucket: '%s'", self._bucket_name)
                    return True
                except StorageApiError as create_exc:
                    c_status = getattr(create_exc, "status", None)
                    c_msg = str(getattr(create_exc, "message", "")).lower()
                    # If already exists (race condition / 409 / 400), treat as success
                    if c_status in (400, 409, "400", "409") or "already exists" in c_msg:
                        logger.info("Storage bucket '%s' already exists", self._bucket_name)
                        return True
                    if c_status in (401, 403, "401", "403") or "unauthorized" in c_msg:
                        raise StorageAuthenticationError(
                            f"Unauthorized to create bucket '{self._bucket_name}'.",
                            original_error=create_exc,
                        ) from create_exc
                    raise StorageBucketError(
                        f"Failed to create bucket '{self._bucket_name}': {create_exc}",
                        original_error=create_exc,
                    ) from create_exc

            raise StorageBucketError(
                f"Failed to validate bucket '{self._bucket_name}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            logger.error("Unexpected error ensuring bucket '%s' exists: %s", self._bucket_name, exc)
            raise StorageBucketError(
                f"Unexpected error validating bucket '{self._bucket_name}': {exc}",
                original_error=exc,
            ) from exc

    async def create_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Generate a temporary signed URL for an object."""
        clean_path = self._normalize_path(path)
        try:
            bucket_proxy = await self._get_bucket_proxy()
            res = await bucket_proxy.create_signed_url(clean_path, expires_in=expires_in)
            signed_url = (
                res.get("signedURL")
                or res.get("signedUrl")
                if isinstance(res, dict)
                else getattr(res, "signed_url", None) or getattr(res, "signedURL", None)
            )
            if not signed_url:
                raise StorageError(f"Failed to extract signed URL from response: {res}")
            return signed_url
        except StorageApiError as exc:
            status = getattr(exc, "status", None)
            msg = str(getattr(exc, "message", "")).lower()

            if status in (404, "404") or "not found" in msg:
                raise ObjectNotFoundError(
                    f"Object '{clean_path}' not found in bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc
            if status in (401, 403, "401", "403") or "unauthorized" in msg:
                raise StorageAuthenticationError(
                    f"Unauthorized access to bucket '{self._bucket_name}'.",
                    original_error=exc,
                ) from exc

            raise StorageError(
                f"Failed to create signed URL for '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
        except (StorageConfigurationError, StorageConnectionError):
            raise
        except Exception as exc:
            raise StorageError(
                f"Unexpected error creating signed URL for '{clean_path}': {exc}",
                original_error=exc,
            ) from exc
