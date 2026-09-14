"""Storage-based acquisition service discovering source documents for ingestion."""

from pathlib import PurePosixPath
from typing import Any, Optional

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.acquisition.base import BaseAcquisitionService
from rag.acquisition.formats import (
    DocumentType,
    get_canonical_mime_type,
    get_document_type,
    is_supported_extension,
    normalize_extension,
)
from rag.acquisition.models import (
    AcquisitionResult,
    AcquisitionStatus,
    SourceDocument,
    SourceReference,
)
from exceptions.acquisition import (
    AcquisitionConfigurationError,
    InvalidSourceReferenceError,
    SourceDiscoveryError,
    UnsupportedDocumentError,
)
from exceptions.storage import (
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageConfigurationError,
    StorageError,
)
from storage.object import BaseObjectStorage, StorageObjectMetadata, get_object_storage

logger = get_logger(__name__)


class StorageAcquisitionService(BaseAcquisitionService):
    """Acquisition service that discovers and validates source documents from Object Storage.
    
    Reuses the existing BaseObjectStorage abstraction without direct coupling to Supabase SDKs,
    ensuring strict project isolation, zero file downloads, and O(n) scan complexity.
    """

    def __init__(
        self,
        storage: Optional[BaseObjectStorage] = None,
        settings: Optional[Settings] = None,
        batch_size: int = 100,
    ) -> None:
        """Initialize the Storage Acquisition Service.
        
        Args:
            storage: Object storage provider instance. If None, resolves from get_object_storage().
            settings: Application settings instance.
            batch_size: Page limit for listing operations from storage. Defaults to 100.
        """
        self._settings = settings or get_settings()
        self._storage = storage or get_object_storage(settings=self._settings)
        self._batch_size = max(1, batch_size)

    # --------------------------------------------------------------------------
    # Synchronous Helpers (Validation & Pure Local Computation)
    # --------------------------------------------------------------------------

    def _validate_project_id(self, project_id: Any) -> str:
        """Validate and normalize a project identifier for strict project isolation."""
        if not isinstance(project_id, str) or not project_id.strip():
            raise InvalidSourceReferenceError(
                "project_id must be a non-empty string to enforce project isolation."
            )
        clean = project_id.strip().replace("\\", "/").strip("/")
        if not clean or "/" in clean:
            # Enforce single path segment for project identifier
            if not clean:
                raise InvalidSourceReferenceError("project_id cannot be empty.")
        return clean

    def _normalize_storage_path(self, path: str) -> str:
        """Normalize a storage path string."""
        if not path or not path.strip():
            raise InvalidSourceReferenceError("Storage path cannot be empty.")
        return path.strip().replace("\\", "/").strip("/")

    def _build_project_prefix(self, project_id: str, prefix: Optional[str] = None) -> str:
        """Build the storage search prefix scoped strictly to the project namespace."""
        clean_project_id = self._validate_project_id(project_id)
        if not prefix or not prefix.strip():
            return clean_project_id
        clean_prefix = prefix.strip().replace("\\", "/").strip("/")
        return f"{clean_project_id}/{clean_prefix}"

    def _resolve_and_verify_path(self, project_id: str, storage_path: str) -> str:
        """Resolve full storage path and verify it belongs strictly to the project."""
        clean_project_id = self._validate_project_id(project_id)
        clean_path = self._normalize_storage_path(storage_path)

        project_prefix = f"{clean_project_id}/"
        if clean_path.startswith(project_prefix):
            return clean_path
        if clean_path == clean_project_id:
            raise InvalidSourceReferenceError(f"Storage path '{clean_path}' points to project root, not a document.")

        # If path does not start with project prefix, check if it references another project
        parts = clean_path.split("/")
        if len(parts) > 1 and parts[0] != clean_project_id:
            raise InvalidSourceReferenceError(
                f"Cross-project access prohibited: path '{clean_path}' does not belong to project '{clean_project_id}'."
            )

        # Prepend project namespace for unqualified relative paths
        return f"{clean_project_id}/{clean_path}"

    def _is_directory_or_placeholder(self, item: StorageObjectMetadata) -> bool:
        """Determine whether a storage item is a folder placeholder or invalid entry."""
        name = (item.name or "").strip()
        path = (item.path or "").strip()

        # Check directory indicators
        if not name or not path:
            return True
        if name in (".", "..", ".emptyFolderPlaceholder"):
            return True
        if name.endswith(".emptyFolderPlaceholder") or path.endswith(".emptyFolderPlaceholder"):
            return True
        if path.endswith("/") or path.endswith("\\") or name.endswith("/"):
            return True
        if item.content_type in ("application/x-directory", "inode/directory", "directory"):
            return True

        # Check if entry lacks any extension and has no file size or metadata
        has_extension = bool(PurePosixPath(name).suffix)
        if not has_extension and item.size_bytes is None:
            return True

        return False

    def _build_source_document(
        self,
        item: StorageObjectMetadata,
        project_id: str,
        document_type: DocumentType,
    ) -> SourceDocument:
        """Construct a validated SourceDocument domain model from storage metadata."""
        clean_path = self._normalize_storage_path(item.path)
        filename = item.name or PurePosixPath(clean_path).name
        extension = normalize_extension(filename)

        # Determine MIME type: prefer specific format MIME over generic application/octet-stream
        content_type = item.content_type
        if not content_type or content_type == "application/octet-stream":
            content_type = get_canonical_mime_type(document_type)

        etag = item.etag or item.extra_metadata.get("eTag") or item.extra_metadata.get("etag")
        source_ref = SourceReference(
            bucket=item.bucket,
            path=clean_path,
            source_type="supabase_storage",
            etag=etag,
        )

        source_id = f"{item.bucket}/{clean_path}"

        return SourceDocument(
            source_id=source_id,
            project_id=project_id,
            filename=filename,
            storage_path=clean_path,
            storage_bucket=item.bucket,
            document_type=document_type,
            content_type=content_type,
            extension=extension,
            size_bytes=item.size_bytes,
            created_at=item.created_at,
            updated_at=item.updated_at,
            metadata=dict(item.extra_metadata),
            source_ref=source_ref,
        )

    # --------------------------------------------------------------------------
    # Asynchronous Acquisition Operations
    # --------------------------------------------------------------------------

    async def acquire(
        self,
        project_id: str,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AcquisitionResult:
        """Discover and acquire all available supported source documents for a project."""
        clean_project_id = self._validate_project_id(project_id)
        search_prefix = self._build_project_prefix(clean_project_id, prefix=prefix)

        logger.info(
            "Starting document acquisition for project '%s' under prefix '%s'",
            clean_project_id,
            search_prefix,
        )

        acquired_documents: list[SourceDocument] = []
        skipped_paths: list[str] = []
        total_discovered = 0
        offset = 0

        project_prefix_guard = f"{clean_project_id}/"

        try:
            while True:
                # Determine next batch size
                batch_limit = self._batch_size
                if limit is not None:
                    remaining = limit - len(acquired_documents)
                    if remaining <= 0:
                        break
                    batch_limit = min(self._batch_size, remaining)

                logger.debug(
                    "Listing storage objects for project '%s' (prefix='%s', limit=%d, offset=%d)",
                    clean_project_id,
                    search_prefix,
                    batch_limit,
                    offset,
                )

                items = await self._storage.list_objects(
                    prefix=search_prefix,
                    limit=batch_limit,
                    offset=offset,
                )

                if not items:
                    break

                for item in items:
                    total_discovered += 1
                    clean_item_path = self._normalize_storage_path(item.path)

                    # Strict Project Isolation Guard:
                    # Enforce that every candidate object strictly belongs to clean_project_id
                    if not clean_item_path.startswith(project_prefix_guard):
                        logger.warning(
                            "Skipping object '%s' outside project boundary '%s'",
                            clean_item_path,
                            clean_project_id,
                        )
                        skipped_paths.append(clean_item_path)
                        continue

                    # Directory-like or placeholder filter
                    if self._is_directory_or_placeholder(item):
                        logger.debug("Skipping directory/placeholder storage entry: '%s'", clean_item_path)
                        continue

                    # Document type validation
                    doc_type = get_document_type(clean_item_path)
                    if doc_type is None:
                        logger.debug("Skipping unsupported file format: '%s'", clean_item_path)
                        skipped_paths.append(clean_item_path)
                        continue

                    # Valid supported document acquired
                    doc = self._build_source_document(
                        item=item,
                        project_id=clean_project_id,
                        document_type=doc_type,
                    )
                    acquired_documents.append(doc)

                    if limit is not None and len(acquired_documents) >= limit:
                        break

                if len(items) < batch_limit:
                    break
                offset += len(items)

        except StorageConfigurationError as exc:
            logger.error("Storage configuration error during acquisition for project '%s': %s", clean_project_id, exc)
            raise AcquisitionConfigurationError(
                f"Storage configuration error for project '{clean_project_id}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageAuthenticationError as exc:
            logger.error("Storage authentication error during acquisition for project '%s': %s", clean_project_id, exc)
            raise SourceDiscoveryError(
                f"Authentication failed while discovering documents for project '{clean_project_id}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageError as exc:
            logger.error("Storage error during acquisition for project '%s': %s", clean_project_id, exc)
            raise SourceDiscoveryError(
                f"Storage failure while discovering documents for project '{clean_project_id}': {exc.message}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during acquisition for project '%s': %s", clean_project_id, exc)
            raise SourceDiscoveryError(
                f"Unexpected failure discovering documents for project '{clean_project_id}': {exc}",
                original_error=exc,
            ) from exc

        # Determine overall status
        if not acquired_documents:
            status = AcquisitionStatus.EMPTY
        elif skipped_paths:
            status = AcquisitionStatus.PARTIAL
        else:
            status = AcquisitionStatus.SUCCESS

        logger.info(
            "Acquisition completed for project '%s': %d documents acquired, %d skipped, %d total discovered (status: %s)",
            clean_project_id,
            len(acquired_documents),
            len(skipped_paths),
            total_discovered,
            status.value,
        )

        return AcquisitionResult(
            project_id=clean_project_id,
            documents=acquired_documents,
            skipped_paths=skipped_paths,
            total_discovered=total_discovered,
            status=status,
        )

    async def acquire_document(
        self,
        project_id: str,
        storage_path: str,
    ) -> SourceDocument:
        """Acquire and validate a single specific source document by its storage path."""
        clean_project_id = self._validate_project_id(project_id)
        full_path = self._resolve_and_verify_path(clean_project_id, storage_path)

        # Validate format support
        doc_type = get_document_type(full_path)
        if doc_type is None:
            raise UnsupportedDocumentError(
                f"Document '{full_path}' has an unsupported file format. "
                "Supported formats are: .pdf, .docx, .txt, .md, .markdown"
            )

        logger.debug("Retrieving metadata for single document: '%s' (project: '%s')", full_path, clean_project_id)

        try:
            metadata = await self._storage.get_metadata(full_path)
        except ObjectNotFoundError as exc:
            logger.warning("Document '%s' not found in storage", full_path)
            raise SourceDiscoveryError(
                f"Document '{full_path}' not found in storage bucket '{self._storage.bucket_name}'.",
                original_error=exc,
            ) from exc
        except StorageAuthenticationError as exc:
            raise SourceDiscoveryError(
                f"Authentication failed retrieving document '{full_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageConfigurationError as exc:
            raise AcquisitionConfigurationError(
                f"Storage configuration error retrieving document '{full_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageError as exc:
            raise SourceDiscoveryError(
                f"Failed to retrieve metadata for document '{full_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise SourceDiscoveryError(
                f"Unexpected error retrieving document '{full_path}': {exc}",
                original_error=exc,
            ) from exc

        if self._is_directory_or_placeholder(metadata):
            raise InvalidSourceReferenceError(f"Target path '{full_path}' is a directory, not a document.")

        return self._build_source_document(
            item=metadata,
            project_id=clean_project_id,
            document_type=doc_type,
        )
