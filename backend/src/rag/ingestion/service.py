"""Storage-backed document ingestion service for the indexing pipeline."""

from pathlib import PurePosixPath
from typing import Any, Optional

from rag.acquisition.formats import DocumentType
from rag.acquisition.models import SourceDocument, SourceReference
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.ingestion import (
    DocumentRetrievalError,
    InvalidIngestionInputError,
    UnsupportedDocumentTypeError,
)
from exceptions.storage import (
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageDownloadError,
    StorageError,
)
from rag.ingestion.base import BaseIngestionService, IngestionSourceInput
from rag.ingestion.models import DocumentSourceReference, IngestedDocument
from rag.ingestion.validators import (
    resolve_and_validate_document_type,
    resolve_and_verify_storage_path,
    validate_project_id,
)
from storage.object import BaseObjectStorage, get_object_storage

logger = get_logger(__name__)


class StorageIngestionService(BaseIngestionService):
    """Ingestion service that retrieves acquired source documents from Object Storage.
    
    Reuses the BaseObjectStorage abstraction without direct coupling to Supabase SDKs,
    ensuring strict project tenant isolation, standardized IngestedDocument emission,
    and clear boundary separation from rag.parsing and extraction stages.
    """

    def __init__(
        self,
        storage: Optional[BaseObjectStorage] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Storage Ingestion Service.
        
        Args:
            storage: Object storage provider instance conforming to BaseObjectStorage.
                     If None, resolves from get_object_storage().
            settings: Application settings instance.
        """
        self._settings = settings or get_settings()
        self._storage = storage or get_object_storage(settings=self._settings)

    def _extract_source_parameters(
        self,
        source: IngestionSourceInput,
        project_id: Optional[str] = None,
    ) -> tuple[str, str, str, Optional[str], Optional[DocumentType], Optional[str], Optional[str], dict[str, Any]]:
        """Extract and normalize parameters from supported input reference types.
        
        Returns:
            Tuple of (project_id, storage_path, filename, content_type, document_type,
                      document_id, document_version_id, source_metadata).
        """
        if source is None:
            raise InvalidIngestionInputError("Source document reference cannot be None.")

        # Case 1: SourceDocument (from Acquisition stage)
        if isinstance(source, SourceDocument):
            resolved_project_id = source.project_id or project_id
            if not resolved_project_id:
                raise InvalidIngestionInputError("project_id missing from SourceDocument reference.")
            filename = source.filename or PurePosixPath(source.storage_path).name
            version_id = source.metadata.get("etag") or source.metadata.get("eTag")
            return (
                resolved_project_id,
                source.storage_path,
                filename,
                source.content_type,
                source.document_type,
                source.source_id,
                version_id,
                dict(source.metadata),
            )

        # Case 2: DocumentSourceReference (explicit ingestion input reference)
        if isinstance(source, DocumentSourceReference):
            resolved_project_id = source.project_id or project_id
            if not resolved_project_id:
                raise InvalidIngestionInputError("project_id missing from DocumentSourceReference.")
            filename = source.original_filename or PurePosixPath(source.storage_path).name
            return (
                resolved_project_id,
                source.storage_path,
                filename,
                source.content_type,
                source.document_type,
                source.document_id,
                source.document_version_id,
                dict(source.source_metadata),
            )

        # Case 3: SourceReference (storage coordinate reference)
        if isinstance(source, SourceReference):
            if not project_id:
                raise InvalidIngestionInputError("project_id is required when ingesting from a SourceReference.")
            filename = PurePosixPath(source.path).name
            meta: dict[str, Any] = {"bucket": source.bucket, "source_type": source.source_type}
            if source.etag:
                meta["etag"] = source.etag
            return (
                project_id,
                source.path,
                filename,
                None,
                None,
                None,
                source.etag,
                meta,
            )

        # Case 4: Raw storage path string
        if isinstance(source, str):
            if not project_id:
                raise InvalidIngestionInputError("project_id is required when ingesting from a storage path string.")
            filename = PurePosixPath(source.replace("\\", "/")).name
            return (
                project_id,
                source,
                filename,
                None,
                None,
                None,
                None,
                {},
            )

        raise InvalidIngestionInputError(
            f"Unsupported source reference type '{type(source).__name__}'. "
            "Expected SourceDocument, DocumentSourceReference, SourceReference, or storage path string."
        )

    async def ingest(
        self,
        source: IngestionSourceInput,
        project_id: Optional[str] = None,
    ) -> IngestedDocument:
        """Ingest an acquired document reference by retrieving raw bytes from storage.
        
        Workflow:
        1. Extract and validate source coordinates & tenant isolation.
        2. Resolve and validate document type (MIME check -> extension fallback).
        3. Asynchronously download original bytes via BaseObjectStorage.
        4. Preserve source metadata.
        5. Construct and return standardized IngestedDocument.
        """
        # Step 1: Extract and validate input coordinates
        (
            raw_project_id,
            raw_storage_path,
            filename,
            content_type,
            explicit_type,
            explicit_doc_id,
            version_id,
            metadata,
        ) = self._extract_source_parameters(source, project_id=project_id)

        clean_project_id = validate_project_id(raw_project_id)
        full_storage_path = resolve_and_verify_storage_path(clean_project_id, raw_storage_path)

        # Step 2: Determine and validate document type
        detected_type, canonical_mime = resolve_and_validate_document_type(
            content_type=content_type,
            filename_or_path=filename or full_storage_path,
            explicit_type=explicit_type,
        )

        doc_id = explicit_doc_id or f"{self._storage.bucket_name}/{full_storage_path}"

        logger.info(
            "Starting ingestion for document '%s' in project '%s' (path: '%s', type: '%s')",
            doc_id,
            clean_project_id,
            full_storage_path,
            detected_type.value,
        )

        # Step 3: Asynchronously retrieve raw bytes from storage
        try:
            raw_bytes = await self._storage.download(full_storage_path)
        except ObjectNotFoundError as exc:
            logger.warning("Document '%s' not found in storage", full_storage_path)
            raise DocumentRetrievalError(
                f"Document not found in storage at path '{full_storage_path}'.",
                original_error=exc,
            ) from exc
        except StorageAuthenticationError as exc:
            logger.error("Storage authentication failed downloading '%s': %s", full_storage_path, exc.message)
            raise DocumentRetrievalError(
                f"Authentication failed downloading document '{full_storage_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageDownloadError as exc:
            logger.error("Download failed for document '%s': %s", full_storage_path, exc.message)
            raise DocumentRetrievalError(
                f"Failed to download document at '{full_storage_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except StorageError as exc:
            logger.error("Storage error downloading document '%s': %s", full_storage_path, exc.message)
            raise DocumentRetrievalError(
                f"Storage error retrieving document at '{full_storage_path}': {exc.message}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error downloading document '%s': %s", full_storage_path, exc)
            raise DocumentRetrievalError(
                f"Unexpected error retrieving document at '{full_storage_path}': {exc}",
                original_error=exc,
            ) from exc

        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise DocumentRetrievalError(
                f"Invalid storage payload received for document '{full_storage_path}'. Expected bytes."
            )

        # Step 4: Preserve source metadata
        preserved_metadata = dict(metadata)
        if "storage_bucket" not in preserved_metadata:
            preserved_metadata["storage_bucket"] = self._storage.bucket_name
        if "storage_path" not in preserved_metadata:
            preserved_metadata["storage_path"] = full_storage_path
        if version_id and "etag" not in preserved_metadata:
            preserved_metadata["etag"] = version_id

        # Step 5: Construct standardized IngestedDocument
        ingested_doc = IngestedDocument(
            document_id=doc_id,
            project_id=clean_project_id,
            source_storage_path=full_storage_path,
            original_filename=filename,
            detected_document_type=detected_type,
            content_type=canonical_mime,
            raw_bytes=bytes(raw_bytes),
            source_metadata=preserved_metadata,
            document_version_id=version_id,
            size_bytes=len(raw_bytes),
        )

        logger.info(
            "Completed ingestion for document '%s' (type: %s, size: %d bytes)",
            doc_id,
            detected_type.value,
            len(raw_bytes),
        )

        return ingested_doc

    async def ingest_batch(
        self,
        sources: list[IngestionSourceInput],
        project_id: Optional[str] = None,
    ) -> list[IngestedDocument]:
        """Ingest multiple acquired document references into the pipeline."""
        if not sources:
            return []

        logger.info("Starting batch ingestion for %d documents", len(sources))
        results: list[IngestedDocument] = []

        for index, source in enumerate(sources):
            logger.debug("Ingesting batch document %d/%d", index + 1, len(sources))
            doc = await self.ingest(source, project_id=project_id)
            results.append(doc)

        logger.info("Successfully ingested %d batch documents", len(results))
        return results
