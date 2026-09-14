"""Default document processing pipeline boundary implementation."""

from typing import Any, Callable, Optional, Awaitable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from rag.pipeline.base import BaseDocumentProcessingPipeline
from storage.object import BaseObjectStorage

logger = get_logger(__name__)


class DefaultDocumentProcessingPipeline(BaseDocumentProcessingPipeline):
    """Default processing pipeline establishing the boundary for document version execution.

    Serves as the integration point between the Document Processing Lifecycle and
    downstream RAG stages. In the upcoming End-to-End Indexing phase, this boundary
    will connect all 10 stages (Acquisition -> Ingestion -> Parsing -> Cleaning ->
    Normalization -> Chunking -> Metadata Enrichment -> Contextual Enrichment ->
    Embeddings/Sparse -> Indexing).
    """

    def __init__(
        self,
        storage: Optional[BaseObjectStorage] = None,
        stage_hook: Optional[Callable[..., Awaitable[Any]]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the default processing pipeline.

        Args:
            storage: Optional object storage client for reading raw source binaries.
            stage_hook: Optional custom async processing hook (useful for testing or delegates).
            settings: Optional application settings override.
        """
        self._storage = storage
        self._stage_hook = stage_hook
        self._settings = settings or get_settings()

    async def process_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        storage_bucket: str,
        storage_path: str,
        original_filename: str,
        content_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute processing for a verified document version.

        Args:
            project_id: Verified owning project identifier.
            document_id: Verified owning document identifier.
            document_version_id: Verified physical document version identifier.
            storage_bucket: Object storage bucket containing the raw document.
            storage_path: Relative storage path of the raw document file.
            original_filename: Original name of the uploaded document file.
            content_type: Optional MIME content type.

        Returns:
            Dictionary summary of the pipeline execution report.

        Raises:
            Exception: If downstream processing fails.
        """
        logger.info(
            "Pipeline processing initiated for version '%s' (project: '%s', doc: '%s', path: '%s')",
            document_version_id,
            project_id,
            document_id,
            storage_path,
        )

        # If a custom processing stage hook is provided (e.g. test delegate or pipeline step), execute it
        if self._stage_hook is not None:
            hook_result = await self._stage_hook(
                project_id=project_id,
                document_id=document_id,
                document_version_id=document_version_id,
                storage_bucket=storage_bucket,
                storage_path=storage_path,
                original_filename=original_filename,
                content_type=content_type,
            )
            return {
                "status": "success",
                "document_version_id": document_version_id,
                "hook_result": hook_result,
            }

        # If storage client is provided, verify raw object can be read/retrieved
        if self._storage is not None:
            metadata = await self._storage.get_metadata(storage_path)
            logger.debug(
                "Verified storage object for version '%s': size=%s bytes",
                document_version_id,
                metadata.size_bytes if metadata else None,
            )

        logger.info(
            "Pipeline processing boundary successfully verified for version '%s'",
            document_version_id,
        )
        return {
            "status": "success",
            "document_version_id": document_version_id,
            "project_id": project_id,
            "document_id": document_id,
            "storage_path": storage_path,
        }


_default_pipeline: Optional[BaseDocumentProcessingPipeline] = None


def get_processing_pipeline(
    storage: Optional[BaseObjectStorage] = None,
    stage_hook: Optional[Callable[..., Awaitable[Any]]] = None,
    settings: Optional[Settings] = None,
) -> BaseDocumentProcessingPipeline:
    """Get or create the singleton BaseDocumentProcessingPipeline instance."""
    global _default_pipeline

    if storage is not None or stage_hook is not None:
        return DefaultDocumentProcessingPipeline(
            storage=storage,
            stage_hook=stage_hook,
            settings=settings,
        )

    if _default_pipeline is None:
        _default_pipeline = DefaultDocumentProcessingPipeline(settings=settings)

    return _default_pipeline


def reset_processing_pipeline() -> None:
    """Reset the cached default pipeline instance (useful for tests)."""
    global _default_pipeline
    _default_pipeline = None
