"""Default document processing pipeline boundary implementation."""

from typing import Any, Awaitable, Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.indexing.orchestrator import EndToEndIndexingService, get_end_to_end_indexing_service
from rag.pipeline.base import BaseDocumentProcessingPipeline
from storage.object import BaseObjectStorage

logger = get_logger(__name__)


class DefaultDocumentProcessingPipeline(BaseDocumentProcessingPipeline):
    """Default processing pipeline connecting Document Processing Lifecycle to End-to-End Indexing.

    Coordinates all 10 indexing stages through EndToEndIndexingService:
      Acquisition -> Ingestion -> Parsing -> Cleaning -> Normalization ->
      Chunking -> Metadata Enrichment -> Contextual Enrichment ->
      Representation Generation (Dense + Sparse) -> PostgreSQL Persistence -> Qdrant Indexing.
    """

    def __init__(
        self,
        storage: Optional[BaseObjectStorage] = None,
        stage_hook: Optional[Callable[..., Awaitable[Any]]] = None,
        indexing_service: Optional[EndToEndIndexingService] = None,
        session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the default processing pipeline.

        Args:
            storage: Optional object storage client for reading raw source binaries.
            stage_hook: Optional custom async processing hook (useful for testing or delegates).
            indexing_service: Optional EndToEndIndexingService instance override.
            session_maker: Optional async session maker factory for short chunk persistence transactions.
            settings: Optional application settings override.
        """
        self._storage = storage
        self._stage_hook = stage_hook
        self._indexing_service = indexing_service
        self._session_maker = session_maker
        self._settings = settings or get_settings()

    def _get_indexing_service(self) -> EndToEndIndexingService:
        """Resolve active EndToEndIndexingService."""
        if self._indexing_service is not None:
            return self._indexing_service
        return get_end_to_end_indexing_service(
            storage=self._storage,
            session_maker=self._session_maker,
            settings=self._settings,
        )

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

        # Execute end-to-end indexing orchestration
        indexing_svc = self._get_indexing_service()
        report = await indexing_svc.index_document_version(
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
            storage_bucket=storage_bucket,
            storage_path=storage_path,
            original_filename=original_filename,
            content_type=content_type,
        )

        return report.to_dict()


_default_pipeline: Optional[BaseDocumentProcessingPipeline] = None


def get_processing_pipeline(
    storage: Optional[BaseObjectStorage] = None,
    stage_hook: Optional[Callable[..., Awaitable[Any]]] = None,
    indexing_service: Optional[EndToEndIndexingService] = None,
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    settings: Optional[Settings] = None,
) -> BaseDocumentProcessingPipeline:
    """Get or create the singleton BaseDocumentProcessingPipeline instance."""
    global _default_pipeline

    if (
        storage is not None
        or stage_hook is not None
        or indexing_service is not None
        or session_maker is not None
    ):
        return DefaultDocumentProcessingPipeline(
            storage=storage,
            stage_hook=stage_hook,
            indexing_service=indexing_service,
            session_maker=session_maker,
            settings=settings,
        )

    if _default_pipeline is None:
        _default_pipeline = DefaultDocumentProcessingPipeline(settings=settings)

    return _default_pipeline


def reset_processing_pipeline() -> None:
    """Reset the cached default pipeline instance (useful for tests)."""
    global _default_pipeline
    _default_pipeline = None
