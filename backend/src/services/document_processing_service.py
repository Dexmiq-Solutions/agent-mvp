"""Application service orchestrating the Document Processing Lifecycle for DocumentVersions."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import re
import time
from typing import Any, AsyncIterator, Optional, Union

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from db.session import get_async_session_maker
from exceptions.document import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentStorageSourceError,
    DocumentVersionMismatchError,
    DocumentVersionNotFoundError,
    InvalidDocumentStateTransitionError,
    ProjectDocumentMismatchError,
)
from exceptions.project import ProjectNotFoundError
from models.document import DocumentModel, DocumentVersionModel, DocumentVersionStatus
from models.project import ProjectModel
from rag.pipeline.base import BaseDocumentProcessingPipeline
from rag.pipeline.default import get_processing_pipeline
from storage.object.base import BaseObjectStorage

logger = get_logger(__name__)

# Explicit valid state transitions table
VALID_STATE_TRANSITIONS: dict[str, set[str]] = {
    DocumentVersionStatus.PENDING.value: {DocumentVersionStatus.INDEXING.value},
    DocumentVersionStatus.INDEXING.value: {
        DocumentVersionStatus.READY.value,
        DocumentVersionStatus.FAILED.value,
    },
    DocumentVersionStatus.READY.value: set(),  # Terminal state (reprocessing not yet finalized)
    DocumentVersionStatus.FAILED.value: set(),  # Terminal state (retry orchestration not yet finalized)
}


def sanitize_error_message(exc: Exception) -> str:
    """Sanitize error messages by masking credentials, connection strings, and provider secrets."""
    msg = str(exc)
    if not msg:
        msg = type(exc).__name__
    # Mask passwords in URLs: protocol://user:pass@host
    msg = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", msg)
    # Mask OpenAI / Voyage / Supabase keys
    msg = re.sub(r"\b(sk-[a-zA-Z0-9_-]{10,})\b", "sk-***", msg)
    msg = re.sub(r"\b(ey[a-zA-Z0-9_-]{20,})\b", "ey-***", msg)
    return msg[:1000]


class DocumentProcessingService:
    """Service orchestrating the Document Processing Lifecycle for physical DocumentVersions.

    Owns:
      - Processing initiation and triggering
      - Validating project, document, and version relationships (project isolation)
      - Validating storage coordinates
      - Managing lifecycle state transitions (PENDING -> INDEXING -> READY / FAILED)
      - Enforcing duplicate-processing and invalid-transition protections
      - Invoking the RAG processing pipeline boundary without coupling to stage internals
      - Managing short transaction boundaries
      - Persisting failure diagnostics and completion status
      - Emitting lifecycle observability logs
    """

    def __init__(
        self,
        session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
        session: Optional[AsyncSession] = None,
        storage: Optional[BaseObjectStorage] = None,
        pipeline: Optional[BaseDocumentProcessingPipeline] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize DocumentProcessingService.

        Args:
            session_maker: Optional async_sessionmaker factory for independent short transactions.
            session: Optional active AsyncSession (used directly when session_maker is None).
            storage: Optional BaseObjectStorage instance for source object validation.
            pipeline: Optional BaseDocumentProcessingPipeline implementation.
            settings: Optional application settings override.
        """
        self._session_maker = session_maker
        self._session = session
        self._storage = storage
        self._settings = settings or get_settings()
        self._pipeline = pipeline or get_processing_pipeline(
            storage=storage,
            settings=self._settings,
        )

    @asynccontextmanager
    async def _get_session(self) -> AsyncIterator[AsyncSession]:
        """Provide an active AsyncSession, yielding either an injected session or a newly scoped session."""
        if self._session_maker is not None:
            async with self._session_maker() as session:
                yield session
        elif self._session is not None:
            yield self._session
        else:
            # Fallback to application session maker
            maker = get_async_session_maker(settings=self._settings)
            async with maker() as session:
                yield session

    # --------------------------------------------------------------------------
    # Validation Helpers
    # --------------------------------------------------------------------------

    async def validate_ownership_and_get_version(
        self,
        session: AsyncSession,
        project_id: str,
        document_id: str,
        document_version_id: str,
        for_update: bool = False,
    ) -> DocumentVersionModel:
        """Validate project, document, and version relationships and return the DocumentVersionModel.

        Ensures strict tenant isolation in SQL and produces specific diagnostic domain errors.

        Args:
            session: Active database session.
            project_id: Target project identifier.
            document_id: Target document identifier.
            document_version_id: Target physical document version identifier.
            for_update: Whether to lock the row using SELECT ... FOR UPDATE.

        Returns:
            DocumentVersionModel instance.

        Raises:
            ProjectNotFoundError: If project does not exist.
            DocumentNotFoundError: If document does not exist in the project.
            DocumentVersionNotFoundError: If document version does not exist.
            DocumentVersionMismatchError: If document version belongs to a different document/project.
        """
        # 1. Verify project exists
        proj_stmt = select(ProjectModel.id).where(ProjectModel.id == project_id)
        proj_res = await session.execute(proj_stmt)
        if proj_res.scalar_one_or_none() is None:
            logger.warning("Project '%s' does not exist during version validation", project_id)
            raise ProjectNotFoundError(project_id)

        # 2. Verify document exists under project
        doc_stmt = select(DocumentModel).where(
            DocumentModel.id == document_id,
            DocumentModel.project_id == project_id,
        )
        doc_res = await session.execute(doc_stmt)
        doc = doc_res.scalar_one_or_none()
        if doc is None:
            # Check if document exists in another project to raise explicit mismatch
            any_doc_stmt = select(DocumentModel.project_id).where(DocumentModel.id == document_id)
            any_doc_res = await session.execute(any_doc_stmt)
            actual_proj = any_doc_res.scalar_one_or_none()
            if actual_proj is not None:
                logger.warning(
                    "Document '%s' belongs to project '%s', not '%s'",
                    document_id,
                    actual_proj,
                    project_id,
                )
                raise ProjectDocumentMismatchError(
                    document_id=document_id,
                    expected_project_id=project_id,
                    actual_project_id=actual_proj,
                )
            logger.warning("Document '%s' not found under project '%s'", document_id, project_id)
            raise DocumentNotFoundError(document_id=document_id, project_id=project_id)

        # 3. Verify version exists and belongs to document and project
        query = select(DocumentVersionModel).where(
            DocumentVersionModel.id == document_version_id,
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.project_id == project_id,
        )
        if for_update:
            query = query.with_for_update()

        ver_res = await session.execute(query)
        version = ver_res.scalar_one_or_none()

        if version is None:
            # Check if version exists under a different document or project
            any_ver_stmt = select(DocumentVersionModel).where(
                DocumentVersionModel.id == document_version_id
            )
            any_ver_res = await session.execute(any_ver_stmt)
            any_ver = any_ver_res.scalar_one_or_none()
            if any_ver is not None:
                if any_ver.project_id != project_id:
                    logger.warning(
                        "Version '%s' belongs to project '%s', not '%s'",
                        document_version_id,
                        any_ver.project_id,
                        project_id,
                    )
                    raise DocumentVersionMismatchError(
                        version_id=document_version_id,
                        expected_parent=project_id,
                        actual_parent=any_ver.project_id,
                        entity_type="project",
                    )
                if any_ver.document_id != document_id:
                    logger.warning(
                        "Version '%s' belongs to document '%s', not '%s'",
                        document_version_id,
                        any_ver.document_id,
                        document_id,
                    )
                    raise DocumentVersionMismatchError(
                        version_id=document_version_id,
                        expected_parent=document_id,
                        actual_parent=any_ver.document_id,
                        entity_type="document",
                    )
            logger.warning(
                "Version '%s' not found for document '%s' in project '%s'",
                document_version_id,
                document_id,
                project_id,
            )
            raise DocumentVersionNotFoundError(
                document_id=document_id,
                version_number=0,
                project_id=project_id,
                message=f"Document version '{document_version_id}' not found for document '{document_id}'.",
            )

        return version

    @staticmethod
    def validate_state_transition(
        version_id: str,
        current_status: str,
        target_status: str,
    ) -> None:
        """Validate whether transitioning from current_status to target_status is permitted.

        Raises:
            DocumentAlreadyProcessingError: If version is already actively in indexing state.
            InvalidDocumentStateTransitionError: If the transition is disallowed.
        """
        # Duplicate processing protection
        if current_status == DocumentVersionStatus.INDEXING.value and target_status == DocumentVersionStatus.INDEXING.value:
            logger.warning(
                "Duplicate processing rejected: version '%s' is already actively in INDEXING state",
                version_id,
            )
            raise DocumentAlreadyProcessingError(version_id=version_id)

        allowed_targets = VALID_STATE_TRANSITIONS.get(current_status, set())
        if target_status not in allowed_targets:
            logger.warning(
                "Invalid state transition rejected for version '%s': '%s' -> '%s'",
                version_id,
                current_status,
                target_status,
            )
            raise InvalidDocumentStateTransitionError(
                version_id=version_id,
                current_status=current_status,
                target_status=target_status,
            )

    @staticmethod
    def validate_storage_coordinates(version: DocumentVersionModel) -> None:
        """Validate required object storage metadata and coordinates.

        Raises:
            DocumentStorageSourceError: If bucket, path, or filename is missing or empty.
        """
        if not version.storage_bucket or not version.storage_bucket.strip():
            raise DocumentStorageSourceError(
                version_id=version.id,
                message=f"Storage bucket missing for document version '{version.id}'.",
            )
        if not version.storage_path or not version.storage_path.strip():
            raise DocumentStorageSourceError(
                version_id=version.id,
                message=f"Storage path missing for document version '{version.id}'.",
            )
        if not version.original_filename or not version.original_filename.strip():
            raise DocumentStorageSourceError(
                version_id=version.id,
                message=f"Original filename missing for document version '{version.id}'.",
            )

    # --------------------------------------------------------------------------
    # Lifecycle Execution
    # --------------------------------------------------------------------------

    async def process_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
    ) -> DocumentVersionModel:
        """Execute the Document Processing Lifecycle for a specific physical DocumentVersion.

        Workflow:
          1. Validate project, document, version existence and tenant boundary in a short transaction.
          2. Check current state and validate storage source coordinates.
          3. Transition state: PENDING -> INDEXING and commit transaction 1.
          4. Invoke the RAG processing pipeline boundary outside open DB transactions.
          5. On success: Transition state: INDEXING -> READY, record indexed_at, commit transaction 2.
          6. On failure: Transition state: INDEXING -> FAILED, record sanitized error_message, commit transaction 2.

        Args:
            project_id: Owning project identifier.
            document_id: Owning document identifier.
            document_version_id: Physical document version identifier.

        Returns:
            Updated DocumentVersionModel reflecting the final state.

        Raises:
            DocumentProcessingError: If processing fails or invalid state prevents processing.
        """
        start_time = time.monotonic()
        logger.info(
            "Initiating document processing lifecycle for version '%s' (project: '%s', doc: '%s')",
            document_version_id,
            project_id,
            document_id,
        )

        # ----------------------------------------------------------------------
        # Phase 1: Short Transaction - Validation & Transition to INDEXING
        # ----------------------------------------------------------------------
        async with self._get_session() as session:
            version = await self.validate_ownership_and_get_version(
                session=session,
                project_id=project_id,
                document_id=document_id,
                document_version_id=document_version_id,
                for_update=True,
            )

            # Validate state transition from current status to INDEXING
            self.validate_state_transition(
                version_id=version.id,
                current_status=version.status,
                target_status=DocumentVersionStatus.INDEXING.value,
            )

            # Validate storage source coordinates
            try:
                self.validate_storage_coordinates(version)
            except DocumentStorageSourceError as storage_err:
                logger.error(
                    "Storage validation failed for version '%s': %s",
                    version.id,
                    storage_err,
                )
                version.status = DocumentVersionStatus.FAILED.value
                version.error_message = sanitize_error_message(storage_err)
                version.updated_at = datetime.now(timezone.utc)
                await session.commit()
                raise

            # Transition to INDEXING
            version.status = DocumentVersionStatus.INDEXING.value
            version.error_message = None
            version.updated_at = datetime.now(timezone.utc)
            await session.commit()

            # Cache needed attributes for external pipeline execution
            storage_bucket = version.storage_bucket
            storage_path = version.storage_path
            original_filename = version.original_filename
            content_type = version.content_type

            logger.info(
                "Document version '%s' successfully transitioned to INDEXING",
                document_version_id,
            )

        # ----------------------------------------------------------------------
        # Phase 2: Pipeline Execution Boundary (Outside DB Transaction)
        # ----------------------------------------------------------------------
        pipeline_error: Optional[Exception] = None
        try:
            await self._pipeline.process_document_version(
                project_id=project_id,
                document_id=document_id,
                document_version_id=document_version_id,
                storage_bucket=storage_bucket,
                storage_path=storage_path,
                original_filename=original_filename,
                content_type=content_type,
            )
        except Exception as exc:
            pipeline_error = exc
            logger.error(
                "Pipeline execution failed for document version '%s': %s",
                document_version_id,
                exc,
                exc_info=True,
            )

        duration = time.monotonic() - start_time

        # ----------------------------------------------------------------------
        # Phase 3: Short Transaction - Final State Update (READY or FAILED)
        # ----------------------------------------------------------------------
        async with self._get_session() as session:
            final_version = await self.validate_ownership_and_get_version(
                session=session,
                project_id=project_id,
                document_id=document_id,
                document_version_id=document_version_id,
                for_update=True,
            )

            if pipeline_error is None:
                # Transition INDEXING -> READY
                self.validate_state_transition(
                    version_id=final_version.id,
                    current_status=final_version.status,
                    target_status=DocumentVersionStatus.READY.value,
                )
                final_version.status = DocumentVersionStatus.READY.value
                final_version.indexed_at = datetime.now(timezone.utc)
                final_version.error_message = None
                final_version.updated_at = datetime.now(timezone.utc)
                await session.commit()

                logger.info(
                    "Document version '%s' successfully processed to READY in %.2fs",
                    document_version_id,
                    duration,
                )
                return final_version
            else:
                # Transition INDEXING -> FAILED
                self.validate_state_transition(
                    version_id=final_version.id,
                    current_status=final_version.status,
                    target_status=DocumentVersionStatus.FAILED.value,
                )
                final_version.status = DocumentVersionStatus.FAILED.value
                final_version.indexed_at = None
                final_version.error_message = sanitize_error_message(pipeline_error)
                final_version.updated_at = datetime.now(timezone.utc)
                await session.commit()

                logger.error(
                    "Document version '%s' marked FAILED after %.2fs. Persisted error: %s",
                    document_version_id,
                    duration,
                    final_version.error_message,
                )
                raise DocumentProcessingError(
                    f"Processing failed for document version '{document_version_id}': {final_version.error_message}",
                    original_error=pipeline_error,
                )

    async def trigger_processing(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        background: bool = False,
    ) -> Union[asyncio.Task, DocumentVersionModel]:
        """Trigger document version processing either synchronously or as a background task.

        Args:
            project_id: Owning project identifier.
            document_id: Owning document identifier.
            document_version_id: Document version identifier.
            background: If True, schedules processing as an asyncio background task.

        Returns:
            DocumentVersionModel if synchronous, or asyncio.Task if background.
        """
        if background:
            logger.info(
                "Dispatching background task for version '%s' (project: '%s')",
                document_version_id,
                project_id,
            )

            async def _bg_runner() -> None:
                try:
                    await self.process_document_version(
                        project_id=project_id,
                        document_id=document_id,
                        document_version_id=document_version_id,
                    )
                except Exception as bg_exc:
                    logger.error(
                        "Background processing task for version '%s' failed: %s",
                        document_version_id,
                        bg_exc,
                    )

            return asyncio.create_task(_bg_runner())

        return await self.process_document_version(
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )
