"""Application service for Document (Source) and DocumentVersion lifecycle management."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import uuid

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from observability.logging import get_logger
from exceptions.document import (
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
    InvalidDocumentDataError,
)
from exceptions.project import ProjectNotFoundError
from models.document import DocumentModel, DocumentVersionModel, DocumentVersionStatus
from models.project import ProjectModel
from storage.object.base import BaseObjectStorage

if TYPE_CHECKING:
    from services.document_processing_service import DocumentProcessingService

logger = get_logger(__name__)


class DocumentService:
    """Service handling logical Document and physical DocumentVersion lifecycles."""

    def __init__(
        self,
        session: AsyncSession,
        storage: BaseObjectStorage,
        processing_service: Optional["DocumentProcessingService"] = None,
        auto_process: bool = False,
    ) -> None:
        """Initialize DocumentService with database session and object storage client.
        
        Args:
            session: Active asynchronous SQLAlchemy database session.
            storage: BaseObjectStorage client for managing binary file objects.
            processing_service: Optional DocumentProcessingService for lifecycle processing.
            auto_process: If True, automatically triggers the processing lifecycle upon upload.
        """
        self._session = session
        self._storage = storage
        self._processing_service = processing_service
        self._auto_process = auto_process

    async def _verify_project_exists(self, project_id: str) -> None:
        """Verify that the owning project exists before executing child operations.
        
        Raises:
            ProjectNotFoundError: If the project does not exist.
        """
        stmt = select(ProjectModel.id).where(ProjectModel.id == project_id)
        result = await self._session.execute(stmt)
        if result.scalar_one_or_none() is None:
            logger.warning("Target project '%s' does not exist", project_id)
            raise ProjectNotFoundError(project_id)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename by stripping directory separators and whitespace.
        
        Raises:
            InvalidDocumentDataError: If filename is empty or invalid.
        """
        if not filename or not filename.strip():
            raise InvalidDocumentDataError("Filename cannot be empty.")
        clean_name = Path(filename.strip()).name
        if not clean_name or clean_name in (".", ".."):
            raise InvalidDocumentDataError("Invalid filename provided.")
        return clean_name

    async def create_document(
        self,
        project_id: str,
        filename: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> DocumentModel:
        """Create a new logical Document and its initial physical DocumentVersion (v1).
        
        Uploads the raw original file to Supabase Storage and records metadata and initial
        'pending' status in PostgreSQL.
        
        Args:
            project_id: Owning project identifier.
            filename: Original uploaded file name.
            file_data: Raw byte content of the uploaded file.
            content_type: Optional MIME content type.
            name: Optional user-facing display name (defaults to filename).
            
        Returns:
            DocumentModel instance with version 1 pre-populated.
            
        Raises:
            ProjectNotFoundError: If project does not exist.
            InvalidDocumentDataError: If filename or file data is invalid.
            StorageUploadError: If upload to Supabase Storage fails.
        """
        await self._verify_project_exists(project_id)

        clean_filename = self._sanitize_filename(filename)
        if not file_data:
            raise InvalidDocumentDataError("Uploaded file content cannot be empty.")

        doc_name = name.strip() if name and name.strip() else clean_filename
        doc_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        storage_path = f"{project_id}/{doc_id}/v1/{clean_filename}"

        # 1. Upload original binary object to Supabase Storage
        logger.debug(
            "Uploading initial version file for doc '%s' to storage path '%s'",
            doc_id,
            storage_path,
        )
        metadata = await self._storage.upload(
            path=storage_path,
            data=file_data,
            content_type=content_type,
        )

        # 2. Persist relational document and version metadata
        document = DocumentModel(
            id=doc_id,
            project_id=project_id,
            name=doc_name,
        )
        version = DocumentVersionModel(
            id=version_id,
            document_id=doc_id,
            project_id=project_id,
            version_number=1,
            storage_bucket=metadata.bucket,
            storage_path=metadata.path,
            original_filename=clean_filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            etag=metadata.etag,
            status=DocumentVersionStatus.PENDING.value,
        )

        self._session.add(document)
        self._session.add(version)

        try:
            await self._session.flush()
        except Exception as exc:
            logger.error(
                "Database error while saving document '%s'. Rolling back and removing storage object: %s",
                doc_id,
                exc,
            )
            # Best-effort cleanup of orphan storage file on DB error
            try:
                await self._storage.delete(metadata.path)
            except Exception as storage_exc:
                logger.warning("Failed to clean up storage object '%s': %s", metadata.path, storage_exc)
            raise

        logger.info(
            "Created source document '%s' (id: %s, v1) in project '%s'",
            document.name,
            document.id,
            project_id,
        )
        if self._auto_process and self._processing_service is not None:
            await self._processing_service.trigger_processing(
                project_id=project_id,
                document_id=doc_id,
                document_version_id=version_id,
                background=False,
            )
        return await self.get_document(project_id=project_id, document_id=doc_id)

    async def create_document_version(
        self,
        project_id: str,
        document_id: str,
        filename: str,
        file_data: bytes,
        content_type: Optional[str] = None,
    ) -> DocumentVersionModel:
        """Create and upload a new sequential DocumentVersion for an existing Document.
        
        Args:
            project_id: Owning project identifier.
            document_id: Existing logical document identifier.
            filename: Name of the newly uploaded file revision.
            file_data: Raw byte content of the file.
            content_type: Optional MIME content type.
            
        Returns:
            DocumentVersionModel representing the newly created version.
            
        Raises:
            DocumentNotFoundError: If document does not exist in the project.
            InvalidDocumentDataError: If filename or file data is invalid.
            StorageUploadError: If upload fails.
        """
        document = await self.get_document(project_id=project_id, document_id=document_id)

        clean_filename = self._sanitize_filename(filename)
        if not file_data:
            raise InvalidDocumentDataError("Uploaded file content cannot be empty.")

        # Determine next sequential version number
        stmt = select(sa.func.max(DocumentVersionModel.version_number)).where(
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.project_id == project_id,
        )
        result = await self._session.execute(stmt)
        current_max = result.scalar() or 0
        next_version = current_max + 1

        version_id = str(uuid.uuid4())
        storage_path = f"{project_id}/{document_id}/v{next_version}/{clean_filename}"

        # 1. Upload new physical file to Supabase Storage
        metadata = await self._storage.upload(
            path=storage_path,
            data=file_data,
            content_type=content_type,
        )

        # 2. Persist new DocumentVersionModel
        version = DocumentVersionModel(
            id=version_id,
            document_id=document_id,
            project_id=project_id,
            version_number=next_version,
            storage_bucket=metadata.bucket,
            storage_path=metadata.path,
            original_filename=clean_filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            etag=metadata.etag,
            status=DocumentVersionStatus.PENDING.value,
            document=document,
        )

        document.updated_at = datetime.now(timezone.utc)
        self._session.add(version)
        self._session.expire(document, ["versions"])

        try:
            await self._session.flush()
        except Exception as exc:
            logger.error("DB error saving version v%d for doc '%s': %s", next_version, document_id, exc)
            try:
                await self._storage.delete(metadata.path)
            except Exception as storage_exc:
                logger.warning("Failed to clean up storage object '%s': %s", metadata.path, storage_exc)
            raise

        logger.info(
            "Created version v%d (id: %s) for document '%s' in project '%s'",
            next_version,
            version.id,
            document_id,
            project_id,
        )
        if self._auto_process and self._processing_service is not None:
            processed = await self._processing_service.trigger_processing(
                project_id=project_id,
                document_id=document_id,
                document_version_id=version.id,
                background=False,
            )
            if isinstance(processed, DocumentVersionModel):
                version = processed
        return version

    async def list_documents(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentModel]:
        """List all documents for a project, enforcing project boundary in SQL.
        
        Args:
            project_id: Target project identifier.
            limit: Maximum items to return.
            offset: Number of items to skip.
            
        Returns:
            List of DocumentModel entities with their versions loaded.
            
        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        await self._verify_project_exists(project_id)

        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        stmt = (
            select(DocumentModel)
            .options(selectinload(DocumentModel.versions))
            .where(DocumentModel.project_id == project_id)
            .order_by(DocumentModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_document(
        self,
        project_id: str,
        document_id: str,
    ) -> DocumentModel:
        """Retrieve a document by ID ensuring strict project isolation at the SQL query level.
        
        Args:
            project_id: Owning project identifier.
            document_id: Document identifier.
            
        Returns:
            DocumentModel instance with all versions loaded.
            
        Raises:
            DocumentNotFoundError: If document not found or belongs to another project.
        """
        stmt = (
            select(DocumentModel)
            .options(selectinload(DocumentModel.versions))
            .where(
                DocumentModel.id == document_id,
                DocumentModel.project_id == project_id,
            )
        )
        result = await self._session.execute(stmt)
        document = result.scalar_one_or_none()

        if document is None:
            logger.warning(
                "Document '%s' not found under project '%s' (isolation enforced)",
                document_id,
                project_id,
            )
            raise DocumentNotFoundError(document_id=document_id, project_id=project_id)

        return document

    async def update_document(
        self,
        project_id: str,
        document_id: str,
        name: Optional[str] = None,
    ) -> DocumentModel:
        """Update mutable attributes (e.g. name) of a document under project isolation.
        
        Args:
            project_id: Owning project identifier.
            document_id: Document identifier.
            name: New display name.
            
        Returns:
            Updated DocumentModel instance.
            
        Raises:
            DocumentNotFoundError: If document not found under project.
            InvalidDocumentDataError: If updated name is empty.
        """
        document = await self.get_document(project_id=project_id, document_id=document_id)

        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise InvalidDocumentDataError("Document name cannot be empty or whitespace only.")
            document.name = clean_name

        await self._session.flush()
        logger.info("Updated document '%s' in project '%s'", document_id, project_id)
        return document

    async def delete_document(
        self,
        project_id: str,
        document_id: str,
    ) -> None:
        """Delete a document and clean up associated physical files in Supabase Storage.
        
        PostgreSQL database cascade handles deleting dependent document_versions and chunks.
        
        Args:
            project_id: Owning project identifier.
            document_id: Document identifier.
            
        Raises:
            DocumentNotFoundError: If document not found under project.
        """
        document = await self.get_document(project_id=project_id, document_id=document_id)

        # Collect all physical storage paths for this document before deletion
        paths_stmt = select(DocumentVersionModel.storage_path).where(
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.project_id == project_id,
        )
        paths_result = await self._session.execute(paths_stmt)
        storage_paths = [p for p in paths_result.scalars().all() if p]

        # Delete from PostgreSQL (cascades delete versions and chunks)
        await self._session.delete(document)
        await self._session.flush()

        # Clean up files in Supabase Storage
        if storage_paths:
            try:
                deleted = await self._storage.delete_many(storage_paths)
                logger.info(
                    "Deleted %d storage objects for document '%s'",
                    len(deleted),
                    document_id,
                )
            except Exception as exc:
                logger.warning(
                    "Storage cleanup failed for document '%s': %s",
                    document_id,
                    exc,
                )

        logger.info("Deleted document '%s' from project '%s'", document_id, project_id)

    async def get_document_version(
        self,
        project_id: str,
        document_id: str,
        version_number: int,
    ) -> DocumentVersionModel:
        """Retrieve a specific physical version of a document.
        
        Args:
            project_id: Owning project identifier.
            document_id: Document identifier.
            version_number: 1-based version number.
            
        Returns:
            DocumentVersionModel instance.
            
        Raises:
            DocumentVersionNotFoundError: If specific version does not exist.
        """
        stmt = select(DocumentVersionModel).where(
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.project_id == project_id,
            DocumentVersionModel.version_number == version_number,
        )
        result = await self._session.execute(stmt)
        version = result.scalar_one_or_none()

        if version is None:
            raise DocumentVersionNotFoundError(
                document_id=document_id,
                version_number=version_number,
                project_id=project_id,
            )

        return version

    async def get_version_by_id(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
    ) -> DocumentVersionModel:
        """Retrieve a specific physical version of a document by version ID.
        
        Args:
            project_id: Owning project identifier.
            document_id: Document identifier.
            document_version_id: Version primary key identifier.
            
        Returns:
            DocumentVersionModel instance.
            
        Raises:
            DocumentVersionNotFoundError: If specific version does not exist.
        """
        stmt = select(DocumentVersionModel).where(
            DocumentVersionModel.id == document_version_id,
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.project_id == project_id,
        )
        result = await self._session.execute(stmt)
        version = result.scalar_one_or_none()

        if version is None:
            raise DocumentVersionNotFoundError(
                document_id=document_id,
                version_number=0,
                project_id=project_id,
                message=f"Document version '{document_version_id}' not found for document '{document_id}' in project '{project_id}'.",
            )

        return version
