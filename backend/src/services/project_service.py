from typing import Any, TYPE_CHECKING, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from observability.logging import get_logger
from exceptions.project import InvalidProjectDataError, ProjectNotFoundError
from models.project import ProjectModel
from storage.object.base import BaseObjectStorage

if TYPE_CHECKING:
    from storage.vector.base import BaseVectorStore

logger = get_logger(__name__)


class ProjectService:
    """Service handling project CRUD operations and tenant lifecycle."""

    def __init__(
        self,
        session: AsyncSession,
        storage: Optional[BaseObjectStorage] = None,
        vector_store: Optional["BaseVectorStore"] = None,
    ) -> None:
        """Initialize the ProjectService with a database session, optional storage, and vector store.
        
        Args:
            session: Active asynchronous SQLAlchemy session.
            storage: Optional BaseObjectStorage implementation for cleaning up project files.
            vector_store: Optional BaseVectorStore instance for cleaning up project vector embeddings.
        """
        self._session = session
        self._storage = storage
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            try:
                from storage.vector import get_vector_store
                self._vector_store = get_vector_store()
            except Exception:
                self._vector_store = None

    async def create_project(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> ProjectModel:
        """Create and persist a new project entity.
        
        Args:
            name: Project display name.
            description: Optional project description.
            
        Returns:
            Newly created ProjectModel instance.
            
        Raises:
            InvalidProjectDataError: If project name is empty or invalid.
        """
        if not name or not name.strip():
            raise InvalidProjectDataError("Project name cannot be empty or whitespace only.")

        clean_name = name.strip()
        project = ProjectModel(
            name=clean_name,
            description=description,
        )
        self._session.add(project)
        await self._session.flush()

        logger.info("Created project '%s' (id: %s)", project.name, project.id)
        return project

    async def get_project(self, project_id: str) -> ProjectModel:
        """Retrieve a project by its primary key identifier.
        
        Args:
            project_id: Project identifier.
            
        Returns:
            ProjectModel instance.
            
        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        stmt = select(ProjectModel).where(ProjectModel.id == project_id)
        result = await self._session.execute(stmt)
        project = result.scalar_one_or_none()

        if project is None:
            logger.warning("Project '%s' not found", project_id)
            raise ProjectNotFoundError(project_id)

        return project

    async def get_project_by_id(self, project_id: str) -> ProjectModel:
        """Alias for get_project to maintain naming compatibility across callers."""
        return await self.get_project(project_id)

    async def get_project_context(self, project_id: str) -> dict[str, Any]:
        """Retrieve project metadata and source document inventory under project boundary isolation.

        Provides authoritative project context (name, description, and available document names)
        derived directly from the database without duplicating data across systems or stuffing full
        document contents into the prompt.

        Args:
            project_id: Authoritative project identifier.

        Returns:
            dict containing project_id, project_name, project_description, and available_documents.

        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        project = await self.get_project(project_id)
        from models.document import DocumentModel

        stmt = (
            select(DocumentModel.name)
            .where(DocumentModel.project_id == project_id)
            .order_by(DocumentModel.name)
        )
        result = await self._session.execute(stmt)
        document_names = list(result.scalars().all())

        return {
            "project_id": project.id,
            "project_name": project.name,
            "project_description": project.description,
            "available_documents": document_names,
        }

    async def list_projects(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProjectModel]:
        """List all projects ordered by creation date descending.
        
        Args:
            limit: Maximum number of projects to return (1..100).
            offset: Number of records to skip.
            
        Returns:
            List of ProjectModel instances.
        """
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        stmt = (
            select(ProjectModel)
            .order_by(ProjectModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ProjectModel:
        """Update mutable fields of an existing project.
        
        Args:
            project_id: Project identifier.
            name: Optional new project name.
            description: Optional new project description.
            
        Returns:
            Updated ProjectModel instance.
            
        Raises:
            ProjectNotFoundError: If project does not exist.
            InvalidProjectDataError: If updated name is empty or invalid.
        """
        project = await self.get_project(project_id)

        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise InvalidProjectDataError("Project name cannot be empty or whitespace only.")
            project.name = clean_name

        if description is not None:
            project.description = description

        await self._session.flush()
        logger.info("Updated project id: %s", project_id)
        return project

    async def delete_project(self, project_id: str) -> None:
        """Delete a project and cascade deletion to all related child entities.
        
        PostgreSQL foreign keys handle cascading deletes to documents, versions, chunks,
        conversations, and messages. If storage is configured, associated objects in object
        storage are cleaned up on a best-effort basis.
        
        Args:
            project_id: Project identifier.
            
        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        project = await self.get_project(project_id)

        # Best-effort cleanup of project objects in Supabase Storage
        if self._storage is not None:
            try:
                prefix = f"{project_id}/"
                objects = await self._storage.list_objects(prefix=prefix, limit=1000)
                if objects:
                    paths = [obj.path for obj in objects]
                    deleted = await self._storage.delete_many(paths)
                    logger.info(
                        "Deleted %d storage objects under prefix '%s' for project %s",
                        len(deleted),
                        prefix,
                        project_id,
                    )
            except Exception as exc:
                logger.warning(
                    "Best-effort storage cleanup failed during deletion of project %s: %s",
                    project_id,
                    exc,
                )

        # Clean up vector embeddings in Qdrant for this project
        if self._vector_store is not None:
            try:
                await self._vector_store.delete_by_filter(project_id=project_id)
                logger.info("Deleted vector points in Qdrant for project '%s'", project_id)
            except Exception as exc:
                logger.warning(
                    "Vector store cleanup failed during deletion of project %s: %s",
                    project_id,
                    exc,
                )

        await self._session.delete(project)
        await self._session.flush()
        logger.info("Deleted project id: %s", project_id)
