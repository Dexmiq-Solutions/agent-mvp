"""FastAPI dependency injection providers for application services."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db_session
from services.document_service import DocumentService
from services.project_service import ProjectService
from storage.object import BaseObjectStorage, get_object_storage


def get_storage() -> BaseObjectStorage:
    """Dependency provider returning the default configured BaseObjectStorage instance."""
    return get_object_storage()


def get_project_service(
    session: AsyncSession = Depends(get_db_session),
    storage: BaseObjectStorage = Depends(get_storage),
) -> ProjectService:
    """Dependency provider for ProjectService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        
    Returns:
        Configured ProjectService instance.
    """
    return ProjectService(session=session, storage=storage)


def get_document_service(
    session: AsyncSession = Depends(get_db_session),
    storage: BaseObjectStorage = Depends(get_storage),
) -> DocumentService:
    """Dependency provider for DocumentService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        
    Returns:
        Configured DocumentService instance.
    """
    return DocumentService(session=session, storage=storage)
