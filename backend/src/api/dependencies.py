"""FastAPI dependency injection providers for application services."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from db.session import get_db_session
from services.document_processing_service import DocumentProcessingService
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


def get_conversation_service(
    session: AsyncSession = Depends(get_db_session),
    project_service: ProjectService = Depends(get_project_service),
) -> "ConversationService":
    """Dependency provider for ConversationService.

    Args:
        session: Injected asynchronous SQLAlchemy session.
        project_service: Injected ProjectService instance.

    Returns:
        Configured ConversationService instance.
    """
    from services.conversation_service import ConversationService

    return ConversationService(session=session, project_service=project_service)


def get_document_processing_service(
    session: AsyncSession = Depends(get_db_session),
    storage: BaseObjectStorage = Depends(get_storage),
) -> DocumentProcessingService:
    """Dependency provider for DocumentProcessingService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        
    Returns:
        Configured DocumentProcessingService instance.
    """
    return DocumentProcessingService(session=session, storage=storage)


def get_document_service(
    session: AsyncSession = Depends(get_db_session),
    storage: BaseObjectStorage = Depends(get_storage),
    processing_service: DocumentProcessingService = Depends(get_document_processing_service),
) -> DocumentService:
    """Dependency provider for DocumentService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        processing_service: Injected DocumentProcessingService instance.
        
    Returns:
        Configured DocumentService instance.
    """
    settings = get_settings()
    return DocumentService(
        session=session,
        storage=storage,
        processing_service=processing_service,
        auto_process=settings.AUTO_PROCESS_DOCUMENTS,
    )


def get_rag_service_dependency() -> "RAGService":
    """Dependency provider returning configured RAGService singleton instance."""
    from services.rag_service import get_rag_service

    return get_rag_service()


def get_generation_service(
    session: AsyncSession = Depends(get_db_session),
    conversation_service: "ConversationService" = Depends(get_conversation_service),
    rag_service: "RAGService" = Depends(get_rag_service_dependency),
) -> "GenerationService":
    """Dependency provider for GenerationService.

    Args:
        session: Injected asynchronous SQLAlchemy session.
        conversation_service: Injected ConversationService instance.
        rag_service: Injected RAGService instance.

    Returns:
        Configured GenerationService instance.
    """
    from services.generation_service import GenerationService

    return GenerationService(
        session=session,
        conversation_service=conversation_service,
        rag_service=rag_service,
    )



