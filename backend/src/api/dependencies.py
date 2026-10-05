from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.session import get_db_session
from exceptions.auth import InvalidTokenError
from models.project import ProjectModel
from models.user import UserModel
from services.auth_service import AuthService
from services.document_processing_service import DocumentProcessingService
from services.document_service import DocumentService
from services.project_service import ProjectService
from storage.object import BaseObjectStorage, get_object_storage
from storage.vector import BaseVectorStore
from observability.logging import get_logger

logger = get_logger(__name__)

oauth2_bearer = HTTPBearer(auto_error=True)


def get_auth_service(
    session: AsyncSession = Depends(get_db_session),
) -> AuthService:
    """Dependency provider for AuthService."""
    return AuthService(session=session)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_bearer),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserModel:
    """Validate bearer token and resolve authenticated active user."""
    from core.security import decode_access_token

    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise InvalidTokenError("Token missing subject identifier.")
    return await auth_service.get_user_by_id(user_id)


def get_vector_store_dependency() -> Optional[BaseVectorStore]:
    """Dependency provider returning default BaseVectorStore instance."""
    try:
        from storage.vector import get_vector_store
        return get_vector_store()
    except Exception:
        return None


def get_storage() -> BaseObjectStorage:
    """Dependency provider returning the default configured BaseObjectStorage instance."""
    return get_object_storage()


def get_project_service(
    session: AsyncSession = Depends(get_db_session),
    storage: BaseObjectStorage = Depends(get_storage),
    vector_store: Optional[BaseVectorStore] = Depends(get_vector_store_dependency),
) -> ProjectService:
    """Dependency provider for ProjectService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        vector_store: Injected BaseVectorStore instance.
        
    Returns:
        Configured ProjectService instance.
    """
    return ProjectService(session=session, storage=storage, vector_store=vector_store)


async def get_current_project(
    project_id: str,
    current_user: UserModel = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectModel:
    """Validate project existence and tenant ownership for the current authenticated user."""
    return await project_service.get_project(project_id=project_id, user_id=current_user.id)


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
    vector_store: Optional[BaseVectorStore] = Depends(get_vector_store_dependency),
) -> DocumentService:
    """Dependency provider for DocumentService.
    
    Args:
        session: Injected asynchronous SQLAlchemy session.
        storage: Injected BaseObjectStorage client.
        processing_service: Injected DocumentProcessingService instance.
        vector_store: Injected BaseVectorStore instance.
        
    Returns:
        Configured DocumentService instance.
    """
    settings = get_settings()
    return DocumentService(
        session=session,
        storage=storage,
        processing_service=processing_service,
        auto_process=settings.AUTO_PROCESS_DOCUMENTS,
        vector_store=vector_store,
    )


def get_rag_service_dependency() -> "RAGService":
    """Dependency provider returning configured RAGService singleton instance."""
    from services.rag_service import get_rag_service

    return get_rag_service()


def get_brd_lead_agent() -> "BRDLeadAgent":
    """Dependency provider returning a fresh BRDLeadAgent instance.

    To eliminate cross-request and cross-tenant mutable state leakage (DEF-005),
    this provider does not store a global process-level agent singleton. Each
    request receives its own agent instance, while durable workflow state is
    conversation-scoped and reconstructed per execution turn (DEF-012).
    """
    from agents.brd.agent import BRDLeadAgent
    from tools.diagnostic import echo_diagnostic_tool
    from tools.rag import create_search_project_knowledge_tool

    from exceptions.retrieval import RetrievalError

    try:
        rag_service = get_rag_service_dependency()
        tools = [
            echo_diagnostic_tool,
            create_search_project_knowledge_tool(rag_service=rag_service),
        ]
    except Exception as exc:
        logger.error(
            "Failed to initialize RAG service for BRDLeadAgent: %s. "
            "RAG capability cannot be equipped.",
            exc,
            exc_info=True,
        )
        raise RetrievalError(
            f"RAG service initialization failed: {exc}. "
            "Cannot construct BRDLeadAgent with required project retrieval capability.",
            original_error=exc,
        ) from exc

    return BRDLeadAgent(tools=tools)





