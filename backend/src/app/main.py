from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

from api import api_router
from db.session import dispose_engine
from exceptions.database import DatabaseError
from exceptions.document import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentStorageSourceError,
    DocumentVersionMismatchError,
    DocumentVersionNotFoundError,
    InvalidDocumentDataError,
    InvalidDocumentStateTransitionError,
    ProjectDocumentMismatchError,
)
from exceptions.project import (
    InvalidProjectDataError,
    ProjectNotFoundError,
)
from exceptions.conversation import (
    ConversationError,
    ConversationMessageMismatchError,
    ConversationNotFoundError,
    InvalidConversationDataError,
    InvalidMessageDataError,
    InvalidMessageRoleError,
    MessageNotFoundError,
    ProjectConversationMismatchError,
)

from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    ProjectBoundaryViolationError,
    QueryLengthExceededError,
    RetrievalError,
)
from exceptions.storage import StorageError
from fastapi.responses import JSONResponse
from fastapi import Request, status

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup and shutdown events."""
    settings = get_settings()
    setup_logging()
    logger.info(
        "Starting %s (v%s) in '%s' environment",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)
    await dispose_engine()


def create_application() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # --------------------------------------------------------------------------
    # Exception Handlers
    # --------------------------------------------------------------------------
    @application.exception_handler(ProjectNotFoundError)
    async def project_not_found_handler(request: Request, exc: ProjectNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(DocumentVersionNotFoundError)
    async def document_version_not_found_handler(
        request: Request, exc: DocumentVersionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(ProjectDocumentMismatchError)
    async def project_document_mismatch_handler(
        request: Request, exc: ProjectDocumentMismatchError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Document '{exc.document_id}' not found in project '{exc.expected_project_id}'."},
        )

    @application.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        request: Request, exc: ConversationNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(ProjectConversationMismatchError)
    async def project_conversation_mismatch_handler(
        request: Request, exc: ProjectConversationMismatchError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Conversation '{exc.conversation_id}' not found in project '{exc.expected_project_id}'."},
        )

    @application.exception_handler(MessageNotFoundError)
    async def message_not_found_handler(
        request: Request, exc: MessageNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(ConversationMessageMismatchError)
    async def conversation_message_mismatch_handler(
        request: Request, exc: ConversationMessageMismatchError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Message '{exc.message_id}' not found in conversation '{exc.expected_conversation_id}'."},
        )

    @application.exception_handler(InvalidConversationDataError)
    async def invalid_conversation_data_handler(
        request: Request, exc: InvalidConversationDataError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(InvalidMessageDataError)
    async def invalid_message_data_handler(
        request: Request, exc: InvalidMessageDataError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(InvalidMessageRoleError)
    async def invalid_message_role_handler(
        request: Request, exc: InvalidMessageRoleError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )


    @application.exception_handler(InvalidProjectDataError)
    async def invalid_project_data_handler(request: Request, exc: InvalidProjectDataError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(InvalidDocumentDataError)
    async def invalid_document_data_handler(request: Request, exc: InvalidDocumentDataError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(InvalidDocumentStateTransitionError)
    async def invalid_document_state_transition_handler(
        request: Request, exc: InvalidDocumentStateTransitionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @application.exception_handler(DocumentVersionMismatchError)
    async def document_version_mismatch_handler(
        request: Request, exc: DocumentVersionMismatchError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @application.exception_handler(DocumentProcessingError)
    async def document_processing_error_handler(
        request: Request, exc: DocumentProcessingError
    ) -> JSONResponse:
        logger.error("Document processing error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError) -> JSONResponse:
        logger.error("Storage error during request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "An object storage operation failed. Please try again."},
        )

    @application.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:
        logger.error("Database error during request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "A database error occurred. Please try again later."},
        )

    @application.exception_handler(InvalidQueryError)
    async def invalid_query_handler(request: Request, exc: InvalidQueryError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(ProjectBoundaryViolationError)
    async def project_boundary_violation_handler(
        request: Request, exc: ProjectBoundaryViolationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @application.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
        logger.error("Retrieval pipeline failure: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Retrieval pipeline failed: {exc.message}"},
        )

    # --------------------------------------------------------------------------
    # API Routers
    # --------------------------------------------------------------------------
    application.include_router(api_router)

    @application.get("/health", tags=["Health"])
    async def health_check() -> dict[str, Any]:
        """Basic health check endpoint."""
        return {
            "status": "ok",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @application.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root status endpoint."""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "environment": settings.ENVIRONMENT,
        }

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
