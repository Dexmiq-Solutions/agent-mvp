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
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
    InvalidDocumentDataError,
    ProjectDocumentMismatchError,
)
from exceptions.project import (
    InvalidProjectDataError,
    ProjectNotFoundError,
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
