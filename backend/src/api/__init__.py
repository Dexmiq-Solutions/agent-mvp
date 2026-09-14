"""API routes package exports."""

from fastapi import APIRouter

from api.conversations import router as conversations_router
from api.projects import router as projects_router
from api.retrieval import router as retrieval_router
from api.sources import router as sources_router

api_router = APIRouter()
api_router.include_router(projects_router)
api_router.include_router(sources_router)
api_router.include_router(conversations_router)
api_router.include_router(retrieval_router)

__all__ = [
    "api_router",
    "projects_router",
    "sources_router",
    "conversations_router",
    "retrieval_router",
]

