"""Ingestion package for bringing acquired source documents into the indexing pipeline."""

from typing import Optional

from rag.acquisition.formats import DocumentType
from core.config import Settings
from exceptions.ingestion import (
    DocumentRetrievalError,
    IngestionConfigurationError,
    IngestionError,
    InvalidIngestionInputError,
    UnsupportedDocumentTypeError,
)
from rag.ingestion.base import BaseIngestionService, IngestionSourceInput
from rag.ingestion.models import DocumentSourceReference, IngestedDocument
from rag.ingestion.service import StorageIngestionService
from rag.ingestion.validators import (
    resolve_and_validate_document_type,
    resolve_and_verify_storage_path,
    validate_project_id,
)
from storage.object import BaseObjectStorage

_default_ingestion_service: Optional[BaseIngestionService] = None


def get_ingestion_service(
    storage: Optional[BaseObjectStorage] = None,
    settings: Optional[Settings] = None,
) -> BaseIngestionService:
    """Get or create an ingestion service instance.
    
    If no custom storage or settings are provided, returns a cached singleton instance.
    
    Args:
        storage: Optional BaseObjectStorage implementation override.
        settings: Optional Settings override.
        
    Returns:
        BaseIngestionService instance (StorageIngestionService).
    """
    global _default_ingestion_service
    if storage is not None:
        return StorageIngestionService(storage=storage, settings=settings)

    if _default_ingestion_service is None:
        _default_ingestion_service = StorageIngestionService(settings=settings)
    return _default_ingestion_service


def reset_ingestion_service() -> None:
    """Reset the cached default ingestion service instance. Useful for tests."""
    global _default_ingestion_service
    _default_ingestion_service = None


__all__ = [
    # Interfaces and Services
    "BaseIngestionService",
    "StorageIngestionService",
    "IngestionSourceInput",
    "get_ingestion_service",
    "reset_ingestion_service",
    # Models
    "DocumentSourceReference",
    "IngestedDocument",
    "DocumentType",
    # Validators and Utilities
    "resolve_and_validate_document_type",
    "validate_project_id",
    "resolve_and_verify_storage_path",
    # Exceptions
    "IngestionError",
    "IngestionConfigurationError",
    "InvalidIngestionInputError",
    "UnsupportedDocumentTypeError",
    "DocumentRetrievalError",
]
