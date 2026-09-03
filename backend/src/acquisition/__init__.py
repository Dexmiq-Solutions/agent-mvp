"""Acquisition package for discovering and validating source documents for the indexing pipeline."""

from typing import Optional

from app.core.config import Settings
from acquisition.base import BaseAcquisitionService
from acquisition.formats import (
    SUPPORTED_EXTENSIONS,
    DocumentType,
    get_canonical_mime_type,
    get_document_type,
    is_supported_extension,
    normalize_extension,
)
from acquisition.models import (
    AcquisitionResult,
    AcquisitionStatus,
    SourceDocument,
    SourceReference,
)
from acquisition.service import StorageAcquisitionService
from exceptions.acquisition import (
    AcquisitionConfigurationError,
    AcquisitionError,
    InvalidSourceReferenceError,
    SourceDiscoveryError,
    UnsupportedDocumentError,
)
from storage.object import BaseObjectStorage

_default_acquisition_service: Optional[BaseAcquisitionService] = None


def get_acquisition_service(
    storage: Optional[BaseObjectStorage] = None,
    settings: Optional[Settings] = None,
    batch_size: int = 100,
) -> BaseAcquisitionService:
    """Get or create an acquisition service instance.
    
    If no custom storage or settings are provided, returns a cached singleton instance.
    
    Args:
        storage: Optional BaseObjectStorage implementation override.
        settings: Optional Settings override.
        batch_size: Optional page limit for listing operations from storage.
        
    Returns:
        BaseAcquisitionService instance (StorageAcquisitionService).
    """
    global _default_acquisition_service
    if storage is not None:
        return StorageAcquisitionService(storage=storage, settings=settings, batch_size=batch_size)

    if _default_acquisition_service is None:
        _default_acquisition_service = StorageAcquisitionService(settings=settings, batch_size=batch_size)
    return _default_acquisition_service


def reset_acquisition_service() -> None:
    """Reset the cached default acquisition service instance. Useful for tests."""
    global _default_acquisition_service
    _default_acquisition_service = None


__all__ = [
    # Interfaces and Services
    "BaseAcquisitionService",
    "StorageAcquisitionService",
    "get_acquisition_service",
    "reset_acquisition_service",
    # Models
    "SourceReference",
    "SourceDocument",
    "AcquisitionResult",
    "AcquisitionStatus",
    "DocumentType",
    # Formats & Utilities
    "SUPPORTED_EXTENSIONS",
    "is_supported_extension",
    "get_document_type",
    "get_canonical_mime_type",
    "normalize_extension",
    # Exceptions
    "AcquisitionError",
    "AcquisitionConfigurationError",
    "SourceDiscoveryError",
    "UnsupportedDocumentError",
    "InvalidSourceReferenceError",
]
