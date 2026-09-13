"""Indexing package providing DocumentIndexingService and Qdrant vector storage integration."""

from typing import Optional

from app.core.config import Settings
from exceptions.indexing import (
    IndexingConfigurationError,
    IndexingConnectionError,
    IndexingError,
    IndexingOperationError,
    IndexingPartialFailureError,
    InvalidIndexingInputError,
)
from indexing.identity import INDEXING_NAMESPACE, generate_point_id
from indexing.models import (
    IndexableRecord,
    IndexingBatchResult,
    IndexingConfig,
    IndexingReport,
)
from indexing.representations import (
    extract_representation_text,
    extract_representation_texts,
    generate_sparse_representations,
)
from indexing.service import DocumentIndexingService
from storage.vector import BaseVectorStore

_default_indexing_service: Optional[DocumentIndexingService] = None


def get_indexing_service(
    vector_store: Optional[BaseVectorStore] = None,
    config: Optional[IndexingConfig] = None,
    settings: Optional[Settings] = None,
) -> DocumentIndexingService:
    """Get or create the singleton DocumentIndexingService instance.

    If custom parameters are supplied, creates and returns a new instance.
    Otherwise, returns the cached singleton instance.

    Args:
        vector_store: Optional BaseVectorStore instance override.
        config: Optional IndexingConfig override.
        settings: Optional Settings override.

    Returns:
        DocumentIndexingService: Configured document indexing service.
    """
    global _default_indexing_service

    if vector_store is not None or config is not None:
        return DocumentIndexingService(
            vector_store=vector_store,
            config=config,
            settings=settings,
        )

    if _default_indexing_service is None:
        _default_indexing_service = DocumentIndexingService(settings=settings)

    return _default_indexing_service


def reset_indexing_service() -> None:
    """Reset the cached default indexing service instance. Useful for tests."""
    global _default_indexing_service
    _default_indexing_service = None


__all__ = [
    # Main Service & Factory
    "DocumentIndexingService",
    "get_indexing_service",
    "reset_indexing_service",
    # Models & Config
    "IndexingConfig",
    "IndexableRecord",
    "IndexingBatchResult",
    "IndexingReport",
    # Identity
    "generate_point_id",
    "INDEXING_NAMESPACE",
    # Representation Generation
    "extract_representation_text",
    "extract_representation_texts",
    "generate_sparse_representations",
    # Exceptions
    "IndexingError",
    "InvalidIndexingInputError",
    "IndexingConfigurationError",
    "IndexingConnectionError",
    "IndexingOperationError",
    "IndexingPartialFailureError",
]
