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
from rag.indexing.identity import INDEXING_NAMESPACE, generate_point_id
from rag.indexing.models import (
    IndexableRecord,
    IndexingBatchResult,
    IndexingConfig,
    IndexingReport,
)
from rag.indexing.orchestrator import (
    EndToEndIndexingService,
    IndexingPipelineReport,
    get_end_to_end_indexing_service,
    reset_end_to_end_indexing_service,
)
from rag.indexing.persistence import (
    ChunkPersistenceService,
    get_chunk_persistence_service,
    reset_chunk_persistence_service,
)
from rag.indexing.representations import (
    extract_representation_text,
    extract_representation_texts,
    generate_sparse_representations,
)
from rag.indexing.service import (
    DocumentIndexingService,
    get_indexing_service,
    reset_indexing_service,
)
from storage.vector import BaseVectorStore


__all__ = [
    # Main Services & Factories
    "DocumentIndexingService",
    "get_indexing_service",
    "reset_indexing_service",
    "EndToEndIndexingService",
    "get_end_to_end_indexing_service",
    "reset_end_to_end_indexing_service",
    "ChunkPersistenceService",
    "get_chunk_persistence_service",
    "reset_chunk_persistence_service",
    # Reports & Models & Config
    "IndexingPipelineReport",
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
