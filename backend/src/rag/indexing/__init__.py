"""RAG Indexing stage package alias re-exporting from indexing."""

from indexing import (
    DocumentIndexingService,
    IndexableRecord,
    IndexingBatchResult,
    IndexingConfig,
    IndexingError,
    IndexingReport,
    generate_point_id,
    get_indexing_service,
    reset_indexing_service,
    extract_representation_text,
    extract_representation_texts,
    generate_sparse_representations,
)

__all__ = [
    "DocumentIndexingService",
    "IndexableRecord",
    "IndexingConfig",
    "IndexingReport",
    "IndexingBatchResult",
    "generate_point_id",
    "get_indexing_service",
    "reset_indexing_service",
    "extract_representation_text",
    "extract_representation_texts",
    "generate_sparse_representations",
    "IndexingError",
]
