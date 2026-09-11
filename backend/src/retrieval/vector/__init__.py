"""Vector Search component of the RAG retrieval pipeline."""

from retrieval.vector.service import (
    VectorSearchService,
    get_vector_search_service,
    reset_vector_search_service,
    search_vectors,
)

__all__ = [
    "VectorSearchService",
    "get_vector_search_service",
    "reset_vector_search_service",
    "search_vectors",
]
