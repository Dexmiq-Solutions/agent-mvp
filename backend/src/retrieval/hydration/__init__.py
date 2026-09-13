"""Chunk Fetching / Hydration stage for retrieval pipeline.

Resolves ranked chunk references from Reranking into authoritative stored records
from the PostgreSQL Content Store, preserving reranking order, retrieval scores,
provenance, and attaching stored text and metadata for Context Assembly.
"""

from exceptions.retrieval import (
    ChunkHydrationError,
    ChunkHydrationValidationError,
    ChunkNotFoundError,
    DatabaseRetrievalError,
    ProjectBoundaryViolationError,
)
from retrieval.hydration.config import ChunkHydrationConfig
from retrieval.hydration.repository import (
    BaseChunkRepository,
    SQLAlchemyChunkRepository,
    get_chunk_repository,
    reset_chunk_repository,
)
from retrieval.hydration.service import (
    ChunkHydrationService,
    get_chunk_hydration_service,
    hydrate_candidates,
    reset_chunk_hydration_service,
)

__all__ = [
    "ChunkHydrationConfig",
    "BaseChunkRepository",
    "SQLAlchemyChunkRepository",
    "get_chunk_repository",
    "reset_chunk_repository",
    "ChunkHydrationService",
    "get_chunk_hydration_service",
    "reset_chunk_hydration_service",
    "hydrate_candidates",
    "ChunkHydrationError",
    "ChunkHydrationValidationError",
    "ChunkNotFoundError",
    "DatabaseRetrievalError",
    "ProjectBoundaryViolationError",
]
