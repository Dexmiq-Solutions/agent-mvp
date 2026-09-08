"""Document chunking layer for the RAG indexing pipeline.

Sits strictly between Normalization and downstream Metadata Enrichment & Embeddings.
Executes Structure-Aware Chunking with Recursive Splitting for Oversized Sections
to produce meaningful, independently retrievable chunks while preserving document
structure, hierarchy, context, and provenance.
"""

from chunking.models import (
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    ChunkingReport,
    DocumentChunk,
)
from chunking.service import (
    DocumentChunkingService,
    get_chunking_service,
    reset_chunking_service,
)
from chunking.sizer import (
    BaseChunkSizer,
    CharacterChunkSizer,
    WordChunkSizer,
)
from chunking.splitter import (
    RecursiveTextSplitter,
    SectionNode,
    StructureAwareChunker,
    build_section_tree,
)
from exceptions.chunking import (
    ChunkingConfigurationError,
    ChunkingError,
    ChunkingProcessingError,
    InvalidChunkingInputError,
)

__all__ = [
    # Service and Orchestrator
    "DocumentChunkingService",
    "get_chunking_service",
    "reset_chunking_service",
    # Domain Models
    "DocumentChunk",
    "Chunk",
    "ChunkedDocument",
    "ChunkingReport",
    "ChunkingConfig",
    # Sizing Abstractions
    "BaseChunkSizer",
    "CharacterChunkSizer",
    "WordChunkSizer",
    # Splitting Logic
    "SectionNode",
    "build_section_tree",
    "RecursiveTextSplitter",
    "StructureAwareChunker",
    # Domain Exceptions
    "ChunkingError",
    "InvalidChunkingInputError",
    "ChunkingProcessingError",
    "ChunkingConfigurationError",
]
