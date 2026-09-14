"""Document metadata enrichment layer for the RAG indexing pipeline.

Sits strictly between Chunking and downstream Embedding Generation & Storage.
Executes Structure-Aware Hybrid Metadata Enrichment to preserve inherited provenance,
maintain document structure/hierarchy, derive deterministic chunk characteristics,
and produce retrieval-oriented metadata while strictly enforcing tenant isolation.
"""

from exceptions.metadata_enrichment import (
    InvalidMetadataEnrichmentInputError,
    MetadataEnrichmentConfigurationError,
    MetadataEnrichmentError,
    MetadataEnrichmentProcessingError,
)
from rag.metadata_enrichment.base import (
    BaseMetadataEnricher,
    DocumentEnrichmentContext,
)
from rag.metadata_enrichment.models import (
    ChunkContentType,
    ChunkMetadata,
    EnrichedChunk,
    EnrichedDocument,
    MetadataCategory,
    MetadataEnrichmentConfig,
    MetadataEnrichmentReport,
)
from rag.metadata_enrichment.rules import (
    CharacteristicsEnricherRule,
    ProvenanceEnricherRule,
    StructuralEnricherRule,
)
from rag.metadata_enrichment.service import (
    DocumentMetadataEnrichmentService,
    get_metadata_enrichment_service,
    reset_metadata_enrichment_service,
)

__all__ = [
    # Service and lifecycle
    "DocumentMetadataEnrichmentService",
    "get_metadata_enrichment_service",
    "reset_metadata_enrichment_service",
    # Extensible base abstractions
    "BaseMetadataEnricher",
    "DocumentEnrichmentContext",
    # Domain models and reports
    "ChunkMetadata",
    "EnrichedChunk",
    "EnrichedDocument",
    "MetadataEnrichmentReport",
    "MetadataEnrichmentConfig",
    "ChunkContentType",
    "MetadataCategory",
    # Deterministic rules
    "ProvenanceEnricherRule",
    "StructuralEnricherRule",
    "CharacteristicsEnricherRule",
    # Domain exceptions
    "MetadataEnrichmentError",
    "InvalidMetadataEnrichmentInputError",
    "MetadataEnrichmentProcessingError",
    "MetadataEnrichmentConfigurationError",
]
