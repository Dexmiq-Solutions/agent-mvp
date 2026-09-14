"""Base interfaces and context abstractions for document metadata enrichment."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag.acquisition.formats import DocumentType
from rag.chunking.models import DocumentChunk


@dataclass(frozen=True)
class DocumentEnrichmentContext:
    """Document-level context providing O(1) metadata access during chunk enrichment."""

    document_id: str
    project_id: str
    document_type: DocumentType
    total_chunks: int
    document_version_id: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    parser_metadata: dict[str, Any] = field(default_factory=dict)


class BaseMetadataEnricher(ABC):
    """Abstract base interface for metadata enrichment rules and providers.

    Supports the hybrid architecture: deterministic rules today, and future
    heuristic or model-based enrichers tomorrow, without altering the pipeline.
    """

    @property
    @abstractmethod
    def enricher_name(self) -> str:
        """Return the unique name of this enricher."""

    @abstractmethod
    def enrich(
        self,
        chunk: DocumentChunk,
        context: DocumentEnrichmentContext,
    ) -> dict[str, Any]:
        """Derive or enrich metadata for a single chunk within its document context.

        Args:
            chunk: The DocumentChunk to enrich.
            context: Document-level context (total chunks, source info, etc.).

        Returns:
            Dictionary of enriched metadata key-value pairs.
        """
