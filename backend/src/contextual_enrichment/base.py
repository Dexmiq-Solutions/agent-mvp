"""Abstract base interfaces and context representations for contextual enrichment."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from acquisition.formats import DocumentType
from metadata_enrichment.models import EnrichedChunk


@dataclass(frozen=True)
class DocumentContext:
    """Immutable document-level contextual representation for chunks within an indexing run.

    Provides high-level document context (title, storage path, document type, structural
    overview) to contextual enrichers while strictly maintaining project isolation.
    """

    document_id: str
    project_id: str
    document_type: DocumentType
    total_chunks: int
    document_version_id: str | None = None
    original_filename: str | None = None
    source_storage_path: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    headings_hierarchy: tuple[str, ...] = ()
    document_summary: str | None = None


class BaseContextProvider(ABC):
    """Abstract interface for contextual enrichment providers and strategies.

    Decouples contextual enrichment from specific models, LLM providers (e.g. OpenAI,
    OpenRouter, Anthropic), or deterministic heuristics.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name or identifier of this context provider strategy."""

    @abstractmethod
    async def generate_context(
        self,
        chunk: EnrichedChunk,
        context: DocumentContext,
    ) -> str | None:
        """Generate a concise contextual description for a single chunk.

        Args:
            chunk: EnrichedChunk carrying original text, provenance, and structural metadata.
            context: DocumentContext providing document-level metadata and hierarchy.

        Returns:
            str | None: Generated contextual description, or None if no meaningful context
                could be generated.

        Raises:
            ContextualEnrichmentProviderError: If an external provider/model call fails.
        """

    async def generate_context_batch(
        self,
        chunks: list[EnrichedChunk],
        context: DocumentContext,
    ) -> list[str | None]:
        """Generate contextual descriptions for a batch of chunks.

        Default implementation processes chunks sequentially or concurrently. Subclasses
        can override for bulk model requests.

        Args:
            chunks: List of EnrichedChunk instances.
            context: DocumentContext providing document-level metadata and hierarchy.

        Returns:
            list[str | None]: Context descriptions corresponding 1-to-1 with input chunks.
        """
        results: list[str | None] = []
        for chunk in chunks:
            res = await self.generate_context(chunk, context)
            results.append(res)
        return results
