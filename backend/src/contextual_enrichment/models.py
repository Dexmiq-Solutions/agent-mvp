"""Domain models, configuration, and data structures for contextual enrichment."""

from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from contextual_enrichment.base import BaseContextProvider
from exceptions.contextual_enrichment import ContextualEnrichmentConfigurationError
from metadata_enrichment.models import EnrichedChunk, EnrichedDocument


@dataclass(frozen=True)
class ContextualEnrichmentConfig:
    """Configuration options for document-level contextual enrichment."""

    enabled: bool = False
    strategy: str = "structured"
    provider: BaseContextProvider | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters upon instantiation."""
        valid_strategies = {"structured", "disabled", "custom"}
        if self.strategy not in valid_strategies and self.provider is None:
            raise ContextualEnrichmentConfigurationError(
                f"Invalid contextual enrichment strategy '{self.strategy}'. "
                f"Must be one of: {', '.join(sorted(valid_strategies))} or provide a custom provider."
            )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ContextualEnrichmentConfig":
        """Construct configuration from application Settings."""
        if settings is None:
            settings = get_settings()
        return cls(
            enabled=settings.CONTEXTUAL_ENRICHMENT_ENABLED,
            strategy=settings.CONTEXTUAL_ENRICHMENT_STRATEGY,
        )

    @classmethod
    def from_env(cls) -> "ContextualEnrichmentConfig":
        """Construct configuration by reading environment variables via application Settings."""
        return cls.from_settings()


@dataclass(frozen=True)
class ContextuallyEnrichedChunk(EnrichedChunk):
    """Represents an enriched chunk with attached contextual representation for embeddings.

    Subclasses EnrichedChunk (and DocumentChunk) to maintain 100% downstream compatibility.
    Strictly preserves verbatim original chunk content while exposing the contextualized
    representation for embedding generation.
    """

    context_text: str | None = None
    is_contextually_enriched: bool = False
    context_strategy: str | None = None

    @property
    def contextual_content(self) -> str:
        """Return representation used for vector embedding generation.

        When contextual enrichment is applied, formats context alongside the verbatim content.
        When disabled or if no context was generated, returns the original content verbatim.
        """
        if self.is_contextually_enriched and self.context_text:
            return f"{self.context_text}\n\n{self.content}"
        return self.content

    def to_embedding_text(self) -> str:
        """Return the text representation to be embedded."""
        return self.contextual_content

    def to_dict(self) -> dict[str, Any]:
        """Serialize contextually enriched chunk to a dictionary."""
        base = super().to_dict()
        base["context_text"] = self.context_text
        base["is_contextually_enriched"] = self.is_contextually_enriched
        base["context_strategy"] = self.context_strategy
        base["contextual_content"] = self.contextual_content
        return base

    def __repr__(self) -> str:
        """Safe string representation omitting raw chunk text."""
        return (
            f"ContextuallyEnrichedChunk(chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"index={self.index}, "
            f"enriched={self.is_contextually_enriched}, "
            f"strategy={self.context_strategy!r})"
        )


@dataclass(frozen=True)
class ContextualEnrichmentReport:
    """Summary report detailing the results and execution metrics of contextual enrichment."""

    enabled: bool = False
    total_chunks: int = 0
    enriched_chunks_count: int = 0
    strategy_used: str = "none"
    provider_name: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize contextual enrichment report to a dictionary."""
        return {
            "enabled": self.enabled,
            "total_chunks": self.total_chunks,
            "enriched_chunks_count": self.enriched_chunks_count,
            "strategy_used": self.strategy_used,
            "provider_name": self.provider_name,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def __repr__(self) -> str:
        """Safe representation omitting sensitive details."""
        return (
            f"ContextualEnrichmentReport(enabled={self.enabled}, "
            f"total_chunks={self.total_chunks}, "
            f"enriched={self.enriched_chunks_count}, "
            f"strategy={self.strategy_used!r}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms)"
        )


@dataclass(frozen=True)
class ContextuallyEnrichedDocument(EnrichedDocument):
    """Container representing the output of the contextual enrichment stage.

    Subclasses EnrichedDocument to maintain complete structural compatibility with
    subsequent stages, holding ContextuallyEnrichedChunks and ContextualEnrichmentReport.
    """

    contextual_enrichment_report: ContextualEnrichmentReport = field(
        default_factory=ContextualEnrichmentReport
    )

    def to_dict(
        self,
        include_chunks: bool = True,
        include_reports: bool = True,
    ) -> dict[str, Any]:
        """Serialize document metadata, chunks, and reports to a dictionary."""
        data = super().to_dict(
            include_chunks=include_chunks,
            include_reports=include_reports,
        )
        if include_reports:
            data["contextual_enrichment_report"] = self.contextual_enrichment_report.to_dict()
        return data

    def __repr__(self) -> str:
        """Safe string representation omitting raw document contents."""
        return (
            f"ContextuallyEnrichedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_chunks={self.total_chunks}, "
            f"enriched_chunks={self.contextual_enrichment_report.enriched_chunks_count}, "
            f"enabled={self.contextual_enrichment_report.enabled})"
        )
