"""Domain models and configuration for the document chunking layer."""

from dataclasses import dataclass, field
from typing import Any

from acquisition.formats import DocumentType
from chunking.sizer import BaseChunkSizer, CharacterChunkSizer
from cleaning.models import CleaningReport
from exceptions.chunking import ChunkingConfigurationError
from normalization.models import NormalizationReport
from parsing.models import ElementType


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration options for structure-aware document chunking."""

    max_chunk_size: int = 1000
    min_chunk_size: int = 50
    chunk_overlap: int = 0
    preserve_hierarchy: bool = True
    sizer: BaseChunkSizer = field(default_factory=CharacterChunkSizer)

    def __post_init__(self) -> None:
        """Validate configuration parameters upon instantiation."""
        if self.max_chunk_size <= 0:
            raise ChunkingConfigurationError(
                f"max_chunk_size must be a positive integer, got {self.max_chunk_size}."
            )
        if self.min_chunk_size < 0:
            raise ChunkingConfigurationError(
                f"min_chunk_size cannot be negative, got {self.min_chunk_size}."
            )
        if self.min_chunk_size > self.max_chunk_size:
            raise ChunkingConfigurationError(
                f"min_chunk_size ({self.min_chunk_size}) cannot exceed max_chunk_size ({self.max_chunk_size})."
            )
        if self.chunk_overlap < 0:
            raise ChunkingConfigurationError(
                f"chunk_overlap cannot be negative, got {self.chunk_overlap}."
            )
        if self.chunk_overlap >= self.max_chunk_size:
            raise ChunkingConfigurationError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than max_chunk_size ({self.max_chunk_size})."
            )


@dataclass(frozen=True)
class DocumentChunk:
    """Represents an independently retrievable, structure-aware document chunk.

    Preserves source hierarchy, heading context, element associations, and provenance
    without altering, normalizing, or paraphrasing the original content.
    """

    chunk_id: str
    document_id: str
    project_id: str
    content: str
    index: int
    document_version_id: str | None = None
    section_path: tuple[str, ...] = ()
    heading: str | None = None
    heading_level: int | None = None
    parent_element_id: str | None = None
    parent_chunk_id: str | None = None
    source_element_ids: tuple[str, ...] = ()
    element_types: tuple[ElementType, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize document chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "document_version_id": self.document_version_id,
            "content": self.content,
            "index": self.index,
            "section_path": list(self.section_path),
            "heading": self.heading,
            "heading_level": self.heading_level,
            "parent_element_id": self.parent_element_id,
            "parent_chunk_id": self.parent_chunk_id,
            "source_element_ids": list(self.source_element_ids),
            "element_types": [et.value for et in self.element_types],
            "metadata": dict(self.metadata),
        }

    def to_representation_text(self) -> str:
        """Return the text representation for downstream embedding and sparse encoders."""
        return self.content

    def __repr__(self) -> str:
        """Safe representation omitting raw document chunk text."""
        return (
            f"DocumentChunk(chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"index={self.index}, "
            f"content_length={len(self.content)}, "
            f"section_path={self.section_path!r}, "
            f"heading={self.heading!r})"
        )


# Canonical alias for convenience and brevity
Chunk = DocumentChunk


@dataclass(frozen=True)
class ChunkingReport:
    """Summary report detailing the results and metrics of the chunking stage.

    Provides explainability and observability without persisting raw text contents.
    """

    total_input_elements: int = 0
    total_chunks: int = 0
    recursively_split_count: int = 0
    structural_chunks_count: int = 0
    fallback_chunks_count: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunking report to a dictionary."""
        return {
            "total_input_elements": self.total_input_elements,
            "total_chunks": self.total_chunks,
            "recursively_split_count": self.recursively_split_count,
            "structural_chunks_count": self.structural_chunks_count,
            "fallback_chunks_count": self.fallback_chunks_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def __repr__(self) -> str:
        """Safe string representation omitting raw document contents."""
        return (
            f"ChunkingReport(total_input_elements={self.total_input_elements}, "
            f"total_chunks={self.total_chunks}, "
            f"recursively_split={self.recursively_split_count}, "
            f"structural_chunks={self.structural_chunks_count}, "
            f"fallback_chunks={self.fallback_chunks_count}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms)"
        )


@dataclass(frozen=True)
class ChunkedDocument:
    """Container representing the output of the document chunking stage.

    Holds ordered, structure-aware chunks ready for metadata enrichment and embedding
    generation, preserving all source metadata and previous pipeline reports.
    """

    document_id: str
    project_id: str
    document_type: DocumentType
    chunks: list[DocumentChunk] = field(default_factory=list)
    document_version_id: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    parser_metadata: dict[str, Any] = field(default_factory=dict)
    cleaning_report: CleaningReport | None = None
    normalization_report: NormalizationReport | None = None
    chunking_report: ChunkingReport = field(default_factory=ChunkingReport)

    @property
    def total_chunks(self) -> int:
        """Return the total number of chunks generated for this document."""
        return len(self.chunks)

    def to_dict(
        self,
        include_chunks: bool = True,
        include_reports: bool = True,
    ) -> dict[str, Any]:
        """Serialize chunked document metadata, chunks, and reports to a dictionary.

        Args:
            include_chunks: Whether to include the list of serialized DocumentChunks.
            include_reports: Whether to include cleaning, normalization, and chunking reports.
        """
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "document_type": self.document_type.value,
            "document_version_id": self.document_version_id,
            "total_chunks": self.total_chunks,
            "source_metadata": dict(self.source_metadata),
            "parser_metadata": dict(self.parser_metadata),
        }
        if include_chunks:
            data["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        if include_reports:
            if self.cleaning_report is not None:
                data["cleaning_report"] = self.cleaning_report.to_dict(include_decisions=False)
            if self.normalization_report is not None:
                data["normalization_report"] = self.normalization_report.to_dict(include_decisions=False)
            data["chunking_report"] = self.chunking_report.to_dict()
        return data

    def __repr__(self) -> str:
        """Safe string representation omitting raw document chunks."""
        return (
            f"ChunkedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_chunks={self.total_chunks}, "
            f"document_version_id={self.document_version_id!r})"
        )
