"""Domain models, configuration, and data structures for metadata enrichment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag.acquisition.formats import DocumentType
from rag.chunking.models import ChunkedDocument, DocumentChunk
from exceptions.metadata_enrichment import MetadataEnrichmentConfigurationError
from rag.metadata_enrichment.base import BaseMetadataEnricher


class MetadataCategory(str, Enum):
    """Categories of metadata captured and enriched for chunks."""

    PROVENANCE = "provenance"
    STRUCTURAL = "structural"
    SOURCE = "source"
    CHARACTERISTICS = "characteristics"
    RETRIEVAL = "retrieval"


class ChunkContentType(str, Enum):
    """Structural content type classifications for document chunks."""

    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"


@dataclass(frozen=True)
class MetadataEnrichmentConfig:
    """Configuration options for structure-aware hybrid metadata enrichment."""

    heading_separator: str = " > "
    enable_provenance: bool = True
    enable_structural: bool = True
    enable_characteristics: bool = True
    custom_enrichers: tuple[BaseMetadataEnricher, ...] = ()

    def __post_init__(self) -> None:
        """Validate configuration parameters upon instantiation."""
        if not self.heading_separator:
            raise MetadataEnrichmentConfigurationError(
                "heading_separator cannot be empty."
            )


@dataclass(frozen=True)
class ChunkMetadata:
    """Structured, strongly-typed container capturing all metadata categories for a chunk.

    Guarantees tenant isolation, structural preservation, deterministic metrics,
    and downstream retrieval compatibility without mutating raw chunk text.
    """

    # 1. Identity & Provenance
    project_id: str
    document_id: str
    chunk_id: str
    document_version_id: str | None = None
    source_element_ids: tuple[str, ...] = ()
    document_type: str = "unknown"
    source_storage_path: str | None = None
    original_filename: str | None = None

    # 2. Structural Metadata
    section_path: tuple[str, ...] = ()
    heading: str | None = None
    heading_level: int | None = None
    heading_path_str: str = ""
    hierarchy_depth: int = 0
    parent_element_id: str | None = None
    parent_chunk_id: str | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    relative_position: float = 0.0

    # 3. Chunk Characteristics
    character_count: int = 0
    word_count: int = 0
    line_count: int = 0
    content_type: str = ChunkContentType.PROSE.value
    has_code: bool = False
    has_table: bool = False
    has_list: bool = False
    is_header_chunk: bool = False
    language: str | None = None

    # 4. Source & Extensible Attributes
    source_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize structured chunk metadata to a grouped dictionary."""
        return {
            "provenance": {
                "project_id": self.project_id,
                "document_id": self.document_id,
                "chunk_id": self.chunk_id,
                "document_version_id": self.document_version_id,
                "source_element_ids": list(self.source_element_ids),
                "document_type": self.document_type,
                "source_storage_path": self.source_storage_path,
                "original_filename": self.original_filename,
            },
            "structural": {
                "section_path": list(self.section_path),
                "heading": self.heading,
                "heading_level": self.heading_level,
                "heading_path_str": self.heading_path_str,
                "hierarchy_depth": self.hierarchy_depth,
                "parent_element_id": self.parent_element_id,
                "parent_chunk_id": self.parent_chunk_id,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "relative_position": self.relative_position,
            },
            "characteristics": {
                "character_count": self.character_count,
                "word_count": self.word_count,
                "line_count": self.line_count,
                "content_type": self.content_type,
                "has_code": self.has_code,
                "has_table": self.has_table,
                "has_list": self.has_list,
                "is_header_chunk": self.is_header_chunk,
                "language": self.language,
            },
            "source_metadata": dict(self.source_metadata),
            "extra": dict(self.extra),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary suitable for storage and retrieval payloads."""
        flat: dict[str, Any] = {
            # Identity & Provenance
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_version_id": self.document_version_id,
            "source_element_ids": list(self.source_element_ids),
            "document_type": self.document_type,
            "source_storage_path": self.source_storage_path,
            "original_filename": self.original_filename,
            # Structural
            "section_path": list(self.section_path),
            "heading": self.heading,
            "heading_level": self.heading_level,
            "heading_path_str": self.heading_path_str,
            "hierarchy_depth": self.hierarchy_depth,
            "parent_element_id": self.parent_element_id,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "relative_position": self.relative_position,
            # Characteristics
            "character_count": self.character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "content_type": self.content_type,
            "has_code": self.has_code,
            "has_table": self.has_table,
            "has_list": self.has_list,
            "is_header_chunk": self.is_header_chunk,
            "language": self.language,
        }
        # Merge extra attributes without overwriting core fields
        for k, v in self.extra.items():
            if k not in flat:
                flat[k] = v
        return flat

    def to_vector_payload(self) -> dict[str, Any]:
        """Serialize into a clean dictionary optimized for vector database (Qdrant) payload.

        Enforces project isolation and supplies key filtering and ranking attributes.
        """
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_version_id": self.document_version_id,
            "document_type": self.document_type,
            "section_path": list(self.section_path),
            "heading": self.heading,
            "heading_path_str": self.heading_path_str,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "relative_position": self.relative_position,
            "content_type": self.content_type,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "has_code": self.has_code,
            "has_table": self.has_table,
        }
        if self.language:
            payload["language"] = self.language
        if self.original_filename:
            payload["original_filename"] = self.original_filename
        if self.source_storage_path:
            payload["source_storage_path"] = self.source_storage_path

        # Include custom extra metadata if present
        for k, v in self.extra.items():
            if k not in payload:
                payload[k] = v
        return payload

    def to_relational_record(self) -> dict[str, Any]:
        """Serialize into a dictionary structured for PostgreSQL storage."""
        return {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_version_id": self.document_version_id,
            "index": self.chunk_index,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "content_type": self.content_type,
            "heading": self.heading,
            "section_path": list(self.section_path),
            "source_element_ids": list(self.source_element_ids),
            "metadata": self.to_dict(),
        }


@dataclass(frozen=True)
class EnrichedChunk(DocumentChunk):
    """Represents an enriched document chunk with structured metadata.

    Subclasses DocumentChunk to maintain 100% compatibility with downstream stages
    while attaching strongly-typed ChunkMetadata. Verbatim preserves original chunk content.
    """

    enriched_metadata: ChunkMetadata | None = None

    def to_vector_payload(self) -> dict[str, Any]:
        """Extract vector storage payload for Qdrant."""
        if self.enriched_metadata is not None:
            return self.enriched_metadata.to_vector_payload()
        return {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_version_id": self.document_version_id,
            "index": self.index,
            "section_path": list(self.section_path),
            "heading": self.heading,
        }

    def to_relational_record(self) -> dict[str, Any]:
        """Extract record dictionary for PostgreSQL."""
        if self.enriched_metadata is not None:
            return self.enriched_metadata.to_relational_record()
        return {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_version_id": self.document_version_id,
            "index": self.index,
            "heading": self.heading,
            "section_path": list(self.section_path),
            "source_element_ids": list(self.source_element_ids),
            "metadata": self.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize enriched chunk to a dictionary."""
        base = super().to_dict()
        if self.enriched_metadata is not None:
            base["enriched_metadata"] = self.enriched_metadata.to_dict()
        return base

    def __repr__(self) -> str:
        """Safe representation omitting raw chunk text."""
        content_type_str = (
            self.enriched_metadata.content_type
            if self.enriched_metadata
            else "unknown"
        )
        return (
            f"EnrichedChunk(chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"index={self.index}, "
            f"content_length={len(self.content)}, "
            f"content_type={content_type_str!r}, "
            f"section_path={self.section_path!r}, "
            f"heading={self.heading!r})"
        )


@dataclass(frozen=True)
class MetadataEnrichmentReport:
    """Summary report detailing the results and execution metrics of metadata enrichment."""

    total_chunks: int = 0
    enriched_count: int = 0
    rules_executed: list[str] = field(default_factory=list)
    content_type_counts: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata enrichment report to a dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "enriched_count": self.enriched_count,
            "rules_executed": list(self.rules_executed),
            "content_type_counts": dict(self.content_type_counts),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def __repr__(self) -> str:
        """Safe representation omitting sensitive contents."""
        return (
            f"MetadataEnrichmentReport(total_chunks={self.total_chunks}, "
            f"enriched={self.enriched_count}, "
            f"rules={self.rules_executed!r}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms)"
        )


@dataclass(frozen=True)
class EnrichedDocument(ChunkedDocument):
    """Container representing the output of the metadata enrichment stage.

    Subclasses ChunkedDocument to maintain complete structural compatibility with
    subsequent stages, holding EnrichedChunks and the MetadataEnrichmentReport.
    """

    enrichment_report: MetadataEnrichmentReport = field(
        default_factory=MetadataEnrichmentReport
    )

    def to_dict(
        self,
        include_chunks: bool = True,
        include_reports: bool = True,
    ) -> dict[str, Any]:
        """Serialize enriched document metadata, chunks, and reports to a dictionary."""
        data = super().to_dict(
            include_chunks=include_chunks,
            include_reports=include_reports,
        )
        if include_reports:
            data["enrichment_report"] = self.enrichment_report.to_dict()
        return data

    def __repr__(self) -> str:
        """Safe string representation omitting raw document contents."""
        return (
            f"EnrichedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_chunks={self.total_chunks}, "
            f"enriched_chunks={self.enrichment_report.enriched_count}, "
            f"document_version_id={self.document_version_id!r})"
        )
