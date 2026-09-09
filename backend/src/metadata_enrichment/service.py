"""Document metadata enrichment service orchestrating structure-aware hybrid enrichment."""

import asyncio
import time
from typing import Any

from app.core.logging import get_logger
from chunking.models import ChunkedDocument, DocumentChunk
from exceptions.metadata_enrichment import (
    InvalidMetadataEnrichmentInputError,
    MetadataEnrichmentProcessingError,
)
from metadata_enrichment.base import BaseMetadataEnricher, DocumentEnrichmentContext
from metadata_enrichment.models import (
    ChunkMetadata,
    EnrichedChunk,
    EnrichedDocument,
    MetadataEnrichmentConfig,
    MetadataEnrichmentReport,
)
from metadata_enrichment.rules import (
    CharacteristicsEnricherRule,
    ProvenanceEnricherRule,
    StructuralEnricherRule,
)

logger = get_logger(__name__)


class DocumentMetadataEnrichmentService:
    """Service orchestrating structure-aware hybrid metadata enrichment.

    Consumes a ChunkedDocument, preserves all inherited provenance and structure,
    derives deterministic chunk characteristics, enforces tenant isolation,
    and returns an EnrichedDocument with strongly-typed ChunkMetadata.
    """

    def __init__(self, config: MetadataEnrichmentConfig | None = None) -> None:
        self._config = config or MetadataEnrichmentConfig()
        self._enrichers: list[BaseMetadataEnricher] = []

        if self._config.enable_provenance:
            self._enrichers.append(ProvenanceEnricherRule())
        if self._config.enable_structural:
            self._enrichers.append(
                StructuralEnricherRule(
                    heading_separator=self._config.heading_separator
                )
            )
        if self._config.enable_characteristics:
            self._enrichers.append(CharacteristicsEnricherRule())

        # Support lightweight extension for custom/future enrichers
        if self._config.custom_enrichers:
            self._enrichers.extend(self._config.custom_enrichers)

    @property
    def config(self) -> MetadataEnrichmentConfig:
        """Return the active configuration."""
        return self._config

    def enrich_sync(self, document: ChunkedDocument) -> EnrichedDocument:
        """Enrich a ChunkedDocument synchronously.

        Enforces strict input validation, multi-tenant isolation, preserves
        original text verbatim, and derives structured metadata in O(N) time.

        Args:
            document: ChunkedDocument from the Chunking stage.

        Returns:
            EnrichedDocument containing EnrichedChunks and MetadataEnrichmentReport.

        Raises:
            InvalidMetadataEnrichmentInputError: If document is not a ChunkedDocument.
            MetadataEnrichmentProcessingError: If tenant isolation fails or processing errors.
        """
        if not isinstance(document, ChunkedDocument):
            raise InvalidMetadataEnrichmentInputError(
                f"Expected ChunkedDocument, got '{type(document).__name__}'."
            )

        start_time = time.perf_counter()
        doc_id = document.document_id
        project_id = document.project_id
        total_chunks = len(document.chunks)

        logger.info(
            "Starting metadata enrichment for document '%s' (project: '%s', chunks: %d)",
            doc_id,
            project_id,
            total_chunks,
        )

        try:
            # Enforce project isolation across all chunks
            for chunk in document.chunks:
                if not isinstance(chunk, DocumentChunk):
                    raise InvalidMetadataEnrichmentInputError(
                        f"Expected DocumentChunk in chunks list, got '{type(chunk).__name__}'."
                    )
                if chunk.project_id != project_id:
                    raise MetadataEnrichmentProcessingError(
                        f"Project isolation violation: Chunk '{chunk.chunk_id}' has project_id "
                        f"'{chunk.project_id}', which does not match document project_id '{project_id}'."
                    )

            # Handle empty document edge case
            if total_chunks == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                report = MetadataEnrichmentReport(
                    total_chunks=0,
                    enriched_count=0,
                    rules_executed=[e.enricher_name for e in self._enrichers],
                    content_type_counts={},
                    elapsed_ms=elapsed_ms,
                )
                return EnrichedDocument(
                    document_id=doc_id,
                    project_id=project_id,
                    document_type=document.document_type,
                    chunks=[],
                    document_version_id=document.document_version_id,
                    source_metadata=dict(document.source_metadata),
                    parser_metadata=dict(document.parser_metadata),
                    cleaning_report=document.cleaning_report,
                    normalization_report=document.normalization_report,
                    chunking_report=document.chunking_report,
                    enrichment_report=report,
                )

            # Construct O(1) document context for all chunks
            context = DocumentEnrichmentContext(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                total_chunks=total_chunks,
                document_version_id=document.document_version_id,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
            )

            enriched_chunks: list[EnrichedChunk] = []
            content_type_counts: dict[str, int] = {}
            rules_executed = [e.enricher_name for e in self._enrichers]

            for chunk in document.chunks:
                merged_data: dict[str, Any] = {}
                extra_data: dict[str, Any] = {}

                # Execute enrichment rules in sequence
                for enricher in self._enrichers:
                    rule_res = enricher.enrich(chunk, context)
                    if rule_res:
                        merged_data.update(rule_res)
                        if enricher.enricher_name not in {
                            "provenance_enricher",
                            "structural_enricher",
                            "characteristics_enricher",
                        }:
                            extra_data.update(rule_res)

                # Construct strongly-typed ChunkMetadata
                chunk_meta = ChunkMetadata(
                    project_id=merged_data.get("project_id", chunk.project_id),
                    document_id=merged_data.get("document_id", chunk.document_id),
                    chunk_id=merged_data.get("chunk_id", chunk.chunk_id),
                    document_version_id=merged_data.get(
                        "document_version_id", chunk.document_version_id
                    ),
                    source_element_ids=merged_data.get(
                        "source_element_ids", chunk.source_element_ids
                    ),
                    document_type=merged_data.get(
                        "document_type", document.document_type.value
                    ),
                    source_storage_path=merged_data.get("source_storage_path"),
                    original_filename=merged_data.get("original_filename"),
                    section_path=merged_data.get("section_path", chunk.section_path),
                    heading=merged_data.get("heading", chunk.heading),
                    heading_level=merged_data.get("heading_level", chunk.heading_level),
                    heading_path_str=merged_data.get("heading_path_str", ""),
                    hierarchy_depth=merged_data.get("hierarchy_depth", 0),
                    parent_element_id=merged_data.get(
                        "parent_element_id", chunk.parent_element_id
                    ),
                    parent_chunk_id=merged_data.get(
                        "parent_chunk_id", chunk.parent_chunk_id
                    ),
                    chunk_index=merged_data.get("chunk_index", chunk.index),
                    total_chunks=merged_data.get("total_chunks", total_chunks),
                    relative_position=merged_data.get("relative_position", 0.0),
                    character_count=merged_data.get(
                        "character_count", len(chunk.content)
                    ),
                    word_count=merged_data.get("word_count", len(chunk.content.split())),
                    line_count=merged_data.get("line_count", 0),
                    content_type=merged_data.get("content_type", "prose"),
                    has_code=merged_data.get("has_code", False),
                    has_table=merged_data.get("has_table", False),
                    has_list=merged_data.get("has_list", False),
                    is_header_chunk=merged_data.get("is_header_chunk", False),
                    language=merged_data.get("language"),
                    source_metadata=dict(document.source_metadata),
                    extra=extra_data,
                )

                # Track metrics
                ctype = chunk_meta.content_type
                content_type_counts[ctype] = content_type_counts.get(ctype, 0) + 1

                # Combine existing chunk metadata with flat derived metadata
                combined_metadata = dict(chunk.metadata)
                combined_metadata.update(chunk_meta.to_flat_dict())

                # Create EnrichedChunk strictly preserving verbatim chunk content
                enriched_chunk = EnrichedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    project_id=chunk.project_id,
                    content=chunk.content,  # Strict verbatim preservation
                    index=chunk.index,
                    document_version_id=chunk_meta.document_version_id,
                    section_path=chunk_meta.section_path,
                    heading=chunk_meta.heading,
                    heading_level=chunk_meta.heading_level,
                    parent_element_id=chunk_meta.parent_element_id,
                    parent_chunk_id=chunk_meta.parent_chunk_id,
                    source_element_ids=chunk_meta.source_element_ids,
                    element_types=chunk.element_types,
                    metadata=combined_metadata,
                    enriched_metadata=chunk_meta,
                )
                enriched_chunks.append(enriched_chunk)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = MetadataEnrichmentReport(
                total_chunks=total_chunks,
                enriched_count=len(enriched_chunks),
                rules_executed=rules_executed,
                content_type_counts=content_type_counts,
                elapsed_ms=elapsed_ms,
            )

            logger.info(
                "Successfully enriched document '%s' in %.2fms (chunks: %d, types: %s)",
                doc_id,
                elapsed_ms,
                len(enriched_chunks),
                content_type_counts,
            )

            return EnrichedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                chunks=enriched_chunks,
                document_version_id=document.document_version_id,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                cleaning_report=document.cleaning_report,
                normalization_report=document.normalization_report,
                chunking_report=document.chunking_report,
                enrichment_report=report,
            )

        except (InvalidMetadataEnrichmentInputError, MetadataEnrichmentProcessingError):
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure enriching document '%s': %s",
                doc_id,
                str(exc),
            )
            raise MetadataEnrichmentProcessingError(
                f"Unexpected failure enriching metadata for document '{doc_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    async def enrich(self, document: ChunkedDocument) -> EnrichedDocument:
        """Enrich a ChunkedDocument asynchronously.

        Offloads CPU-bound enrichment operations to a worker thread via asyncio.to_thread.
        """
        return await asyncio.to_thread(self.enrich_sync, document)

    async def enrich_batch(
        self,
        documents: list[ChunkedDocument],
    ) -> list[EnrichedDocument]:
        """Enrich multiple ChunkedDocuments concurrently while preserving isolation.

        Args:
            documents: List of ChunkedDocument instances.

        Returns:
            List of EnrichedDocument results preserving input order.
        """
        tasks = [self.enrich(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_metadata_enrichment_service: DocumentMetadataEnrichmentService | None = None


def get_metadata_enrichment_service(
    config: MetadataEnrichmentConfig | None = None,
) -> DocumentMetadataEnrichmentService:
    """Get or create singleton DocumentMetadataEnrichmentService instance."""
    global _default_metadata_enrichment_service
    if config is not None:
        return DocumentMetadataEnrichmentService(config=config)
    if _default_metadata_enrichment_service is None:
        _default_metadata_enrichment_service = DocumentMetadataEnrichmentService()
    return _default_metadata_enrichment_service


def reset_metadata_enrichment_service() -> None:
    """Reset the singleton instance. Useful for test isolation."""
    global _default_metadata_enrichment_service
    _default_metadata_enrichment_service = None
