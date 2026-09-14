"""Deterministic rules for structure-aware hybrid metadata enrichment."""

import re
from typing import Any

from rag.metadata_enrichment.base import BaseMetadataEnricher, DocumentEnrichmentContext
from rag.metadata_enrichment.models import ChunkContentType
from rag.parsing.models import ElementType
from rag.chunking.models import DocumentChunk

CODE_FENCE_LANG_REGEX = re.compile(r"```([a-zA-Z0-9_\-\.\+]+)")
MARKDOWN_TABLE_REGEX = re.compile(r"\|[\s\-:]+\|")


class ProvenanceEnricherRule(BaseMetadataEnricher):
    """Preserves and standardizes identity, document version, tenant boundary, and source references."""

    @property
    def enricher_name(self) -> str:
        return "provenance_enricher"

    def enrich(
        self,
        chunk: DocumentChunk,
        context: DocumentEnrichmentContext,
    ) -> dict[str, Any]:
        source_meta = context.source_metadata
        source_storage_path = (
            source_meta.get("source_storage_path")
            or source_meta.get("storage_path")
            or source_meta.get("path")
        )
        original_filename = (
            source_meta.get("original_filename")
            or source_meta.get("filename")
        )
        version_id = chunk.document_version_id or context.document_version_id

        return {
            "project_id": chunk.project_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "document_version_id": version_id,
            "source_element_ids": chunk.source_element_ids,
            "document_type": context.document_type.value,
            "source_storage_path": source_storage_path,
            "original_filename": original_filename,
            "source_metadata": source_meta,
        }


class StructuralEnricherRule(BaseMetadataEnricher):
    """Preserves section hierarchy, computes breadcrumb paths, depth, and relative document positions."""

    def __init__(self, heading_separator: str = " > ") -> None:
        self._heading_separator = heading_separator

    @property
    def enricher_name(self) -> str:
        return "structural_enricher"

    def enrich(
        self,
        chunk: DocumentChunk,
        context: DocumentEnrichmentContext,
    ) -> dict[str, Any]:
        section_path = chunk.section_path
        heading_path_str = (
            self._heading_separator.join(section_path) if section_path else ""
        )
        hierarchy_depth = len(section_path)
        total_chunks = max(1, context.total_chunks)
        relative_position = round(chunk.index / total_chunks, 4)

        return {
            "section_path": section_path,
            "heading": chunk.heading,
            "heading_level": chunk.heading_level,
            "heading_path_str": heading_path_str,
            "hierarchy_depth": hierarchy_depth,
            "parent_element_id": chunk.parent_element_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "chunk_index": chunk.index,
            "total_chunks": context.total_chunks,
            "relative_position": relative_position,
        }


class CharacteristicsEnricherRule(BaseMetadataEnricher):
    """Derives deterministic chunk metrics, structural content classification, and feature flags."""

    @property
    def enricher_name(self) -> str:
        return "characteristics_enricher"

    def enrich(
        self,
        chunk: DocumentChunk,
        context: DocumentEnrichmentContext,
    ) -> dict[str, Any]:
        content = chunk.content
        char_count = len(content)
        word_count = len(content.split())
        line_count = content.count("\n") + 1 if content else 0

        element_types = chunk.element_types
        has_code = ElementType.CODE_BLOCK in element_types or "```" in content
        has_table = (
            ElementType.TABLE in element_types
            or bool(MARKDOWN_TABLE_REGEX.search(content))
        )
        has_list = ElementType.LIST_ITEM in element_types

        # Extract code language if present
        language: str | None = None
        if has_code:
            lang_match = CODE_FENCE_LANG_REGEX.search(content)
            if lang_match:
                language = lang_match.group(1).lower()
            elif "language" in chunk.metadata:
                language = str(chunk.metadata["language"]).lower()

        # Classify content type
        content_type = self._classify_content_type(
            element_types=element_types,
            content=content,
            has_code=has_code,
            has_table=has_table,
            has_list=has_list,
        )

        is_header_chunk = (
            bool(element_types)
            and all(et == ElementType.HEADING for et in element_types)
        )

        return {
            "character_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "content_type": content_type,
            "has_code": has_code,
            "has_table": has_table,
            "has_list": has_list,
            "is_header_chunk": is_header_chunk,
            "language": language,
        }

    def _classify_content_type(
        self,
        element_types: tuple[ElementType, ...],
        content: str,
        has_code: bool,
        has_table: bool,
        has_list: bool,
    ) -> str:
        if element_types:
            # Check for pure or predominant types
            if all(et == ElementType.HEADING for et in element_types):
                return ChunkContentType.HEADING.value
            if all(et == ElementType.TABLE for et in element_types):
                return ChunkContentType.TABLE.value
            if all(et == ElementType.CODE_BLOCK for et in element_types):
                return ChunkContentType.CODE.value
            if all(et == ElementType.LIST_ITEM for et in element_types):
                return ChunkContentType.LIST.value

        # Content heuristics when element types are mixed or fallback
        if has_code and content.strip().startswith("```"):
            return ChunkContentType.CODE.value
        if has_table and "|" in content:
            return ChunkContentType.TABLE.value
        if has_list and re.match(r"^\s*[-*+]|\d+[.)]", content):
            return ChunkContentType.LIST.value

        return ChunkContentType.PROSE.value
