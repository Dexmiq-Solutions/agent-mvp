"""Deterministic structure-aware context provider."""

from rag.contextual_enrichment.base import BaseContextProvider, DocumentContext
from rag.metadata_enrichment.models import EnrichedChunk


class StructuredContextProvider(BaseContextProvider):
    """Deterministic context provider utilizing document structure and metadata hierarchy.

    Synthesizes concise, high-signal context from section breadcrumbs, heading hierarchy,
    and document provenance without invoking external models or inventing facts.
    """

    def __init__(self, prefix_document_name: bool = True) -> None:
        self._prefix_document_name = prefix_document_name

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "structured"

    async def generate_context(
        self,
        chunk: EnrichedChunk,
        context: DocumentContext,
    ) -> str | None:
        """Generate deterministic context from document and section hierarchy.

        Returns None if no meaningful structural context is available.
        """
        parts: list[str] = []

        # 1. Resolve document title or filename
        doc_name: str | None = None
        if self._prefix_document_name:
            doc_name = context.original_filename
            if not doc_name and context.source_metadata:
                doc_name = (
                    context.source_metadata.get("title")
                    or context.source_metadata.get("original_filename")
                )

        # 2. Resolve section hierarchy / heading path
        heading_path: str | None = None
        if chunk.enriched_metadata and chunk.enriched_metadata.heading_path_str:
            heading_path = chunk.enriched_metadata.heading_path_str
        elif chunk.section_path:
            heading_path = " > ".join(chunk.section_path)
        elif chunk.heading:
            heading_path = chunk.heading

        # 3. Assemble context without inventing nonexistent attributes
        if doc_name and heading_path:
            parts.append(f"Document: {doc_name}")
            parts.append(f"Section: {heading_path}")
        elif heading_path:
            parts.append(f"Section: {heading_path}")
        elif doc_name and context.total_chunks > 1:
            parts.append(f"Document: {doc_name}")
        else:
            # Minimal/unstructured chunk with no headings or multi-chunk document context
            return None

        return " | ".join(parts)
