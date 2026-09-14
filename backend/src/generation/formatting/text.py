"""TextContextFormatter implementing the structured, metadata-aware textual representation."""

import time
from typing import Any, Optional, Sequence

from exceptions.generation import ContextFormattingValidationError
from generation.formatting.base import BaseContextFormatter
from generation.formatting.config import ContextFormattingConfig
from generation.formatting.models import FormattedContext, FormattedContextItem


class TextContextFormatter(BaseContextFormatter):
    """Structured, metadata-aware textual context formatter.

    Formats the authoritative Final Context into a clean, model-readable string
    with explicit context delimiters, model-facing metadata, and verbatim evidence
    preservation.
    """

    def __init__(self, config: Optional[ContextFormattingConfig] = None) -> None:
        """Initialize TextContextFormatter.

        Args:
            config: Optional ContextFormattingConfig override.
        """
        super().__init__(config=config)

    def _extract_item_content(self, item: Any, index: int) -> tuple[str, str]:
        """Extract resolved text and verbatim content from a context item.

        Args:
            item: Context item candidate.
            index: 1-based display index for error messages.

        Returns:
            tuple[str, str]: (resolved_text, verbatim_content).

        Raises:
            ContextFormattingValidationError: If item is None or lacks textual content.
        """
        if item is None:
            raise ContextFormattingValidationError(f"Context item at index {index} cannot be None.")

        text = (
            getattr(item, "text", None)
            or getattr(item, "content", None)
            or getattr(item, "chunk_text", None)
        )
        content = (
            getattr(item, "content", None)
            or getattr(item, "chunk_text", None)
            or getattr(item, "text", None)
        )

        if text is None and content is None:
            raise ContextFormattingValidationError(
                f"Malformed context item at index {index}: missing both 'text' and 'content'."
            )

        return str(text or ""), str(content or "")

    def format_item(self, item: Any, index: int) -> FormattedContextItem:
        """Format an individual context item into a FormattedContextItem.

        Preserves verbatim content, source identity, and hierarchy without
        exposing internal retrieval plumbing.

        Args:
            item: AssembledContextItem or duck-typed context item.
            index: 1-based display index.

        Returns:
            FormattedContextItem: Formatted individual item.
        """
        resolved_text, verbatim_content = self._extract_item_content(item, index)

        # Extract coordinates
        chunk_id = getattr(item, "chunk_id", None)
        document_id = getattr(item, "document_id", None)
        doc_version = getattr(item, "document_version_id", None)

        # Extract metadata
        raw_metadata = getattr(item, "metadata", None)
        item_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        # Extract provenance
        source = getattr(item, "source", None) or item_metadata.get("source") or item_metadata.get("file_name")
        heading = getattr(item, "heading", None)
        section_path = getattr(item, "section_path", ()) or ()
        if isinstance(section_path, (list, set)):
            section_path = tuple(section_path)

        title = (
            item_metadata.get("title")
            or item_metadata.get("document_title")
            or item_metadata.get("document_name")
        )

        # Build lines
        item_label = self._config.item_label_template.format(index=index)
        lines: list[str] = [item_label]

        # 1. Document identification
        if title:
            lines.append(f"Document: {title}")
            if self._config.include_metadata and document_id and str(document_id) != str(title):
                lines.append(f"Document ID: {document_id}")
        elif document_id:
            lines.append(f"Document: {document_id}")

        if self._config.include_metadata:
            if doc_version:
                lines.append(f"Document Version: {doc_version}")
            if chunk_id:
                lines.append(f"Chunk ID: {chunk_id}")

        # 2. Structural hierarchy & provenance
        if self._config.include_provenance:
            if source and str(source) != str(title) and str(source) != str(document_id):
                lines.append(f"Source: {source}")

            page = item_metadata.get("page") or item_metadata.get("page_number")
            if page is not None and str(page).strip():
                lines.append(f"Page: {page}")

            if section_path:
                if isinstance(section_path, (list, tuple)):
                    path_parts = [str(p).strip() for p in section_path if str(p).strip()]
                    if path_parts:
                        lines.append(f"Section: {' > '.join(path_parts)}")
                else:
                    lines.append(f"Section: {section_path}")
            elif heading and str(heading).strip():
                lines.append(f"Section: {str(heading).strip()}")

        # 3. Verbatim content (separated by an empty line)
        lines.append("")
        lines.append(resolved_text)

        formatted_text = "\n".join(lines)

        return FormattedContextItem(
            index=index,
            text=formatted_text,
            content=verbatim_content,
            chunk_id=str(chunk_id) if chunk_id else None,
            document_id=str(document_id) if document_id else None,
            document_version_id=str(doc_version) if doc_version else None,
            source=str(source) if source else None,
            heading=str(heading) if heading else None,
            section_path=section_path,
            metadata=item_metadata,
        )

    def format(
        self,
        items: Sequence[Any],
        project_id: Optional[str] = None,
    ) -> FormattedContext:
        """Format an ordered sequence of authoritative context items into a FormattedContext.

        Args:
            items: Ordered sequence of context items (e.g. AssembledContextItem).
            project_id: Optional tenant boundary identifier.

        Returns:
            FormattedContext: Model-readable formatted context representation.

        Raises:
            ContextFormattingValidationError: If context items are malformed or empty
                when allow_empty_context is False.
        """
        start_time = time.perf_counter()

        if items is None:
            raise ContextFormattingValidationError("Context items sequence cannot be None.")

        # Handle empty context
        if len(items) == 0:
            if not self._config.allow_empty_context:
                raise ContextFormattingValidationError(
                    "Retrieved context contains zero items and allow_empty_context is False."
                )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return FormattedContext(
                text=self._config.empty_context_text,
                items=(),
                item_count=0,
                project_id=project_id,
                strategy="text",
                metadata={
                    "elapsed_ms": round(elapsed_ms, 2),
                    "item_count": 0,
                    "characters": len(self._config.empty_context_text),
                },
            )

        # Single-pass format preserving exact upstream order
        formatted_items: list[FormattedContextItem] = []
        for idx, item in enumerate(items, start=1):
            formatted_item = self.format_item(item=item, index=idx)
            formatted_items.append(formatted_item)

        # Delimited context assembly
        blocks: list[str] = []
        if self._config.include_header and self._config.context_header:
            blocks.append(self._config.context_header)

        blocks.append("\n\n".join(item.text for item in formatted_items))

        if self._config.include_footer and self._config.context_footer:
            blocks.append(self._config.context_footer)

        full_text = "\n\n".join(blocks)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FormattedContext(
            text=full_text,
            items=tuple(formatted_items),
            item_count=len(formatted_items),
            project_id=project_id,
            strategy="text",
            metadata={
                "elapsed_ms": round(elapsed_ms, 2),
                "item_count": len(formatted_items),
                "characters": len(full_text),
            },
        )
