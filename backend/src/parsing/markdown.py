"""Structure-aware Markdown document parser."""

import re
from typing import Any

from acquisition.formats import DocumentType
from exceptions.parsing import DocumentExtractionError, InvalidParsingInputError
from ingestion.models import IngestedDocument
from parsing.base import BaseParser
from parsing.models import ElementType, ParsedDocument, ParsedElement

HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?$")
CODE_FENCE_REGEX = re.compile(r"^(`{3,}|~{3,})(.*)$")


class MarkdownParser(BaseParser):
    """Structure-aware parser for Markdown documents (.md, .markdown).

    Extracts headings (H1-H6), paragraphs, code blocks, and tables while preserving:
    - Native document order
    - Heading levels (1 to 6)
    - Nested section hierarchies (parent-child relationships)
    - Full section breadcrumb paths (section_path)
    """

    @property
    def supported_document_type(self) -> DocumentType:
        """Return DocumentType.MARKDOWN."""
        return DocumentType.MARKDOWN

    def _decode_bytes(self, raw_bytes: bytes) -> str:
        """Safely decode raw bytes with standard fallback encodings."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    def parse(self, document: IngestedDocument) -> ParsedDocument:
        """Parse an IngestedDocument containing Markdown.

        Performs a single-pass scan to identify headings, code blocks, and
        paragraphs while maintaining a heading hierarchy stack.

        Args:
            document: IngestedDocument with Markdown bytes.

        Returns:
            ParsedDocument containing ordered, hierarchically-linked elements.

        Raises:
            InvalidParsingInputError: If document is None or missing raw_bytes.
            DocumentExtractionError: If extraction fails unexpectedly.
        """
        if document is None or document.raw_bytes is None:
            raise InvalidParsingInputError("Document and raw_bytes must not be None.")

        try:
            text = self._decode_bytes(document.raw_bytes)
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            lines = normalized.split("\n")

            elements: list[ParsedElement] = []
            # Stack stores tuples of: (heading_level, element_id, heading_title)
            heading_stack: list[tuple[int, str, str]] = []

            in_code_fence = False
            fence_char = ""
            code_buffer: list[str] = []
            code_language = ""

            paragraph_buffer: list[str] = []

            def flush_paragraph() -> None:
                nonlocal paragraph_buffer
                if not paragraph_buffer:
                    return
                content = "\n".join(paragraph_buffer).strip()
                paragraph_buffer = []
                if not content:
                    return

                parent_id = heading_stack[-1][1] if heading_stack else None
                section_path = tuple(item[2] for item in heading_stack)
                order = len(elements)

                elem_type = ElementType.PARAGRAPH
                # If content consists of markdown table lines, mark as TABLE
                lines_in_block = [line.strip() for line in content.split("\n") if line.strip()]
                if len(lines_in_block) >= 2 and all(l.startswith("|") and l.endswith("|") for l in lines_in_block):
                    elem_type = ElementType.TABLE

                elements.append(
                    ParsedElement(
                        element_id=f"elem-{order}",
                        element_type=elem_type,
                        content=content,
                        order=order,
                        heading_level=None,
                        parent_id=parent_id,
                        section_path=section_path,
                        metadata={"character_count": len(content)},
                    )
                )

            for line in lines:
                stripped_line = line.strip()

                # Check for fenced code block toggle
                fence_match = CODE_FENCE_REGEX.match(stripped_line)
                if fence_match:
                    match_fence = fence_match.group(1)
                    if not in_code_fence:
                        # Opening fence
                        flush_paragraph()
                        in_code_fence = True
                        fence_char = match_fence[0]
                        code_language = fence_match.group(2).strip()
                        code_buffer = []
                        continue
                    elif in_code_fence and match_fence.startswith(fence_char):
                        # Closing fence
                        in_code_fence = False
                        code_content = "\n".join(code_buffer)
                        parent_id = heading_stack[-1][1] if heading_stack else None
                        section_path = tuple(item[2] for item in heading_stack)
                        order = len(elements)

                        elements.append(
                            ParsedElement(
                                element_id=f"elem-{order}",
                                element_type=ElementType.CODE_BLOCK,
                                content=code_content,
                                order=order,
                                heading_level=None,
                                parent_id=parent_id,
                                section_path=section_path,
                                metadata={
                                    "language": code_language,
                                    "character_count": len(code_content),
                                },
                            )
                        )
                        code_buffer = []
                        code_language = ""
                        continue

                if in_code_fence:
                    code_buffer.append(line)
                    continue

                # Check for ATX heading
                heading_match = HEADING_REGEX.match(stripped_line)
                if heading_match:
                    flush_paragraph()
                    hashes, title = heading_match.groups()
                    level = len(hashes)
                    clean_title = title.strip()

                    # Pop stack until top is strictly higher level (smaller number)
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()

                    parent_id = heading_stack[-1][1] if heading_stack else None
                    section_path = tuple(item[2] for item in heading_stack) + (clean_title,)
                    order = len(elements)
                    element_id = f"elem-{order}"

                    element = ParsedElement(
                        element_id=element_id,
                        element_type=ElementType.HEADING,
                        content=clean_title,
                        order=order,
                        heading_level=level,
                        parent_id=parent_id,
                        section_path=section_path,
                        metadata={
                            "level": level,
                            "character_count": len(clean_title),
                        },
                    )
                    elements.append(element)
                    heading_stack.append((level, element_id, clean_title))
                    continue

                # Blank line indicates paragraph boundary
                if not stripped_line:
                    flush_paragraph()
                else:
                    paragraph_buffer.append(line)

            # Flush any remaining paragraph or unclosed code fence
            flush_paragraph()
            if in_code_fence and code_buffer:
                code_content = "\n".join(code_buffer)
                parent_id = heading_stack[-1][1] if heading_stack else None
                section_path = tuple(item[2] for item in heading_stack)
                order = len(elements)
                elements.append(
                    ParsedElement(
                        element_id=f"elem-{order}",
                        element_type=ElementType.CODE_BLOCK,
                        content=code_content,
                        order=order,
                        heading_level=None,
                        parent_id=parent_id,
                        section_path=section_path,
                        metadata={
                            "language": code_language,
                            "character_count": len(code_content),
                        },
                    )
                )

            parser_metadata: dict[str, Any] = {
                "parser_class": self.__class__.__name__,
                "total_elements": len(elements),
                "headings_count": len([e for e in elements if e.element_type == ElementType.HEADING]),
            }

            return ParsedDocument(
                document_id=document.document_id,
                project_id=document.project_id,
                document_type=DocumentType.MARKDOWN,
                elements=elements,
                source_metadata=dict(document.source_metadata),
                parser_metadata=parser_metadata,
                document_version_id=document.document_version_id,
            )
        except Exception as exc:
            if isinstance(exc, InvalidParsingInputError):
                raise
            raise DocumentExtractionError(
                f"Failed to parse Markdown document '{document.document_id}': {str(exc)}",
                original_error=exc,
            ) from exc
