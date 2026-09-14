"""Plain text (TXT) document parser."""

import re
from typing import Any

from rag.acquisition.formats import DocumentType
from exceptions.parsing import DocumentExtractionError, InvalidParsingInputError
from rag.ingestion.models import IngestedDocument
from rag.parsing.base import BaseParser
from rag.parsing.models import ElementType, ParsedDocument, ParsedElement


class TXTParser(BaseParser):
    """Parser for plain text documents (.txt).

    Extracts ordered text blocks or paragraphs while preserving document sequence.
    Does not guess or invent structural headings for plain text files.
    """

    @property
    def supported_document_type(self) -> DocumentType:
        """Return DocumentType.TXT."""
        return DocumentType.TXT

    def _decode_bytes(self, raw_bytes: bytes) -> str:
        """Safely decode raw bytes with standard fallback encodings.

        Tries utf-8 first, followed by utf-8-sig and latin-1, before
        falling back to replacement characters to prevent decoding crashes.
        """
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    def parse(self, document: IngestedDocument) -> ParsedDocument:
        """Parse an IngestedDocument containing plain text.

        Identifies non-empty text blocks separated by blank lines, maintaining
        exact document order without speculative heading detection.

        Args:
            document: IngestedDocument with raw text bytes.

        Returns:
            ParsedDocument containing ordered text blocks.

        Raises:
            InvalidParsingInputError: If document is None or missing raw_bytes.
            DocumentExtractionError: If extraction fails unexpectedly.
        """
        if document is None or document.raw_bytes is None:
            raise InvalidParsingInputError("Document and raw_bytes must not be None.")

        try:
            text = self._decode_bytes(document.raw_bytes)
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")

            # Split on two or more consecutive newlines (possibly with whitespace)
            raw_blocks = re.split(r"\n\s*\n", normalized)

            elements: list[ParsedElement] = []
            for block in raw_blocks:
                content = block.strip()
                if not content:
                    continue

                order = len(elements)
                element = ParsedElement(
                    element_id=f"elem-{order}",
                    element_type=ElementType.TEXT_BLOCK,
                    content=content,
                    order=order,
                    heading_level=None,
                    parent_id=None,
                    section_path=(),
                    metadata={"character_count": len(content)},
                )
                elements.append(element)

            parser_metadata: dict[str, Any] = {
                "parser_class": self.__class__.__name__,
                "total_blocks": len(elements),
                "encoding_handled": True,
            }

            return ParsedDocument(
                document_id=document.document_id,
                project_id=document.project_id,
                document_type=DocumentType.TXT,
                elements=elements,
                source_metadata=dict(document.source_metadata),
                parser_metadata=parser_metadata,
                document_version_id=document.document_version_id,
            )
        except Exception as exc:
            if isinstance(exc, InvalidParsingInputError):
                raise
            raise DocumentExtractionError(
                f"Failed to parse TXT document '{document.document_id}': {str(exc)}",
                original_error=exc,
            ) from exc
