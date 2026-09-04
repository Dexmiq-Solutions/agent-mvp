"""Abstract base class for document format parsers."""

from abc import ABC, abstractmethod
from typing import Any

from acquisition.formats import DocumentType, get_canonical_mime_type
from ingestion.models import IngestedDocument
from parsing.models import ParsedDocument


class BaseParser(ABC):
    """Abstract interface defining the document parser contract.

    Parsers are responsible for understanding a specific document format,
    extracting its meaningful content, and preserving structural hierarchy,
    order, and relationships without altering, cleaning, or chunking text.
    """

    @property
    @abstractmethod
    def supported_document_type(self) -> DocumentType:
        """Return the primary DocumentType handled by this parser."""

    @abstractmethod
    def parse(self, document: IngestedDocument) -> ParsedDocument:
        """Parse an IngestedDocument and extract its structural representation.

        Args:
            document: Standardized IngestedDocument containing raw file bytes
                      and preserved source metadata.

        Returns:
            Structured ParsedDocument with ordered elements and hierarchy links.

        Raises:
            DocumentExtractionError: If extracting content or structure fails.
            InvalidParsingInputError: If document input is invalid.
        """

    def parse_content(
        self,
        content: bytes | str,
        document_id: str = "doc-0",
        project_id: str = "default",
        original_filename: str = "document",
        source_metadata: dict[str, Any] | None = None,
        document_version_id: str | None = None,
    ) -> ParsedDocument:
        """Convenience method to parse raw bytes or string directly.

        Useful for unit testing and direct invocation without full storage ingestion.

        Args:
            content: Raw file bytes or string payload.
            document_id: Optional document identifier.
            project_id: Optional project tenant identifier.
            original_filename: File name for format identification.
            source_metadata: Optional dictionary of source metadata.
            document_version_id: Optional version or ETag identifier.

        Returns:
            Structured ParsedDocument.
        """
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
        doc_type = self.supported_document_type
        canonical_mime = get_canonical_mime_type(doc_type)

        ingested = IngestedDocument(
            document_id=document_id,
            project_id=project_id,
            source_storage_path=f"{project_id}/{original_filename}",
            original_filename=original_filename,
            detected_document_type=doc_type,
            content_type=canonical_mime,
            raw_bytes=raw_bytes,
            source_metadata=source_metadata or {},
            document_version_id=document_version_id,
            size_bytes=len(raw_bytes),
        )
        return self.parse(ingested)
