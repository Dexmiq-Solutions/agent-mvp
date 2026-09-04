"""Document parsing service orchestrating format-specific extraction."""

import asyncio
import time
from typing import Optional

from app.core.logging import get_logger
from exceptions.parsing import (
    DocumentExtractionError,
    InvalidParsingInputError,
    ParsingError,
    UnsupportedDocumentTypeError,
)
from ingestion.models import IngestedDocument
from parsing.models import ParsedDocument
from parsing.registry import ParserRegistry, get_parser_registry

logger = get_logger(__name__)


class DocumentParsingService:
    """Service layer coordinating document parsing and structural extraction.

    Consumes standardized IngestedDocuments, resolves the appropriate parser
    from ParserRegistry, executes synchronous extraction with thread offloading
    for non-blocking async execution, and logs operational metrics securely.
    """

    def __init__(self, registry: Optional[ParserRegistry] = None) -> None:
        """Initialize parsing service with an optional custom parser registry.

        Args:
            registry: Optional ParserRegistry override. If None, uses default singleton.
        """
        self._registry = registry or get_parser_registry()

    def parse_sync(self, document: IngestedDocument) -> ParsedDocument:
        """Parse an IngestedDocument synchronously.

        Executes the CPU-bound parsing operation directly on the current thread.

        Args:
            document: Standardized IngestedDocument from the ingestion stage.

        Returns:
            Structured ParsedDocument containing ordered elements and hierarchy.

        Raises:
            InvalidParsingInputError: If document is None or invalid.
            UnsupportedDocumentTypeError: If document type is unsupported.
            DocumentExtractionError: If parsing or extraction fails.
        """
        if not isinstance(document, IngestedDocument):
            raise InvalidParsingInputError(
                f"Expected IngestedDocument, got '{type(document).__name__}'."
            )

        parser = self._registry.resolve_parser(document)

        logger.info(
            "Starting parsing for document '%s' (project: '%s', type: '%s', parser: '%s')",
            document.document_id,
            document.project_id,
            document.detected_document_type.value,
            parser.__class__.__name__,
        )

        start_time = time.perf_counter()
        try:
            parsed_doc = parser.parse(document)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Successfully parsed document '%s' in %.2fms (total_elements: %d, headings: %d)",
                document.document_id,
                elapsed_ms,
                parsed_doc.total_elements,
                len(parsed_doc.headings),
            )
            return parsed_doc

        except (UnsupportedDocumentTypeError, InvalidParsingInputError):
            raise
        except DocumentExtractionError as exc:
            logger.error(
                "Document extraction error parsing '%s' (type: '%s'): %s",
                document.document_id,
                document.detected_document_type.value,
                exc.message,
            )
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure parsing document '%s' (type: '%s'): %s",
                document.document_id,
                document.detected_document_type.value,
                str(exc),
            )
            raise DocumentExtractionError(
                f"Unexpected failure parsing document '{document.document_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    async def parse(self, document: IngestedDocument) -> ParsedDocument:
        """Parse an IngestedDocument asynchronously via controlled thread offloading.

        Ensures CPU-bound document parsing operations do not block the event loop.

        Args:
            document: Standardized IngestedDocument from the ingestion stage.

        Returns:
            Structured ParsedDocument.
        """
        return await asyncio.to_thread(self.parse_sync, document)

    async def parse_batch(self, documents: list[IngestedDocument]) -> list[ParsedDocument]:
        """Parse multiple IngestedDocuments concurrently via thread offloading.

        Args:
            documents: List of IngestedDocuments to parse.

        Returns:
            List of ParsedDocument results preserving input order.
        """
        tasks = [self.parse(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_parsing_service: Optional[DocumentParsingService] = None


def get_parsing_service(registry: Optional[ParserRegistry] = None) -> DocumentParsingService:
    """Get or create the default DocumentParsingService instance.

    Args:
        registry: Optional ParserRegistry override.
    """
    global _default_parsing_service
    if registry is not None:
        return DocumentParsingService(registry=registry)
    if _default_parsing_service is None:
        _default_parsing_service = DocumentParsingService()
    return _default_parsing_service


def reset_parsing_service() -> None:
    """Reset the singleton parsing service instance. Useful for test isolation."""
    global _default_parsing_service
    _default_parsing_service = None
