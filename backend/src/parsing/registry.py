"""Centralized parser registry and resolver for document parsing."""

from typing import Optional, Union

from acquisition.formats import DocumentType, get_document_type
from exceptions.parsing import UnsupportedDocumentTypeError
from ingestion.models import IngestedDocument
from parsing.base import BaseParser
from parsing.docx import DOCXParser
from parsing.markdown import MarkdownParser
from parsing.txt import TXTParser


class ParserRegistry:
    """Central registry and single source of truth for document parsers.

    Maps supported DocumentTypes to their corresponding BaseParser implementations.
    Provides extensible registration so future formats (e.g. PDF) can be plugged in
    without altering existing parsing workflows.
    """

    def __init__(self) -> None:
        """Initialize registry with default supported parsers."""
        self._parsers: dict[DocumentType, BaseParser] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default parsers for TXT, Markdown, and DOCX."""
        self.register_parser(DocumentType.TXT, TXTParser())
        self.register_parser(DocumentType.MARKDOWN, MarkdownParser())
        self.register_parser(DocumentType.DOCX, DOCXParser())

    def register_parser(self, document_type: DocumentType, parser: BaseParser) -> None:
        """Register a parser implementation for a DocumentType.

        Args:
            document_type: Target DocumentType.
            parser: BaseParser implementation instance.
        """
        self._parsers[document_type] = parser

    def is_supported(self, doc_type_or_extension: Union[DocumentType, str]) -> bool:
        """Check if a DocumentType or file extension has a registered parser.

        Args:
            doc_type_or_extension: DocumentType enum or filename/extension string.

        Returns:
            True if supported, False otherwise.
        """
        if isinstance(doc_type_or_extension, DocumentType):
            return doc_type_or_extension in self._parsers
        resolved = get_document_type(doc_type_or_extension)
        return resolved in self._parsers if resolved else False

    def get_parser(self, document_type: DocumentType) -> BaseParser:
        """Retrieve the registered parser for a given DocumentType.

        Args:
            document_type: The DocumentType to look up.

        Returns:
            The registered BaseParser instance.

        Raises:
            UnsupportedDocumentTypeError: If no parser is registered for the document type.
        """
        parser = self._parsers.get(document_type)
        if parser is None:
            type_str = document_type.value if isinstance(document_type, DocumentType) else str(document_type)
            raise UnsupportedDocumentTypeError(
                f"No parser registered for document type '{type_str}'. "
                f"Supported types: {', '.join(t.value for t in self._parsers.keys())}"
            )
        return parser

    def resolve_parser(self, source: Union[IngestedDocument, DocumentType, str]) -> BaseParser:
        """Resolve the appropriate parser from an IngestedDocument, DocumentType, or path/extension.

        Args:
            source: IngestedDocument, DocumentType enum, or path/filename string.

        Returns:
            The resolved BaseParser instance.

        Raises:
            UnsupportedDocumentTypeError: If source type cannot be resolved or is unsupported.
        """
        if isinstance(source, IngestedDocument):
            return self.get_parser(source.detected_document_type)

        if isinstance(source, DocumentType):
            return self.get_parser(source)

        if isinstance(source, str):
            doc_type = get_document_type(source)
            if doc_type is None:
                raise UnsupportedDocumentTypeError(
                    f"Unsupported or unrecognized document format for '{source}'."
                )
            return self.get_parser(doc_type)

        raise UnsupportedDocumentTypeError(
            f"Cannot resolve parser from input of type '{type(source).__name__}'."
        )

    @property
    def supported_document_types(self) -> tuple[DocumentType, ...]:
        """Return tuple of all currently registered DocumentTypes."""
        return tuple(self._parsers.keys())


_default_parser_registry: Optional[ParserRegistry] = None


def get_parser_registry() -> ParserRegistry:
    """Get the singleton ParserRegistry instance."""
    global _default_parser_registry
    if _default_parser_registry is None:
        _default_parser_registry = ParserRegistry()
    return _default_parser_registry


def reset_parser_registry() -> None:
    """Reset the singleton ParserRegistry. Useful for test isolation."""
    global _default_parser_registry
    _default_parser_registry = None
