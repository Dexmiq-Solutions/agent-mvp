"""Parsing and extraction package for the RAG indexing pipeline.

Responsible for format-specific parsing of ingested documents and extracting
meaningful content and hierarchical structure (headings, paragraphs, sections, tables)
while preserving document sequence and parent-child relationships.
"""

from exceptions.parsing import (
    DocumentExtractionError,
    InvalidParsingInputError,
    ParsingError,
    UnsupportedDocumentTypeError,
)
from rag.parsing.base import BaseParser
from rag.parsing.docx import DOCXParser
from rag.parsing.markdown import MarkdownParser
from rag.parsing.models import ElementType, ParsedDocument, ParsedElement
from rag.parsing.registry import ParserRegistry, get_parser_registry, reset_parser_registry
from rag.parsing.service import (
    DocumentParsingService,
    get_parsing_service,
    reset_parsing_service,
)
from rag.parsing.txt import TXTParser

__all__ = [
    # Interfaces and Parsers
    "BaseParser",
    "TXTParser",
    "MarkdownParser",
    "DOCXParser",
    # Registry & Services
    "ParserRegistry",
    "get_parser_registry",
    "reset_parser_registry",
    "DocumentParsingService",
    "get_parsing_service",
    "reset_parsing_service",
    # Domain Models
    "ElementType",
    "ParsedElement",
    "ParsedDocument",
    # Exceptions
    "ParsingError",
    "UnsupportedDocumentTypeError",
    "DocumentExtractionError",
    "InvalidParsingInputError",
]
