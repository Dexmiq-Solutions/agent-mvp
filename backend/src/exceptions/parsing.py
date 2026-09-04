"""Parsing-specific domain exceptions."""


class ParsingError(Exception):
    """Base exception for all document parsing errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class UnsupportedDocumentTypeError(ParsingError):
    """Raised when an unsupported document type is supplied for parsing."""


class DocumentExtractionError(ParsingError):
    """Raised when extracting content and structure from a document fails."""


class InvalidParsingInputError(ParsingError):
    """Raised when parsing input document or payload is invalid or empty."""
