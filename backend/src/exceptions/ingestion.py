"""Ingestion-specific domain exceptions."""


class IngestionError(Exception):
    """Base exception for all document ingestion errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class IngestionConfigurationError(IngestionError):
    """Raised when ingestion configuration is missing or invalid."""


class InvalidIngestionInputError(IngestionError):
    """Raised when ingestion input reference, path, or project identifier is invalid."""


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when an unsupported document format or MIME type is encountered during ingestion."""


class DocumentRetrievalError(IngestionError):
    """Raised when retrieving the document from object storage fails."""
