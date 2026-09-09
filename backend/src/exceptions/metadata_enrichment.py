"""Metadata enrichment-specific domain exceptions."""


class MetadataEnrichmentError(Exception):
    """Base exception for all document metadata enrichment errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidMetadataEnrichmentInputError(MetadataEnrichmentError):
    """Raised when metadata enrichment input is not a valid ChunkedDocument."""


class MetadataEnrichmentProcessingError(MetadataEnrichmentError):
    """Raised when an unexpected error occurs during document metadata enrichment."""


class MetadataEnrichmentConfigurationError(MetadataEnrichmentError):
    """Raised when an invalid metadata enrichment configuration or parameter is supplied."""
