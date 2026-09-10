"""Contextual enrichment-specific domain exceptions."""


class ContextualEnrichmentError(Exception):
    """Base exception for all document contextual enrichment errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidContextualEnrichmentInputError(ContextualEnrichmentError):
    """Raised when contextual enrichment input is not a valid EnrichedDocument or contains invalid chunks."""


class ContextualEnrichmentConfigurationError(ContextualEnrichmentError):
    """Raised when an invalid contextual enrichment configuration or parameter is supplied."""


class ContextualEnrichmentProcessingError(ContextualEnrichmentError):
    """Raised when an error occurs during document contextual enrichment or project isolation violation."""


class ContextualEnrichmentProviderError(ContextualEnrichmentProcessingError):
    """Raised when an external model/provider fails during contextual enrichment."""
