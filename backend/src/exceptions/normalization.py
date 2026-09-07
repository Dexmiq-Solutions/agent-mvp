"""Normalization-specific domain exceptions."""


class NormalizationError(Exception):
    """Base exception for all document normalization errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidNormalizationInputError(NormalizationError):
    """Raised when normalization input document or payload is invalid."""


class NormalizationProcessingError(NormalizationError):
    """Raised when an unexpected fatal error occurs during document normalization."""


class NormalizationConfigurationError(NormalizationError):
    """Raised when an invalid normalization configuration or parameter is supplied."""
