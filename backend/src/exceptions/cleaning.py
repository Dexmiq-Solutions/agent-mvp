"""Cleaning-specific domain exceptions."""


class CleaningError(Exception):
    """Base exception for all document cleaning errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidCleaningInputError(CleaningError):
    """Raised when cleaning input document or payload is invalid."""


class CleaningProcessingError(CleaningError):
    """Raised when an unexpected fatal error occurs during document cleaning."""


class CleaningConfigurationError(CleaningError):
    """Raised when an invalid cleaning threshold or rule configuration is supplied."""
