"""Acquisition-specific domain exceptions."""


class AcquisitionError(Exception):
    """Base exception for all acquisition-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class AcquisitionConfigurationError(AcquisitionError):
    """Raised when acquisition configuration is missing or invalid."""


class SourceDiscoveryError(AcquisitionError):
    """Raised when discovering source documents in storage fails."""


class UnsupportedDocumentError(AcquisitionError):
    """Raised when an unsupported document format is encountered or requested."""


class InvalidSourceReferenceError(AcquisitionError):
    """Raised when a source reference, path, or project identifier is invalid."""
