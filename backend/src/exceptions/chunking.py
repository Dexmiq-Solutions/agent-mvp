"""Chunking-specific domain exceptions."""


class ChunkingError(Exception):
    """Base exception for all document chunking errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidChunkingInputError(ChunkingError):
    """Raised when chunking input document is not a valid NormalizedDocument."""


class ChunkingProcessingError(ChunkingError):
    """Raised when an unexpected fatal error occurs during document chunking."""


class ChunkingConfigurationError(ChunkingError):
    """Raised when an invalid chunking configuration or parameter is supplied."""
