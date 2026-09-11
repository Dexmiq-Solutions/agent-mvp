"""Embedding-specific domain exceptions."""


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration or API credentials are missing or invalid."""


class EmbeddingConnectionError(EmbeddingError):
    """Raised when unable to establish connection to the embedding provider or on timeouts."""


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when authentication with the embedding provider fails."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the embedding provider rate limit is exceeded."""


class EmbeddingInputValidationError(EmbeddingError):
    """Raised when input text or batches for embedding generation are invalid."""


class EmbeddingRequestError(EmbeddingError):
    """Raised when an embedding request fails or returns an error from the provider."""


class EmbeddingResponseValidationError(EmbeddingError):
    """Raised when the embedding provider returns an invalid, malformed, or mismatched response."""
