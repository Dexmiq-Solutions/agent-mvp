"""Retrieval-specific domain exceptions."""


class RetrievalError(Exception):
    """Base exception for all retrieval pipeline errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class QueryPreprocessingError(RetrievalError):
    """Base exception for query preprocessing stage failures."""


class InvalidQueryError(QueryPreprocessingError):
    """Raised when query input validation fails (e.g. non-string or structurally invalid input)."""


class EmptyQueryError(InvalidQueryError):
    """Raised when the input query is empty or contains only whitespace."""


class QueryLengthExceededError(InvalidQueryError):
    """Raised when the input query exceeds the maximum allowed length."""

    def __init__(
        self,
        message: str,
        length: int,
        max_length: int,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error=original_error)
        self.length = length
        self.max_length = max_length


class QueryTransformationError(RetrievalError):
    """Base exception for query transformation stage failures."""


class TransformationValidationError(QueryTransformationError):
    """Raised when query transformation output fails validation."""


class TransformationProviderError(QueryTransformationError):
    """Raised when an external LLM provider encounters an error during query transformation."""


class TransformationTimeoutError(TransformationProviderError):
    """Raised when the LLM provider call times out."""


class TransformationUnavailableError(TransformationProviderError):
    """Raised when the LLM provider service is unreachable or connection fails."""


class TransformationFallbackLimitExceededError(QueryTransformationError):
    """Raised when retrieval fallback attempts exceed the configured maximum bound."""


class QueryEmbeddingError(RetrievalError):
    """Base exception for query embedding stage failures."""


class VectorRetrievalError(RetrievalError):
    """Base exception for vector similarity search retrieval failures."""
