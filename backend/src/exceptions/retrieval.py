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


class KeywordRetrievalError(RetrievalError):
    """Base exception for keyword/sparse retrieval failures."""


class SparseEncodingError(RetrievalError):
    """Base exception for sparse representation encoding failures."""


class FusionError(RetrievalError):
    """Base exception for hybrid retrieval fusion stage failures."""


class MetadataFilteringError(RetrievalError):
    """Base exception for metadata filtering stage failures."""


class InvalidFilterError(MetadataFilteringError):
    """Raised when a filter expression or constraint is structurally invalid or malformed."""


class FilterEvaluationError(MetadataFilteringError):
    """Raised when evaluation of a filter against candidate metadata encounters an error."""


class RerankingError(RetrievalError):
    """Base exception for reranking stage failures."""


class RerankingConfigurationError(RerankingError):
    """Raised when reranking configuration or parameters are invalid."""


class RerankingValidationError(RerankingError):
    """Raised when query, candidates, or provider outputs fail structural validation."""


class RerankingProviderError(RerankingError):
    """Raised when an external reranker provider encounters an error."""


class RerankingTimeoutError(RerankingProviderError):
    """Raised when an external reranker provider call times out."""


class ChunkHydrationError(RetrievalError):
    """Base exception for chunk fetching / hydration stage failures."""


class ChunkHydrationValidationError(ChunkHydrationError):
    """Raised when hydration input arguments or candidate references fail validation."""


class ChunkNotFoundError(ChunkHydrationError):
    """Raised when one or more requested chunks cannot be found in the PostgreSQL Content Store."""

    def __init__(
        self,
        message: str,
        missing_chunk_ids: list[str] | None = None,
        project_id: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error=original_error)
        self.missing_chunk_ids = list(missing_chunk_ids or [])
        self.project_id = project_id


class DatabaseRetrievalError(ChunkHydrationError):
    """Raised when PostgreSQL Content Store query fails during chunk hydration."""


class ProjectBoundaryViolationError(ChunkHydrationError):
    """Raised when candidate retrieval references violate project boundary constraints."""

