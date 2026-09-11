"""Retrieval package encompassing preprocessing and query transformation.

Serves the first stages of the Retrieval pipeline:
User Query -> Query Preprocessing -> Query Transformation -> Retrieval Query Set -> Retrieval
"""

from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    QueryEmbeddingError,
    QueryLengthExceededError,
    QueryPreprocessingError,
    QueryTransformationError,
    RetrievalError,
    TransformationFallbackLimitExceededError,
    TransformationProviderError,
    TransformationTimeoutError,
    TransformationUnavailableError,
    TransformationValidationError,
)
from retrieval.config import (
    QueryPreprocessingConfig,
    QueryTransformationConfig,
)
from retrieval.embedding import (
    QueryEmbeddingService,
    embed_query_set,
    get_query_embedding_service,
    reset_query_embedding_service,
)
from retrieval.models import (
    EmbeddedQuery,
    EmbeddedQuerySet,
    PolicyDecision,
    ProcessedQuery,
    RetrievalQuerySet,
)
from retrieval.preprocessing import (
    QueryPreprocessingService,
    QueryPreprocessor,
    get_query_preprocessor,
    preprocess_query,
    reset_query_preprocessor,
)
from retrieval.transformation import (
    AdaptiveTransformationPolicy,
    BaseLLMClient,
    BaseTransformationPolicy,
    BaseTransformationStrategy,
    LLMQueryRewriteStrategy,
    OpenAICompatibleLLMClient,
    QueryTransformationService,
    TransformationOutputValidator,
    ValidationResult,
    get_query_transformation_service,
    reset_query_transformation_service,
    transform_query,
)

__all__ = [
    # Domain Models
    "ProcessedQuery",
    "PolicyDecision",
    "RetrievalQuerySet",
    "EmbeddedQuery",
    "EmbeddedQuerySet",
    # Configuration
    "QueryPreprocessingConfig",
    "QueryTransformationConfig",
    # Preprocessing Service & Helpers
    "QueryPreprocessor",
    "QueryPreprocessingService",
    "get_query_preprocessor",
    "reset_query_preprocessor",
    "preprocess_query",
    # Transformation Subsystem & Helpers
    "QueryTransformationService",
    "get_query_transformation_service",
    "reset_query_transformation_service",
    "transform_query",
    "BaseTransformationPolicy",
    "AdaptiveTransformationPolicy",
    "BaseTransformationStrategy",
    "LLMQueryRewriteStrategy",
    "BaseLLMClient",
    "OpenAICompatibleLLMClient",
    "TransformationOutputValidator",
    "ValidationResult",
    # Query Embedding Subsystem & Helpers
    "QueryEmbeddingService",
    "get_query_embedding_service",
    "reset_query_embedding_service",
    "embed_query_set",
    # Domain Exceptions
    "RetrievalError",
    "QueryPreprocessingError",
    "InvalidQueryError",
    "EmptyQueryError",
    "QueryLengthExceededError",
    "QueryTransformationError",
    "TransformationValidationError",
    "TransformationProviderError",
    "TransformationTimeoutError",
    "TransformationUnavailableError",
    "TransformationFallbackLimitExceededError",
    "QueryEmbeddingError",
]

