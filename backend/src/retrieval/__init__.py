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
    VectorRetrievalError,
)
from retrieval.config import (
    QueryPreprocessingConfig,
    QueryTransformationConfig,
    VectorSearchConfig,
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
    VectorSearchCandidate,
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
from retrieval.vector import (
    VectorSearchService,
    get_vector_search_service,
    reset_vector_search_service,
    search_vectors,
)

__all__ = [
    # Domain Models
    "ProcessedQuery",
    "PolicyDecision",
    "RetrievalQuerySet",
    "EmbeddedQuery",
    "EmbeddedQuerySet",
    "VectorSearchCandidate",
    # Configuration
    "QueryPreprocessingConfig",
    "QueryTransformationConfig",
    "VectorSearchConfig",
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
    # Vector Search Subsystem & Helpers
    "VectorSearchService",
    "get_vector_search_service",
    "reset_vector_search_service",
    "search_vectors",
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
    "VectorRetrievalError",
]

