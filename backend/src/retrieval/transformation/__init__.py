"""Query Transformation stage of the Retrieval pipeline.

Serves as the second stage of the Retrieval pipeline:
User Query -> Query Preprocessing -> Query Transformation -> Retrieval Query Set -> Retrieval
"""

from retrieval.transformation.policy import (
    AdaptiveTransformationPolicy,
    BaseTransformationPolicy,
)
from retrieval.transformation.provider import (
    BaseLLMClient,
    OpenAICompatibleLLMClient,
)
from retrieval.transformation.service import (
    QueryTransformationService,
    get_query_transformation_service,
    reset_query_transformation_service,
    transform_query,
)
from retrieval.transformation.strategy import (
    BaseTransformationStrategy,
    LLMQueryRewriteStrategy,
)
from retrieval.transformation.validator import (
    TransformationOutputValidator,
    ValidationResult,
)

__all__ = [
    # Policy
    "BaseTransformationPolicy",
    "AdaptiveTransformationPolicy",
    # Strategy
    "BaseTransformationStrategy",
    "LLMQueryRewriteStrategy",
    # Providers
    "BaseLLMClient",
    "OpenAICompatibleLLMClient",
    # Validation
    "TransformationOutputValidator",
    "ValidationResult",
    # Service & Helpers
    "QueryTransformationService",
    "get_query_transformation_service",
    "reset_query_transformation_service",
    "transform_query",
]
