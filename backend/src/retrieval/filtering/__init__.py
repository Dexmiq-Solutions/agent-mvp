"""Metadata Filtering package for RAG retrieval pipeline."""

from retrieval.filtering.evaluator import (
    CandidateMetadataResolver,
    MetadataConditionEvaluator,
)
from retrieval.filtering.models import (
    BaseFilterCondition,
    BooleanCondition,
    ComparisonOperator,
    CompoundCondition,
    ExactMatchCondition,
    LogicalOperator,
    MetadataFilter,
    RangeCondition,
    SetCondition,
)
from retrieval.filtering.qdrant import (
    to_qdrant_condition,
    to_qdrant_filter,
)
from retrieval.filtering.service import (
    MetadataFilteringService,
    filter_candidates,
    get_metadata_filtering_service,
    reset_metadata_filtering_service,
)

__all__ = [
    # Operators & Enums
    "ComparisonOperator",
    "LogicalOperator",
    # Condition Models
    "BaseFilterCondition",
    "ExactMatchCondition",
    "SetCondition",
    "RangeCondition",
    "BooleanCondition",
    "CompoundCondition",
    "MetadataFilter",
    # Evaluator
    "MetadataConditionEvaluator",
    "CandidateMetadataResolver",
    # Service & Facades
    "MetadataFilteringService",
    "get_metadata_filtering_service",
    "reset_metadata_filtering_service",
    "filter_candidates",
    # Qdrant Translation
    "to_qdrant_condition",
    "to_qdrant_filter",
]
