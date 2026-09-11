"""Query Preprocessing stage of the Retrieval pipeline.

Serves as the first stage of the Retrieval pipeline:
User Query -> Query Preprocessing -> Optional Query Transformation -> Query Embedding -> Retrieval
"""

from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    QueryLengthExceededError,
    QueryPreprocessingError,
    RetrievalError,
)
from retrieval.config import QueryPreprocessingConfig
from retrieval.models import ProcessedQuery
from retrieval.preprocessing import (
    QueryPreprocessingService,
    QueryPreprocessor,
    get_query_preprocessor,
    preprocess_query,
    reset_query_preprocessor,
)

__all__ = [
    # Domain Models
    "ProcessedQuery",
    # Configuration
    "QueryPreprocessingConfig",
    # Preprocessing Service & Helpers
    "QueryPreprocessor",
    "QueryPreprocessingService",
    "get_query_preprocessor",
    "reset_query_preprocessor",
    "preprocess_query",
    # Domain Exceptions
    "RetrievalError",
    "QueryPreprocessingError",
    "InvalidQueryError",
    "EmptyQueryError",
    "QueryLengthExceededError",
]
