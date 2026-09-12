"""Keyword Search (lexical/sparse retrieval) module."""

from retrieval.keyword.encoder import (
    BaseSparseEncoder,
    TechnicalSparseEncoder,
    get_sparse_encoder,
    reset_sparse_encoder,
)
from retrieval.keyword.service import (
    KeywordSearchService,
    get_keyword_search_service,
    reset_keyword_search_service,
    search_keywords,
)

__all__ = [
    "BaseSparseEncoder",
    "TechnicalSparseEncoder",
    "get_sparse_encoder",
    "reset_sparse_encoder",
    "KeywordSearchService",
    "get_keyword_search_service",
    "reset_keyword_search_service",
    "search_keywords",
]
