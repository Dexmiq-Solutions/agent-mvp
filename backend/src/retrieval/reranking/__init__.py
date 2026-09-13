"""Cross-encoder reranking subsystem for the retrieval pipeline."""

from retrieval.reranking.base import BaseReranker, ScoredDocument
from retrieval.reranking.service import (
    RerankingService,
    get_reranking_service,
    rerank_candidates,
    reset_reranking_service,
    resolve_candidate_text,
)
from retrieval.reranking.voyage import VoyageReranker

__all__ = [
    "BaseReranker",
    "ScoredDocument",
    "VoyageReranker",
    "RerankingService",
    "get_reranking_service",
    "reset_reranking_service",
    "rerank_candidates",
    "resolve_candidate_text",
]
