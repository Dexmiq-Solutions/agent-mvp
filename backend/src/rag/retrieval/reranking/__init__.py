"""Cross-encoder reranking subsystem for the retrieval pipeline."""

from rag.retrieval.reranking.base import BaseReranker, ScoredDocument
from rag.retrieval.reranking.jina import JinaReranker
from rag.retrieval.reranking.service import (
    RerankingService,
    get_reranker_provider,
    get_reranking_service,
    rerank_candidates,
    reset_reranking_service,
    resolve_candidate_text,
)
from rag.retrieval.reranking.voyage import VoyageReranker

__all__ = [
    "BaseReranker",
    "ScoredDocument",
    "JinaReranker",
    "VoyageReranker",
    "get_reranker_provider",
    "RerankingService",
    "get_reranking_service",
    "reset_reranking_service",
    "rerank_candidates",
    "resolve_candidate_text",
]
