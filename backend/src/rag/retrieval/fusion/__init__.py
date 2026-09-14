"""Hybrid retrieval fusion subsystem implementing Reciprocal Rank Fusion (RRF)."""

from rag.retrieval.fusion.service import (
    FusionService,
    fuse_results,
    get_fusion_service,
    reset_fusion_service,
)
from rag.retrieval.fusion.strategy import (
    BaseFusionStrategy,
    ReciprocalRankFusionStrategy,
)

__all__ = [
    "BaseFusionStrategy",
    "ReciprocalRankFusionStrategy",
    "FusionService",
    "get_fusion_service",
    "reset_fusion_service",
    "fuse_results",
]
