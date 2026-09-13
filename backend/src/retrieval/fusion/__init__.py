"""Hybrid retrieval fusion subsystem implementing Reciprocal Rank Fusion (RRF)."""

from retrieval.fusion.service import (
    FusionService,
    fuse_results,
    get_fusion_service,
    reset_fusion_service,
)
from retrieval.fusion.strategy import (
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
