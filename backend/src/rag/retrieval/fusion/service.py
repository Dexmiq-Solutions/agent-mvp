"""Fusion service combining dense vector and sparse keyword candidate lists into a unified ranking."""

import time
from typing import Optional, Sequence

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.retrieval import FusionError
from rag.retrieval.config import FusionConfig
from rag.retrieval.fusion.strategy import (
    _DEFAULT_TOP_K,
    BaseFusionStrategy,
    ReciprocalRankFusionStrategy,
)
from rag.retrieval.models import FusedCandidate, KeywordSearchCandidate, VectorSearchCandidate

logger = get_logger(__name__)

_fusion_service: Optional["FusionService"] = None


class FusionService:
    """Service responsible for combining heterogeneous retrieval candidate lists into a unified ranking.

    Consumes dense candidates (from VectorSearchService) and sparse candidates (from
    KeywordSearchService), orchestrates deduplication and score accumulation via the configured
    fusion strategy (default: Reciprocal Rank Fusion / RRF), strictly enforces project isolation,
    and returns deterministic, unified FusedCandidate instances for downstream stages.
    """

    def __init__(
        self,
        strategy: Optional[BaseFusionStrategy] = None,
        config: Optional[FusionConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize FusionService.

        Args:
            strategy: Pre-configured BaseFusionStrategy instance. Defaults to RRF strategy.
            config: FusionConfig instance. Defaults to config loaded from application Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or FusionConfig.from_settings(self._settings)
        self._strategy = strategy or ReciprocalRankFusionStrategy(
            rrf_k=self._config.rrf_k,
            default_top_k=self._config.top_k,
        )

    @property
    def strategy(self) -> BaseFusionStrategy:
        """Return the active fusion strategy."""
        return self._strategy

    @property
    def config(self) -> FusionConfig:
        """Return the active fusion configuration."""
        return self._config

    def fuse(
        self,
        dense_results: Sequence[VectorSearchCandidate],
        sparse_results: Sequence[KeywordSearchCandidate],
        project_id: Optional[str] = None,
        top_k: Optional[int] = _DEFAULT_TOP_K,
    ) -> list[FusedCandidate]:
        """Combine dense and sparse search candidates into a unified, deduplicated ranking.

        Args:
            dense_results: Ordered sequence of candidates retrieved via dense vector search.
            sparse_results: Ordered sequence of candidates retrieved via sparse keyword search.
            project_id: Optional project/tenant ID strictly enforcing isolation.
            top_k: Optional output candidate limit override.

        Returns:
            list[FusedCandidate]: Deduplicated, scored, and ordered candidates.

        Raises:
            FusionError: If input sequences are malformed or fusion computation fails.
        """
        start_time = time.perf_counter()
        clean_project_id = project_id.strip() if (project_id and isinstance(project_id, str)) else None

        logger.debug(
            "Executing candidate fusion (strategy='%s', dense_count=%d, sparse_count=%d, project_id=%s, top_k=%s)",
            self._strategy.strategy_name,
            len(dense_results) if isinstance(dense_results, Sequence) else -1,
            len(sparse_results) if isinstance(sparse_results, Sequence) else -1,
            clean_project_id,
            top_k,
        )

        try:
            fused = self._strategy.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k,
                project_id=clean_project_id,
            )
        except FusionError:
            raise
        except Exception as exc:
            logger.error("Unexpected error during candidate fusion: %s", exc)
            raise FusionError(f"Unexpected error during candidate fusion: {exc}", original_error=exc) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Fusion completed: strategy='%s', project_id='%s', dense_in=%d, sparse_in=%d, unified_out=%d, latency_ms=%.2f",
            self._strategy.strategy_name,
            clean_project_id or "unspecified",
            len(dense_results) if isinstance(dense_results, Sequence) else 0,
            len(sparse_results) if isinstance(sparse_results, Sequence) else 0,
            len(fused),
            elapsed_ms,
        )

        return fused


def get_fusion_service(
    strategy: Optional[BaseFusionStrategy] = None,
    config: Optional[FusionConfig] = None,
    settings: Optional[Settings] = None,
) -> FusionService:
    """Retrieve or initialize the singleton FusionService instance.

    Args:
        strategy: Optional BaseFusionStrategy dependency override.
        config: Optional FusionConfig override.
        settings: Optional Settings override.

    Returns:
        FusionService: Configured fusion service instance.
    """
    global _fusion_service
    if _fusion_service is None or any(arg is not None for arg in (strategy, config, settings)):
        service = FusionService(
            strategy=strategy,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (strategy, config, settings)):
            _fusion_service = service
        return service
    return _fusion_service


def reset_fusion_service() -> None:
    """Reset the cached singleton FusionService instance (primarily for testing)."""
    global _fusion_service
    _fusion_service = None


def fuse_results(
    dense_results: Sequence[VectorSearchCandidate],
    sparse_results: Sequence[KeywordSearchCandidate],
    project_id: Optional[str] = None,
    top_k: Optional[int] = _DEFAULT_TOP_K,
    service: Optional[FusionService] = None,
) -> list[FusedCandidate]:
    """Convenience functional entrypoint to perform hybrid candidate fusion.

    Args:
        dense_results: Ordered candidates from dense vector search.
        sparse_results: Ordered candidates from sparse keyword search.
        project_id: Optional project/tenant ID strictly enforcing isolation.
        top_k: Optional output candidate limit override.
        service: Optional FusionService instance. Defaults to singleton service.

    Returns:
        list[FusedCandidate]: Deduplicated, scored, and ordered unified candidate chunks.
    """
    active_service = service or get_fusion_service()
    return active_service.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        project_id=project_id,
        top_k=top_k,
    )
