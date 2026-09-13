"""Metadata Filtering service executing structured constraint filtering on unified candidates."""

import time
from typing import Any, Callable, Mapping, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import InvalidFilterError, MetadataFilteringError
from retrieval.config import MetadataFilteringConfig
from retrieval.filtering.evaluator import MetadataConditionEvaluator
from retrieval.filtering.models import MetadataFilter
from retrieval.models import FusedCandidate

logger = get_logger(__name__)

_metadata_filtering_service: Optional["MetadataFilteringService"] = None


class MetadataFilteringService:
    """Service responsible for applying structured metadata constraints to retrieval candidates.

    Positions directly between Hybrid Fusion (RRF) and downstream stages (Reranking / Fetch Chunks).
    Consumes unified `FusedCandidate` instances, filters them against structured metadata
    constraints (exact-match, set, range, boolean, and compound), strictly preserves tenant isolation,
    and returns eligible candidates without mutating relevance scores or rankings.
    """

    def __init__(
        self,
        evaluator: Optional[MetadataConditionEvaluator] = None,
        config: Optional[MetadataFilteringConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize MetadataFilteringService.

        Args:
            evaluator: MetadataConditionEvaluator instance. Defaults to configured evaluator.
            config: MetadataFilteringConfig instance. Defaults to config from Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or MetadataFilteringConfig.from_settings(self._settings)
        self._evaluator = evaluator or MetadataConditionEvaluator(
            strict_mode=self._config.strict_mode
        )

    @property
    def config(self) -> MetadataFilteringConfig:
        """Return active metadata filtering configuration."""
        return self._config

    @property
    def evaluator(self) -> MetadataConditionEvaluator:
        """Return active condition evaluator."""
        return self._evaluator

    def filter(
        self,
        candidates: Sequence[FusedCandidate],
        filter_spec: Optional[MetadataFilter | dict[str, Any]] = None,
        project_id: Optional[str] = None,
        metadata_lookup: Optional[Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any]]] = None,
    ) -> list[FusedCandidate]:
        """Filter unified candidates against metadata constraints, strictly preserving scores and ranks.

        Args:
            candidates: Ordered sequence of FusedCandidate instances from Fusion.
            filter_spec: Structured MetadataFilter instance, configuration dict, or None.
            project_id: Optional project/tenant ID strictly enforcing project isolation.
            metadata_lookup: Optional external dictionary or callable resolving chunk_id -> metadata.

        Returns:
            list[FusedCandidate]: Retained candidates that satisfy all requested constraints.

        Raises:
            MetadataFilteringError: If inputs are malformed or filter evaluation fails.
        """
        # 1. Input Sequence Validation
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise MetadataFilteringError(
                f"Expected Sequence for candidates, got '{type(candidates).__name__}'."
            )

        if len(candidates) == 0:
            return []

        # 2. Project Isolation Parameter Normalization
        clean_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise MetadataFilteringError("project_id must be a non-empty string when specified.")
            clean_project_id = project_id.strip()

        # 3. Resolve MetadataFilter Object
        resolved_filter: MetadataFilter
        if filter_spec is None:
            resolved_filter = MetadataFilter()
        elif isinstance(filter_spec, MetadataFilter):
            resolved_filter = filter_spec
        elif isinstance(filter_spec, dict):
            try:
                resolved_filter = MetadataFilter.from_dict(filter_spec)
            except InvalidFilterError:
                raise
            except Exception as exc:
                raise InvalidFilterError(f"Failed to parse filter specification dictionary: {exc}") from exc
        else:
            raise InvalidFilterError(
                f"Expected MetadataFilter, dict, or None for filter_spec, got '{type(filter_spec).__name__}'."
            )

        # 4. If filtering is disabled via configuration and no project isolation is requested
        if not self._config.enabled and clean_project_id is None:
            return list(candidates)

        start_time = time.perf_counter()
        logger.debug(
            "Executing metadata filtering (candidates_in=%d, project_id=%s, filter_is_empty=%s)",
            len(candidates),
            clean_project_id,
            resolved_filter.is_empty,
        )

        eligible: list[FusedCandidate] = []
        discarded_count = 0

        # 5. Filter Evaluation: O(N) single pass over unified candidate set
        for candidate in candidates:
            if candidate is None or not hasattr(candidate, "chunk_id"):
                discarded_count += 1
                logger.warning("Discarded null or malformed candidate entry during metadata filtering")
                continue

            # Defensive project isolation check
            if clean_project_id is not None:
                cand_pid = getattr(candidate, "project_id", None)
                if not isinstance(cand_pid, str) or cand_pid.strip() != clean_project_id:
                    discarded_count += 1
                    logger.warning(
                        "Discarded leaked candidate '%s' in metadata filtering: project_id mismatch "
                        "(expected='%s', actual='%s')",
                        getattr(candidate, "chunk_id", "unknown"),
                        clean_project_id,
                        cand_pid,
                    )
                    continue

            # Evaluate metadata constraints
            try:
                matches = self._evaluator.evaluate_filter(
                    metadata_filter=resolved_filter,
                    candidate=candidate,
                    project_id=clean_project_id,
                    metadata_lookup=metadata_lookup,
                )
            except Exception as exc:
                if isinstance(exc, MetadataFilteringError):
                    raise
                raise MetadataFilteringError(
                    f"Unexpected failure evaluating metadata filter on candidate '{candidate.chunk_id}': {exc}",
                    original_error=exc,
                ) from exc

            if matches:
                eligible.append(candidate)
            else:
                discarded_count += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Metadata filtering completed: project_id='%s', candidates_in=%d, eligible_out=%d, "
            "discarded=%d, latency_ms=%.2f",
            clean_project_id or "unspecified",
            len(candidates),
            len(eligible),
            discarded_count,
            elapsed_ms,
        )

        return eligible


def get_metadata_filtering_service(
    evaluator: Optional[MetadataConditionEvaluator] = None,
    config: Optional[MetadataFilteringConfig] = None,
    settings: Optional[Settings] = None,
) -> MetadataFilteringService:
    """Retrieve or initialize the singleton MetadataFilteringService instance.

    Args:
        evaluator: Optional MetadataConditionEvaluator dependency override.
        config: Optional MetadataFilteringConfig override.
        settings: Optional Settings override.

    Returns:
        MetadataFilteringService: Configured service instance.
    """
    global _metadata_filtering_service
    if _metadata_filtering_service is None or any(arg is not None for arg in (evaluator, config, settings)):
        service = MetadataFilteringService(
            evaluator=evaluator,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (evaluator, config, settings)):
            _metadata_filtering_service = service
        return service
    return _metadata_filtering_service


def reset_metadata_filtering_service() -> None:
    """Reset the cached singleton MetadataFilteringService instance (useful for tests)."""
    global _metadata_filtering_service
    _metadata_filtering_service = None


def filter_candidates(
    candidates: Sequence[FusedCandidate],
    filter_spec: Optional[MetadataFilter | dict[str, Any]] = None,
    project_id: Optional[str] = None,
    metadata_lookup: Optional[Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any]]] = None,
    service: Optional[MetadataFilteringService] = None,
) -> list[FusedCandidate]:
    """Convenience functional entrypoint to filter candidates by metadata constraints.

    Args:
        candidates: Unified candidates from Fusion.
        filter_spec: MetadataFilter instance, configuration dict, or None.
        project_id: Optional project/tenant ID enforcing project isolation.
        metadata_lookup: Optional chunk_id -> metadata dictionary or callable.
        service: Optional MetadataFilteringService instance override.

    Returns:
        list[FusedCandidate]: Retained candidates preserving scores and ranks.
    """
    active_service = service or get_metadata_filtering_service()
    return active_service.filter(
        candidates=candidates,
        filter_spec=filter_spec,
        project_id=project_id,
        metadata_lookup=metadata_lookup,
    )
