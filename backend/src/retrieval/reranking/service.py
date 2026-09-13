"""Reranking service orchestrating cross-encoder re-ranking on eligible candidates."""

import time
from typing import Any, Callable, Mapping, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import (
    RerankingConfigurationError,
    RerankingError,
    RerankingValidationError,
)
from retrieval.config import RerankingConfig
from retrieval.models import FusedCandidate, ProcessedQuery, RerankedCandidate, RetrievalQuerySet
from retrieval.reranking.base import BaseReranker, ScoredDocument
from retrieval.reranking.voyage import VoyageReranker

logger = get_logger(__name__)

_reranking_service: Optional["RerankingService"] = None


def resolve_candidate_text(
    candidate: Any,
    content_lookup: Optional[Mapping[str, str] | Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Resolve raw chunk text for a candidate.

    Resolution precedence:
        1. External content_lookup (Mapping or Callable) if provided.
        2. Candidate's top-level attributes: candidate.content or candidate.text.
        3. Candidate's metadata dictionary: 'content', 'text', 'chunk_text', or 'representation_text'.

    Args:
        candidate: Candidate instance (e.g. FusedCandidate).
        content_lookup: Optional external lookup mapping or callable.

    Returns:
        Optional[str]: Extracted chunk text or None if missing.
    """
    chunk_id = getattr(candidate, "chunk_id", None)

    # 1. External content_lookup
    if content_lookup is not None and chunk_id is not None:
        if isinstance(content_lookup, Mapping):
            val = content_lookup.get(chunk_id)
            if isinstance(val, str) and val.strip():
                return val
        elif callable(content_lookup):
            val = content_lookup(chunk_id)
            if isinstance(val, str) and val.strip():
                return val

    # 2. Candidate attributes
    for attr in ("content", "text"):
        val = getattr(candidate, attr, None)
        if isinstance(val, str) and val.strip():
            return val

    # 3. Candidate metadata dictionary
    meta = getattr(candidate, "metadata", None)
    if isinstance(meta, dict):
        for key in ("content", "text", "chunk_text", "representation_text"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val

    return None


class RerankingService:
    """Service responsible for applying cross-encoder reranking to retrieval candidates.

    Positions directly between Metadata Filtering and downstream Context Assembly.
    Consumes project-scoped, metadata-eligible candidates, scores them against the query
    using a cross-encoder model, and returns deterministically reordered RerankedCandidate
    instances with updated ranks and attached relevance scores.
    """

    def __init__(
        self,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankingConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize RerankingService.

        Args:
            reranker: BaseReranker instance. Defaults to configured VoyageReranker.
            config: RerankingConfig instance. Defaults to config from Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or RerankingConfig.from_settings(self._settings)
        self._reranker = reranker or VoyageReranker(
            model=self._config.model,
            timeout_seconds=self._config.timeout_seconds,
            settings=self._settings,
        )

    @property
    def config(self) -> RerankingConfig:
        """Return active reranking configuration."""
        return self._config

    @property
    def reranker(self) -> BaseReranker:
        """Return active reranker provider."""
        return self._reranker

    def _extract_query_text(self, query: str | ProcessedQuery | RetrievalQuerySet | Any) -> str:
        """Extract validated non-empty query string from supported query representations."""
        if isinstance(query, str):
            clean = query.strip()
        elif isinstance(query, ProcessedQuery):
            clean = query.processed_query.strip()
        elif isinstance(query, RetrievalQuerySet):
            clean = query.original_query.strip()
        elif hasattr(query, "original_query") and isinstance(query.original_query, str):
            clean = query.original_query.strip()
        elif hasattr(query, "processed_query") and isinstance(query.processed_query, str):
            clean = query.processed_query.strip()
        else:
            raise RerankingValidationError(
                f"Expected query to be str, ProcessedQuery, or RetrievalQuerySet, got '{type(query).__name__}'."
            )

        if not clean:
            raise RerankingValidationError("Query string cannot be empty or whitespace only.")
        return clean

    async def rerank(
        self,
        query: str | ProcessedQuery | RetrievalQuerySet,
        candidates: Sequence[FusedCandidate],
        project_id: Optional[str] = None,
        content_lookup: Optional[Mapping[str, str] | Callable[[str], Optional[str]]] = None,
        candidate_limit: Optional[int] = None,
        result_limit: Optional[int] = None,
    ) -> list[RerankedCandidate] | list[FusedCandidate]:
        """Rerank candidates using the cross-encoder model.

        Args:
            query: Query string or query domain model representation.
            candidates: Sequence of FusedCandidate instances from Metadata Filtering.
            project_id: Optional project/tenant ID strictly enforcing project isolation.
            content_lookup: Optional mapping or callable to resolve chunk_id -> raw chunk text.
            candidate_limit: Optional override for maximum input candidates sent to reranker.
            result_limit: Optional override for maximum output candidates returned.

        Returns:
            list[RerankedCandidate] | list[FusedCandidate]: Reordered candidates with cross-encoder
                scores and ranks when enabled, or sliced input candidates when disabled.

        Raises:
            RerankingValidationError: If inputs, candidates, or provider output are malformed.
            RerankingProviderError: If the external reranker provider fails.
            RerankingTimeoutError: If the external reranker call times out.
        """
        # 1. Input Validation
        query_text = self._extract_query_text(query)

        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise RerankingValidationError(
                f"Expected Sequence for candidates, got '{type(candidates).__name__}'."
            )

        if len(candidates) == 0:
            return []

        # 2. Project Isolation Parameter Normalization
        clean_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise RerankingValidationError("project_id must be a non-empty string when specified.")
            clean_project_id = project_id.strip()

        # 3. Defensive Candidate Project Isolation Check & Deduplication
        seen_chunk_ids: set[str] = set()
        deduped_candidates: list[FusedCandidate] = []
        discarded_count = 0

        for cand in candidates:
            if cand is None or not hasattr(cand, "chunk_id"):
                discarded_count += 1
                logger.warning("Discarded null or malformed candidate entry during reranking")
                continue

            cid = str(cand.chunk_id).strip()
            if not cid:
                discarded_count += 1
                logger.warning("Discarded candidate with empty chunk_id")
                continue

            # Defensive project isolation check
            if clean_project_id is not None:
                cand_pid = getattr(cand, "project_id", None)
                if not isinstance(cand_pid, str) or cand_pid.strip() != clean_project_id:
                    discarded_count += 1
                    logger.warning(
                        "Discarded leaked candidate '%s' in reranking: project_id mismatch "
                        "(expected='%s', actual='%s')",
                        cid,
                        clean_project_id,
                        cand_pid,
                    )
                    continue

            # Deduplication: preserve first occurrence (highest first-stage ranking)
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            deduped_candidates.append(cand)

        if len(deduped_candidates) == 0:
            return []

        # 4. Resolve limits
        eff_cand_limit = self._config.candidate_limit if candidate_limit is None else candidate_limit
        if not isinstance(eff_cand_limit, int) or eff_cand_limit <= 0:
            raise RerankingValidationError(f"candidate_limit must be a positive integer, got {eff_cand_limit}.")

        eff_result_limit = self._config.result_limit if result_limit is None else result_limit
        if not isinstance(eff_result_limit, int) or eff_result_limit <= 0:
            raise RerankingValidationError(f"result_limit must be a positive integer, got {eff_result_limit}.")

        # 5. Check if Reranking is Disabled via Configuration
        if not self._config.enabled:
            logger.debug("Reranking is disabled via configuration. Bypassing cross-encoder scoring.")
            return list(deduped_candidates)[:eff_result_limit]

        # 6. Bound Candidates to Candidate Limit
        bounded_candidates = deduped_candidates[:eff_cand_limit]

        # 7. Resolve Candidate Chunk Texts
        doc_texts: list[str] = []
        for cand in bounded_candidates:
            text = resolve_candidate_text(cand, content_lookup=content_lookup)
            if text is None or not isinstance(text, str) or not text.strip():
                raise RerankingValidationError(
                    f"Could not resolve non-empty chunk text for candidate '{cand.chunk_id}'. "
                    f"Cross-encoder reranking requires chunk text."
                )
            doc_texts.append(text)

        start_time = time.perf_counter()
        logger.debug(
            "Executing cross-encoder reranking: model='%s', candidates_in=%d, bounded=%d, project_id='%s'",
            self._reranker.model_name,
            len(candidates),
            len(bounded_candidates),
            clean_project_id or "unspecified",
        )

        # 8. Single Batched External Cross-Encoder Call
        try:
            scored_docs: list[ScoredDocument] = await self._reranker.rerank(
                query=query_text,
                documents=doc_texts,
                top_k=None,
            )
        except RerankingError:
            raise
        except Exception as exc:
            logger.error("Unexpected error executing cross-encoder rerank: %s", exc)
            raise RerankingError(f"Unexpected reranking error: {exc}", original_error=exc) from exc

        # 9. Map and Deterministically Re-rank Candidates
        # Validate that all returned indices map to bounded_candidates
        paired_results: list[tuple[float, str, FusedCandidate]] = []
        seen_result_indices: set[int] = set()

        for sd in scored_docs:
            if sd.index in seen_result_indices:
                raise RerankingValidationError(f"Duplicate result index '{sd.index}' returned by reranker.")
            seen_result_indices.add(sd.index)

            cand = bounded_candidates[sd.index]
            # Primary sort key: descending score; Deterministic tie-breaker: ascending chunk_id
            paired_results.append((sd.score, cand.chunk_id, cand))

        # Sort deterministically: highest score first, stable tie-breaking on chunk_id
        paired_results.sort(key=lambda item: (-item[0], item[1]))

        # 10. Construct RerankedCandidate Models with 1-based Ranks
        reranked_candidates: list[RerankedCandidate] = []
        for new_rank, (rerank_score, _, cand) in enumerate(paired_results, start=1):
            reranked_candidates.append(
                RerankedCandidate.from_candidate(
                    candidate=cand,
                    rerank_score=rerank_score,
                    rank=new_rank,
                )
            )

        # 11. Truncate to Result Limit
        final_results = reranked_candidates[:eff_result_limit]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Cross-encoder reranking completed: project_id='%s', model='%s', candidates_in=%d, "
            "bounded_in=%d, results_out=%d, latency_ms=%.2f",
            clean_project_id or "unspecified",
            self._reranker.model_name,
            len(candidates),
            len(bounded_candidates),
            len(final_results),
            elapsed_ms,
        )

        return final_results


def get_reranking_service(
    reranker: Optional[BaseReranker] = None,
    config: Optional[RerankingConfig] = None,
    settings: Optional[Settings] = None,
) -> RerankingService:
    """Retrieve or initialize the singleton RerankingService instance.

    Args:
        reranker: Optional BaseReranker dependency override.
        config: Optional RerankingConfig override.
        settings: Optional Settings override.

    Returns:
        RerankingService: Configured service instance.
    """
    global _reranking_service
    if _reranking_service is None or any(arg is not None for arg in (reranker, config, settings)):
        service = RerankingService(
            reranker=reranker,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (reranker, config, settings)):
            _reranking_service = service
        return service
    return _reranking_service


def reset_reranking_service() -> None:
    """Reset the cached singleton RerankingService instance (useful for tests)."""
    global _reranking_service
    _reranking_service = None


async def rerank_candidates(
    query: str | ProcessedQuery | RetrievalQuerySet,
    candidates: Sequence[FusedCandidate],
    project_id: Optional[str] = None,
    content_lookup: Optional[Mapping[str, str] | Callable[[str], Optional[str]]] = None,
    candidate_limit: Optional[int] = None,
    result_limit: Optional[int] = None,
    service: Optional[RerankingService] = None,
) -> list[RerankedCandidate] | list[FusedCandidate]:
    """Convenience functional entrypoint to rerank candidates with cross-encoder model.

    Args:
        query: Query string or query representation.
        candidates: Sequence of FusedCandidate instances from Metadata Filtering.
        project_id: Optional project/tenant ID enforcing project isolation.
        content_lookup: Optional mapping or callable to resolve chunk_id -> raw chunk text.
        candidate_limit: Optional input candidate limit override.
        result_limit: Optional output result limit override.
        service: Optional RerankingService instance override.

    Returns:
        list[RerankedCandidate] | list[FusedCandidate]: Reordered candidates with updated scores.
    """
    active_service = service or get_reranking_service()
    return await active_service.rerank(
        query=query,
        candidates=candidates,
        project_id=project_id,
        content_lookup=content_lookup,
        candidate_limit=candidate_limit,
        result_limit=result_limit,
    )
