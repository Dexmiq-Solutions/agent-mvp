"""Chunk Hydration service orchestrating batched resolution against PostgreSQL Content Store."""

import time
from typing import Any, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import (
    ChunkHydrationError,
    ChunkHydrationValidationError,
    ChunkNotFoundError,
    DatabaseRetrievalError,
)
from models.chunk import ChunkModel
from retrieval.hydration.config import ChunkHydrationConfig
from retrieval.hydration.repository import (
    BaseChunkRepository,
    get_chunk_repository,
)
from retrieval.models import HydratedCandidate

logger = get_logger(__name__)

_chunk_hydration_service: Optional["ChunkHydrationService"] = None


class ChunkHydrationService:
    """Service responsible for resolving ranked chunk references into authoritative stored records.

    Positions directly between Reranking and Context Assembly. Consumes project-scoped,
    ranked chunk candidate references (e.g. RerankedCandidate or FusedCandidate), performs
    a single batched lookup against PostgreSQL Content Store, deterministically restores
    the original reranking order, and returns fully hydrated candidates ready for downstream
    Context Assembly.
    """

    def __init__(
        self,
        repository: Optional[BaseChunkRepository] = None,
        config: Optional[ChunkHydrationConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize ChunkHydrationService.

        Args:
            repository: Optional BaseChunkRepository override. Defaults to global repository.
            config: Optional ChunkHydrationConfig override. Defaults to config from Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or ChunkHydrationConfig.from_settings(self._settings)
        self._repository = repository or get_chunk_repository()

    @property
    def config(self) -> ChunkHydrationConfig:
        """Return active hydration configuration."""
        return self._config

    @property
    def repository(self) -> BaseChunkRepository:
        """Return active chunk repository."""
        return self._repository

    async def hydrate(
        self,
        candidates: Sequence[Any],
        project_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        fail_on_missing: Optional[bool] = None,
    ) -> list[HydratedCandidate]:
        """Hydrate ranked chunk candidate references against the PostgreSQL Content Store.

        Args:
            candidates: Sequence of ranked candidate references (e.g. RerankedCandidate, FusedCandidate).
            project_id: Optional project/tenant ID strictly enforcing isolation boundary.
            session: Optional active AsyncSession for transactional database queries.
            fail_on_missing: Optional override to raise ChunkNotFoundError on missing chunks.

        Returns:
            list[HydratedCandidate]: Fully hydrated candidates preserving original reranking order.

        Raises:
            ChunkHydrationValidationError: If inputs or candidates fail validation.
            ChunkNotFoundError: If chunks are missing and fail_on_missing is True.
            DatabaseRetrievalError: If PostgreSQL Content Store query fails.
        """
        start_time = time.perf_counter()

        # 1. Input Validation
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ChunkHydrationValidationError(
                f"Expected Sequence for candidates, got '{type(candidates).__name__}'."
            )

        # 2. Empty Input Short-Circuit (Zero DB Queries)
        if len(candidates) == 0:
            return []

        # 3. Project ID Resolution & Normalization
        eff_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise ChunkHydrationValidationError("project_id must be a non-empty string when specified.")
            eff_project_id = project_id.strip()

        # If project_id was not provided explicitly, attempt to infer from first valid candidate
        if eff_project_id is None:
            for cand in candidates:
                cand_pid = getattr(cand, "project_id", None)
                if isinstance(cand_pid, str) and cand_pid.strip():
                    eff_project_id = cand_pid.strip()
                    break

        if eff_project_id is None:
            raise ChunkHydrationValidationError(
                "project_id must be provided explicitly or present on input candidates."
            )

        # 4. Defensive Candidate Filtering & Deduplication
        seen_chunk_ids: set[str] = set()
        valid_candidates: list[Any] = []
        discarded_count = 0

        for cand in candidates:
            if cand is None or not hasattr(cand, "chunk_id"):
                discarded_count += 1
                logger.warning("Discarded null or malformed candidate entry during chunk hydration")
                continue

            cid = str(cand.chunk_id).strip()
            if not cid:
                discarded_count += 1
                logger.warning("Discarded candidate with empty chunk_id during hydration")
                continue

            # Defensive project isolation check
            cand_pid = getattr(cand, "project_id", None)
            if cand_pid is not None and str(cand_pid).strip() != eff_project_id:
                discarded_count += 1
                logger.warning(
                    "Discarded cross-tenant candidate '%s' during hydration: project_id mismatch "
                    "(expected='%s', actual='%s')",
                    cid,
                    eff_project_id,
                    cand_pid,
                )
                continue

            # Deduplication: preserve first occurrence (highest ranking position)
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            valid_candidates.append(cand)

        if not valid_candidates:
            return []

        # 5. Bound Candidates to Configured Maximum Batch Size
        if len(valid_candidates) > self._config.max_batch_size:
            logger.warning(
                "Truncating input candidates from %d to max_batch_size=%d",
                len(valid_candidates),
                self._config.max_batch_size,
            )
            valid_candidates = valid_candidates[: self._config.max_batch_size]

        target_chunk_ids = [c.chunk_id for c in valid_candidates]

        # 6. Single Batched PostgreSQL Lookup (No N+1)
        try:
            records_by_id = await self._repository.fetch_chunks(
                project_id=eff_project_id,
                chunk_ids=target_chunk_ids,
                session=session,
            )
        except ChunkHydrationError:
            raise
        except Exception as exc:
            logger.error(
                "Repository error fetching chunks for project '%s': %s",
                eff_project_id,
                exc,
            )
            raise DatabaseRetrievalError(
                f"Failed to fetch chunks from repository for project '{eff_project_id}': {exc}",
                original_error=exc,
            ) from exc

        # 7. Missing Chunk Detection & Consistency Audit
        missing_chunk_ids = [cid for cid in target_chunk_ids if cid not in records_by_id]
        if missing_chunk_ids:
            logger.warning(
                "Content Store inconsistency detected: %d chunk(s) missing from PostgreSQL for project '%s': %s",
                len(missing_chunk_ids),
                eff_project_id,
                sorted(missing_chunk_ids),
            )
            should_fail = (
                self._config.fail_on_missing
                if fail_on_missing is None
                else fail_on_missing
            )
            if should_fail:
                raise ChunkNotFoundError(
                    f"Content Store missing {len(missing_chunk_ids)} chunk(s) for project '{eff_project_id}': "
                    f"{sorted(missing_chunk_ids)}",
                    missing_chunk_ids=missing_chunk_ids,
                    project_id=eff_project_id,
                )

        # 8. Deterministic Order Restoration & Candidate Hydration
        hydrated_results: list[HydratedCandidate] = []

        for cand in valid_candidates:
            record: Optional[ChunkModel] = records_by_id.get(cand.chunk_id)
            if record is None:
                # Omit missing records; do not fabricate content
                continue

            # Validate Document Identity
            cand_doc_id = getattr(cand, "document_id", None)
            record_doc_id = getattr(record, "document_id", None)
            if cand_doc_id and record_doc_id and str(cand_doc_id).strip() != str(record_doc_id).strip():
                logger.warning(
                    "Document ID mismatch for chunk '%s' during hydration: expected='%s', record='%s'. Skipping.",
                    cand.chunk_id,
                    cand_doc_id,
                    record_doc_id,
                )
                continue

            # Validate Document Version Identity if configured
            if self._config.verify_version_identity:
                cand_version = getattr(cand, "document_version_id", None)
                record_version = getattr(record, "document_version_id", None)
                if cand_version and record_version and str(cand_version).strip() != str(record_version).strip():
                    logger.warning(
                        "Document version mismatch for chunk '%s' during hydration: expected='%s', record='%s'. Skipping.",
                        cand.chunk_id,
                        cand_version,
                        record_version,
                    )
                    continue

            # Construct HydratedCandidate preserving retrieval rank and scores
            hydrated = HydratedCandidate.from_candidate_and_record(
                candidate=cand,
                record=record,
            )
            hydrated_results.append(hydrated)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Chunk hydration completed: project_id='%s', candidates_in=%d, valid_in=%d, "
            "hydrated_out=%d, missing=%d, latency_ms=%.2f",
            eff_project_id,
            len(candidates),
            len(valid_candidates),
            len(hydrated_results),
            len(missing_chunk_ids),
            elapsed_ms,
        )

        return hydrated_results


def get_chunk_hydration_service(
    repository: Optional[BaseChunkRepository] = None,
    config: Optional[ChunkHydrationConfig] = None,
    settings: Optional[Settings] = None,
) -> ChunkHydrationService:
    """Retrieve or initialize singleton ChunkHydrationService instance.

    Args:
        repository: Optional BaseChunkRepository override.
        config: Optional ChunkHydrationConfig override.
        settings: Optional Settings override.

    Returns:
        ChunkHydrationService: Configured service instance.
    """
    global _chunk_hydration_service
    if _chunk_hydration_service is None or any(arg is not None for arg in (repository, config, settings)):
        service = ChunkHydrationService(
            repository=repository,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (repository, config, settings)):
            _chunk_hydration_service = service
        return service
    return _chunk_hydration_service


def reset_chunk_hydration_service() -> None:
    """Reset the cached singleton ChunkHydrationService instance (useful for tests)."""
    global _chunk_hydration_service
    _chunk_hydration_service = None


async def hydrate_candidates(
    candidates: Sequence[Any],
    project_id: Optional[str] = None,
    session: Optional[AsyncSession] = None,
    fail_on_missing: Optional[bool] = None,
    service: Optional[ChunkHydrationService] = None,
) -> list[HydratedCandidate]:
    """Convenience functional entrypoint to hydrate ranked chunk candidates.

    Args:
        candidates: Sequence of ranked candidate references.
        project_id: Optional project/tenant ID enforcing project boundary.
        session: Optional active AsyncSession.
        fail_on_missing: Optional override to raise ChunkNotFoundError on missing chunks.
        service: Optional ChunkHydrationService override.

    Returns:
        list[HydratedCandidate]: Hydrated candidate records preserving rerank order.
    """
    active_service = service or get_chunk_hydration_service()
    return await active_service.hydrate(
        candidates=candidates,
        project_id=project_id,
        session=session,
        fail_on_missing=fail_on_missing,
    )
