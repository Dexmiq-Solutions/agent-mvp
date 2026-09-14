"""Context Assembly service transforming hydrated chunks into structured retrieval context."""

import time
from typing import Any, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import ContextAssemblyValidationError
from retrieval.assembly.config import ContextAssemblyConfig
from retrieval.models import (
    AssembledContext,
    AssembledContextItem,
    ProcessedQuery,
    RetrievalQuerySet,
)

logger = get_logger(__name__)

_context_assembly_service: Optional["ContextAssemblyService"] = None


class ContextAssemblyService:
    """Service responsible for transforming hydrated chunks into structured retrieval context.

    Positions directly between Chunk Fetching / Hydration and downstream Relevance Check / Fallback.
    Consumes project-scoped, authoritative HydratedCandidate instances, strictly preserves
    retrieval rank ordering and identity coordinates, resolves the stored text representation,
    and returns a structured AssembledContext ready for relevance evaluation and generation.
    """

    def __init__(
        self,
        config: Optional[ContextAssemblyConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize ContextAssemblyService.

        Args:
            config: Optional ContextAssemblyConfig override. Defaults to config from Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or ContextAssemblyConfig.from_settings(self._settings)

    @property
    def config(self) -> ContextAssemblyConfig:
        """Return active context assembly configuration."""
        return self._config

    def _extract_query_string(self, query: Optional[Any]) -> Optional[str]:
        """Extract a clean query string representation if provided."""
        if query is None:
            return None
        if isinstance(query, str):
            clean = query.strip()
            return clean if clean else None
        if isinstance(query, ProcessedQuery):
            clean = query.processed_query.strip()
            return clean if clean else None
        if isinstance(query, RetrievalQuerySet):
            clean = query.original_query.strip()
            return clean if clean else None
        if hasattr(query, "original_query") and isinstance(query.original_query, str):
            clean = query.original_query.strip()
            return clean if clean else None
        if hasattr(query, "processed_query") and isinstance(query.processed_query, str):
            clean = query.processed_query.strip()
            return clean if clean else None
        clean = str(query).strip()
        return clean if clean else None

    def assemble(
        self,
        candidates: Sequence[Any],
        project_id: Optional[str] = None,
        query: Optional[Any] = None,
        max_context_items: Optional[int] = None,
        use_contextual_enrichment: Optional[bool] = None,
        strict_project_validation: Optional[bool] = None,
    ) -> AssembledContext:
        """Assemble hydrated candidates into a structured retrieval context.

        Args:
            candidates: Sequence of hydrated chunk candidates (e.g. HydratedCandidate).
            project_id: Optional project/tenant ID strictly enforcing isolation boundary.
            query: Optional query representation associated with retrieval.
            max_context_items: Optional override bounding the number of context items.
            use_contextual_enrichment: Optional override to prefer stored contextual content.
            strict_project_validation: Optional override to raise on cross-tenant items.

        Returns:
            AssembledContext: Structured retrieval context containing ordered context items.

        Raises:
            ContextAssemblyValidationError: If inputs or candidates fail validation.
        """
        start_time = time.perf_counter()

        # 1. Input Validation
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ContextAssemblyValidationError(
                f"Expected Sequence for candidates, got '{type(candidates).__name__}'."
            )

        query_str = self._extract_query_string(query)

        # 2. Project ID Resolution & Validation
        eff_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise ContextAssemblyValidationError("project_id must be a non-empty string when specified.")
            eff_project_id = project_id.strip()

        # If project_id was not provided explicitly, attempt to infer from first valid candidate
        if eff_project_id is None:
            for cand in candidates:
                cand_pid = getattr(cand, "project_id", None)
                if isinstance(cand_pid, str) and cand_pid.strip():
                    eff_project_id = cand_pid.strip()
                    break

        # 3. Empty Input Short-Circuit
        if len(candidates) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AssembledContext(
                items=(),
                project_id=eff_project_id or "",
                query=query_str,
                metadata={
                    "elapsed_ms": round(elapsed_ms, 2),
                    "candidates_in": 0,
                    "items_out": 0,
                    "total_characters": 0,
                },
            )

        if eff_project_id is None:
            raise ContextAssemblyValidationError(
                "project_id must be provided explicitly or present on input candidates."
            )

        # 4. Defensive Filtering, Tenant Isolation, and Order-Preserving Deduplication
        eff_strict_project = (
            self._config.strict_project_validation
            if strict_project_validation is None
            else strict_project_validation
        )

        seen_chunk_ids: set[str] = set()
        valid_candidates: list[Any] = []
        discarded_count = 0

        for cand in candidates:
            if cand is None or not hasattr(cand, "chunk_id"):
                discarded_count += 1
                logger.warning("Discarded null or malformed candidate entry during context assembly")
                continue

            cid = str(cand.chunk_id).strip()
            if not cid:
                discarded_count += 1
                logger.warning("Discarded candidate with empty chunk_id during context assembly")
                continue

            # Defensive tenant isolation check
            cand_pid = getattr(cand, "project_id", None)
            if cand_pid is not None and str(cand_pid).strip() != eff_project_id:
                if eff_strict_project:
                    raise ContextAssemblyValidationError(
                        f"Candidate '{cid}' project_id '{cand_pid}' violates project boundary '{eff_project_id}'."
                    )
                discarded_count += 1
                logger.warning(
                    "Discarded cross-tenant candidate '%s' during context assembly: project_id mismatch "
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

        # 5. Bound Candidates to Configured Context Limit
        limit = max_context_items if max_context_items is not None else self._config.max_context_items
        if limit is not None:
            if limit <= 0:
                raise ContextAssemblyValidationError(
                    f"max_context_items must be a positive integer, got {limit}."
                )
            if len(valid_candidates) > limit:
                logger.debug(
                    "Bounding context candidates from %d to limit=%d",
                    len(valid_candidates),
                    limit,
                )
                valid_candidates = valid_candidates[:limit]

        # 6. Transform to AssembledContextItem Preserving Order and Stored Representation
        eff_use_contextual = (
            self._config.use_contextual_enrichment
            if use_contextual_enrichment is None
            else use_contextual_enrichment
        )

        assembled_items: list[AssembledContextItem] = []
        for cand in valid_candidates:
            item = AssembledContextItem.from_hydrated_candidate(
                candidate=cand,
                use_contextual_enrichment=eff_use_contextual,
            )
            assembled_items.append(item)

        # 7. Metrics & Observability
        total_chars = sum(len(item.text) for item in assembled_items)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "Context assembly completed in %.2fms: project_id='%s', candidates_in=%d, items_out=%d, chars=%d",
            elapsed_ms,
            eff_project_id,
            len(candidates),
            len(assembled_items),
            total_chars,
        )

        return AssembledContext(
            items=tuple(assembled_items),
            project_id=eff_project_id,
            query=query_str,
            metadata={
                "elapsed_ms": round(elapsed_ms, 2),
                "candidates_in": len(candidates),
                "items_out": len(assembled_items),
                "discarded_candidates": discarded_count,
                "total_characters": total_chars,
            },
        )


def get_context_assembly_service(
    config: Optional[ContextAssemblyConfig] = None,
    settings: Optional[Settings] = None,
) -> ContextAssemblyService:
    """Retrieve or initialize the singleton ContextAssemblyService instance.

    Args:
        config: Optional ContextAssemblyConfig override.
        settings: Optional Settings override.

    Returns:
        ContextAssemblyService: Configured service instance.
    """
    global _context_assembly_service
    if _context_assembly_service is None or any(arg is not None for arg in (config, settings)):
        service = ContextAssemblyService(
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (config, settings)):
            _context_assembly_service = service
        return service
    return _context_assembly_service


def reset_context_assembly_service() -> None:
    """Reset the cached singleton ContextAssemblyService instance (useful for tests)."""
    global _context_assembly_service
    _context_assembly_service = None


def assemble_context(
    candidates: Sequence[Any],
    project_id: Optional[str] = None,
    query: Optional[Any] = None,
    max_context_items: Optional[int] = None,
    use_contextual_enrichment: Optional[bool] = None,
    strict_project_validation: Optional[bool] = None,
    service: Optional[ContextAssemblyService] = None,
) -> AssembledContext:
    """Convenience functional entrypoint to assemble hydrated candidates into structured context.

    Args:
        candidates: Sequence of hydrated chunk candidate records.
        project_id: Optional project/tenant ID enforcing project boundary.
        query: Optional retrieval query representation.
        max_context_items: Optional maximum number of items in output context.
        use_contextual_enrichment: Optional flag to prefer stored contextual content.
        strict_project_validation: Optional flag to raise on cross-tenant items.
        service: Optional ContextAssemblyService override.

    Returns:
        AssembledContext: Structured retrieval context preserving ranking and provenance.
    """
    active_service = service or get_context_assembly_service()
    return active_service.assemble(
        candidates=candidates,
        project_id=project_id,
        query=query,
        max_context_items=max_context_items,
        use_contextual_enrichment=use_contextual_enrichment,
        strict_project_validation=strict_project_validation,
    )


async def assemble_context_async(
    candidates: Sequence[Any],
    project_id: Optional[str] = None,
    query: Optional[Any] = None,
    max_context_items: Optional[int] = None,
    use_contextual_enrichment: Optional[bool] = None,
    strict_project_validation: Optional[bool] = None,
    service: Optional[ContextAssemblyService] = None,
) -> AssembledContext:
    """Async convenience wrapper for assemble_context in async pipeline chains."""
    return assemble_context(
        candidates=candidates,
        project_id=project_id,
        query=query,
        max_context_items=max_context_items,
        use_contextual_enrichment=use_contextual_enrichment,
        strict_project_validation=strict_project_validation,
        service=service,
    )
