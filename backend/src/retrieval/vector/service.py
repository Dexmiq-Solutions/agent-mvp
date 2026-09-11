"""Vector Search service executing project-scoped similarity search against Qdrant."""

import math
import time
from typing import Optional

from app.core.logging import get_logger
from exceptions.retrieval import VectorRetrievalError
from exceptions.vector import VectorStoreError
from retrieval.config import VectorSearchConfig
from retrieval.models import EmbeddedQuerySet, VectorSearchCandidate
from storage.vector import BaseVectorStore, get_vector_store

logger = get_logger(__name__)

_vector_search_service: Optional["VectorSearchService"] = None


class VectorSearchService:
    """Service responsible for project-scoped dense similarity search against vector storage.

    Consumes EmbeddedQuerySet representations produced by Query Embedding, executes
    project-isolated similarity search via BaseVectorStore, validates candidate records,
    preserves query representation provenance, and returns ranked VectorSearchCandidate
    instances for downstream fusion and reranking.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[VectorSearchConfig] = None,
    ) -> None:
        """Initialize VectorSearchService.

        Args:
            vector_store: Pre-configured BaseVectorStore instance. Defaults to default vector store.
            config: VectorSearchConfig instance. Defaults to config loaded from application Settings.
        """
        self._config = config or VectorSearchConfig.from_settings()
        self._vector_store = vector_store or get_vector_store()

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the active vector store repository."""
        return self._vector_store

    @property
    def config(self) -> VectorSearchConfig:
        """Return the active vector search configuration."""
        return self._config

    async def search(
        self,
        embedded_query_set: EmbeddedQuerySet,
        project_id: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchCandidate]:
        """Perform project-scoped vector similarity search for all embedded query representations.

        Args:
            embedded_query_set: Validated EmbeddedQuerySet container from Query Embedding.
            project_id: Mandatory project/tenant ID strictly enforcing database-level isolation.
            top_k: Optional per-query result limit override. Defaults to config top_k.
            score_threshold: Optional similarity score threshold override. Defaults to config threshold.

        Returns:
            list[VectorSearchCandidate]: Ranked candidate matches preserving query provenance.

        Raises:
            VectorRetrievalError: If inputs are invalid or vector store infrastructure fails.
        """
        # 1. Project Isolation Validation
        if not isinstance(project_id, str) or not project_id.strip():
            raise VectorRetrievalError(
                "project_id must be a non-empty string to enforce project isolation."
            )
        clean_project_id = project_id.strip()

        # 2. Input Structure Validation
        if not isinstance(embedded_query_set, EmbeddedQuerySet):
            raise VectorRetrievalError(
                f"Expected EmbeddedQuerySet instance, got '{type(embedded_query_set).__name__}'."
            )
        if (
            embedded_query_set.original is None
            or not isinstance(embedded_query_set.original.vector, (list, tuple))
            or len(embedded_query_set.original.vector) == 0
        ):
            raise VectorRetrievalError(
                "Invalid EmbeddedQuerySet: original query embedding vector is missing or empty."
            )

        # 3. Operational Parameter Resolution (Explicit None Checks)
        effective_top_k = self._config.top_k if top_k is None else top_k
        if not isinstance(effective_top_k, int) or effective_top_k <= 0:
            raise VectorRetrievalError(
                f"top_k must be a positive integer, got {effective_top_k}."
            )

        effective_score_threshold = (
            self._config.score_threshold if score_threshold is None else score_threshold
        )
        if effective_score_threshold is not None:
            if not isinstance(effective_score_threshold, (int, float)) or not math.isfinite(
                effective_score_threshold
            ):
                raise VectorRetrievalError(
                    f"score_threshold must be a finite float or None, got {effective_score_threshold}."
                )

        # 4. Extract Query Representations Preserving Provenance
        query_reps: list[tuple[str, list[float]]] = []
        for eq in embedded_query_set.queries:
            if eq is not None and isinstance(eq.vector, (list, tuple)) and len(eq.vector) > 0:
                query_reps.append((eq.query_type or "original", list(eq.vector)))

        if not query_reps:
            raise VectorRetrievalError("No valid query vectors found in EmbeddedQuerySet.")

        query_types = [qr[0] for qr in query_reps]
        query_vectors = [qr[1] for qr in query_reps]

        # 5. Execute Vector Search via Batch API
        start_time = time.perf_counter()
        logger.debug(
            "Executing vector search for project '%s' (queries=%d, top_k=%d, score_threshold=%s)",
            clean_project_id,
            len(query_vectors),
            effective_top_k,
            effective_score_threshold,
        )

        try:
            batch_results = await self._vector_store.search_batch(
                query_vectors=query_vectors,
                project_id=clean_project_id,
                limit=effective_top_k,
                score_threshold=effective_score_threshold,
            )
        except VectorStoreError as exc:
            logger.error(
                "Vector store failure during search for project '%s': %s",
                clean_project_id,
                exc,
            )
            raise VectorRetrievalError(f"Vector search failed: {exc}", original_error=exc) from exc
        except Exception as exc:
            logger.error(
                "Unexpected error during vector search for project '%s': %s",
                clean_project_id,
                exc,
            )
            raise VectorRetrievalError(
                f"Unexpected error during vector search: {exc}", original_error=exc
            ) from exc

        # 6. Result Validation, Mapping, and Observability
        candidates: list[VectorSearchCandidate] = []
        discarded_count = 0

        for q_idx, results_for_query in enumerate(batch_results):
            q_type = query_types[q_idx]
            for item in results_for_query:
                payload = item.payload

                # Validate required identifiers
                if not payload.chunk_id or not payload.chunk_id.strip():
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed vector candidate: point_id=%s, reason='missing or empty chunk_id'",
                        item.id,
                    )
                    continue

                if not payload.document_id or not payload.document_id.strip():
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed vector candidate: point_id=%s, reason='missing or empty document_id'",
                        item.id,
                    )
                    continue

                # Defensive project isolation check
                if payload.project_id != clean_project_id:
                    discarded_count += 1
                    logger.warning(
                        "Discarded leaked vector candidate: point_id=%s, reason='project_id mismatch' "
                        "(expected='%s', actual='%s')",
                        item.id,
                        clean_project_id,
                        payload.project_id,
                    )
                    continue

                # Validate similarity score
                if not isinstance(item.score, (int, float)) or not math.isfinite(item.score):
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed vector candidate: point_id=%s, reason='non-finite score %s'",
                        item.id,
                        item.score,
                    )
                    continue

                doc_version = (
                    payload.document_version_id.strip()
                    if (payload.document_version_id and payload.document_version_id.strip())
                    else None
                )

                candidates.append(
                    VectorSearchCandidate(
                        chunk_id=payload.chunk_id.strip(),
                        document_id=payload.document_id.strip(),
                        project_id=clean_project_id,
                        score=float(item.score),
                        query_type=q_type,
                        document_version_id=doc_version,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Vector search completed: project_id='%s', queries=%d, returned=%d, discarded=%d, latency_ms=%.2f",
            clean_project_id,
            len(query_vectors),
            len(candidates),
            discarded_count,
            elapsed_ms,
        )

        return candidates


def get_vector_search_service(
    vector_store: Optional[BaseVectorStore] = None,
    config: Optional[VectorSearchConfig] = None,
) -> VectorSearchService:
    """Get or create singleton VectorSearchService instance.

    Args:
        vector_store: Optional BaseVectorStore instance override.
        config: Optional VectorSearchConfig instance override.

    Returns:
        VectorSearchService: Active vector search service.
    """
    global _vector_search_service
    if vector_store is not None or config is not None:
        return VectorSearchService(vector_store=vector_store, config=config)

    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
    return _vector_search_service


def reset_vector_search_service() -> None:
    """Reset cached singleton VectorSearchService instance. Useful for tests."""
    global _vector_search_service
    _vector_search_service = None


async def search_vectors(
    embedded_query_set: EmbeddedQuerySet,
    project_id: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    service: Optional[VectorSearchService] = None,
) -> list[VectorSearchCandidate]:
    """Convenience functional entrypoint to perform vector similarity search.

    Args:
        embedded_query_set: Validated EmbeddedQuerySet container from Query Embedding.
        project_id: Mandatory project/tenant ID strictly enforcing database-level isolation.
        top_k: Optional per-query result limit override.
        score_threshold: Optional similarity score threshold override.
        service: Optional VectorSearchService instance. Defaults to singleton.

    Returns:
        list[VectorSearchCandidate]: Ranked candidate matches.
    """
    active_service = service or get_vector_search_service()
    return await active_service.search(
        embedded_query_set=embedded_query_set,
        project_id=project_id,
        top_k=top_k,
        score_threshold=score_threshold,
    )
