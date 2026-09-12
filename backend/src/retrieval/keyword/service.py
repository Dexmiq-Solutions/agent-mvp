"""Keyword Search service executing project-scoped sparse/lexical retrieval against Qdrant."""

import math
import time
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import KeywordRetrievalError, SparseEncodingError
from exceptions.vector import VectorStoreError
from retrieval.config import KeywordSearchConfig
from retrieval.keyword.encoder import get_sparse_encoder
from retrieval.keyword.encoder.base import BaseSparseEncoder
from retrieval.models import KeywordSearchCandidate, RetrievalQuerySet, SparseVector
from storage.vector import BaseVectorStore, get_vector_store

logger = get_logger(__name__)

_keyword_search_service: Optional["KeywordSearchService"] = None


class KeywordSearchService:
    """Service responsible for project-scoped lexical/sparse search against vector storage.

    Consumes RetrievalQuerySet representations produced by Query Transformation, encodes
    query representations into sparse vectors via BaseSparseEncoder, executes project-isolated
    sparse retrieval via BaseVectorStore, validates candidate records, preserves query
    representation provenance, and returns ranked KeywordSearchCandidate instances for downstream
    fusion and reranking.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        sparse_encoder: Optional[BaseSparseEncoder] = None,
        config: Optional[KeywordSearchConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize KeywordSearchService.

        Args:
            vector_store: Pre-configured BaseVectorStore instance. Defaults to default vector store.
            sparse_encoder: BaseSparseEncoder instance. Defaults to encoder configured via settings.
            config: KeywordSearchConfig instance. Defaults to config loaded from application Settings.
            settings: Application Settings instance. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or KeywordSearchConfig.from_settings(self._settings)
        self._vector_store = vector_store or get_vector_store(
            collection_name=getattr(self._settings, "QDRANT_COLLECTION_NAME", "rag_documents"),
            settings=self._settings,
        )
        self._sparse_encoder = sparse_encoder or get_sparse_encoder(
            strategy=self._config.encoder_strategy,
            settings=self._settings,
        )

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the active vector store repository."""
        return self._vector_store

    @property
    def sparse_encoder(self) -> BaseSparseEncoder:
        """Return the active sparse encoder."""
        return self._sparse_encoder

    @property
    def config(self) -> KeywordSearchConfig:
        """Return the active keyword search configuration."""
        return self._config

    async def search(
        self,
        retrieval_query_set: RetrievalQuerySet,
        project_id: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[KeywordSearchCandidate]:
        """Perform project-scoped keyword/sparse search for all query representations in the query set.

        Args:
            retrieval_query_set: Validated RetrievalQuerySet container from Query Transformation.
            project_id: Mandatory project/tenant ID strictly enforcing database-level isolation.
            top_k: Optional per-query result limit override. Defaults to config top_k.
            score_threshold: Optional similarity score threshold override. Defaults to config threshold.

        Returns:
            list[KeywordSearchCandidate]: Ranked candidate matches preserving query provenance.

        Raises:
            KeywordRetrievalError: If inputs are invalid, encoding fails, or vector store fails.
        """
        # 1. Project Isolation Validation
        if not isinstance(project_id, str) or not project_id.strip():
            raise KeywordRetrievalError(
                "project_id must be a non-empty string to enforce project isolation."
            )
        clean_project_id = project_id.strip()

        # 2. Input Structure Validation
        if not isinstance(retrieval_query_set, RetrievalQuerySet):
            raise KeywordRetrievalError(
                f"Expected RetrievalQuerySet instance, got '{type(retrieval_query_set).__name__}'."
            )
        if (
            retrieval_query_set.original_query is None
            or not isinstance(retrieval_query_set.original_query, str)
            or not retrieval_query_set.original_query.strip()
        ):
            raise KeywordRetrievalError(
                "Invalid RetrievalQuerySet: original_query is missing or empty."
            )

        # 3. Operational Parameter Resolution (Explicit None Checks)
        effective_top_k = self._config.top_k if top_k is None else top_k
        if not isinstance(effective_top_k, int) or effective_top_k <= 0:
            raise KeywordRetrievalError(
                f"top_k must be a positive integer, got {effective_top_k}."
            )

        effective_score_threshold = (
            self._config.score_threshold if score_threshold is None else score_threshold
        )
        if effective_score_threshold is not None:
            if not isinstance(effective_score_threshold, (int, float)) or not math.isfinite(
                effective_score_threshold
            ):
                raise KeywordRetrievalError(
                    f"score_threshold must be a finite float or None, got {effective_score_threshold}."
                )

        # 4. Extract Query Representations Preserving Provenance
        query_reps: list[tuple[str, str]] = [
            ("original", retrieval_query_set.original_query.strip())
        ]
        if retrieval_query_set.has_transformed_query and retrieval_query_set.transformed_query:
            query_reps.append(("transformed", retrieval_query_set.transformed_query.strip()))

        query_types = [qr[0] for qr in query_reps]
        query_texts = [qr[1] for qr in query_reps]

        # 5. Sparse Encode Queries
        try:
            sparse_queries: list[SparseVector] = self._sparse_encoder.encode_queries(query_texts)
        except SparseEncodingError as exc:
            logger.error("Sparse encoding error for queries: %s", exc)
            raise KeywordRetrievalError(f"Failed to encode query text: {exc}", original_error=exc) from exc
        except Exception as exc:
            logger.error("Unexpected error during sparse query encoding: %s", exc)
            raise KeywordRetrievalError(
                f"Unexpected error during sparse query encoding: {exc}", original_error=exc
            ) from exc

        # 6. Execute Sparse Search via Batch API
        start_time = time.perf_counter()
        logger.debug(
            "Executing keyword search for project '%s' (queries=%d, top_k=%d, score_threshold=%s)",
            clean_project_id,
            len(sparse_queries),
            effective_top_k,
            effective_score_threshold,
        )

        try:
            batch_results = await self._vector_store.search_sparse_batch(
                query_sparse_vectors=sparse_queries,
                project_id=clean_project_id,
                limit=effective_top_k,
                score_threshold=effective_score_threshold,
                vector_name=self._config.sparse_vector_name,
            )
        except VectorStoreError as exc:
            logger.error(
                "Vector store failure during sparse search for project '%s': %s",
                clean_project_id,
                exc,
            )
            raise KeywordRetrievalError(f"Keyword search failed: {exc}", original_error=exc) from exc
        except Exception as exc:
            logger.error(
                "Unexpected error during sparse search for project '%s': %s",
                clean_project_id,
                exc,
            )
            raise KeywordRetrievalError(
                f"Unexpected error during keyword search: {exc}", original_error=exc
            ) from exc

        # 7. Result Validation, Mapping, and Observability
        candidates: list[KeywordSearchCandidate] = []
        discarded_count = 0

        for q_idx, results_for_query in enumerate(batch_results):
            q_type = query_types[q_idx]
            for item in results_for_query:
                payload = item.payload

                # Validate required identifiers
                if not payload.chunk_id or not payload.chunk_id.strip():
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed keyword candidate: point_id=%s, reason='missing or empty chunk_id'",
                        item.id,
                    )
                    continue

                if not payload.document_id or not payload.document_id.strip():
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed keyword candidate: point_id=%s, reason='missing or empty document_id'",
                        item.id,
                    )
                    continue

                # Defensive project isolation check
                if payload.project_id != clean_project_id:
                    discarded_count += 1
                    logger.warning(
                        "Discarded leaked keyword candidate: point_id=%s, reason='project_id mismatch' "
                        "(expected='%s', actual='%s')",
                        item.id,
                        clean_project_id,
                        payload.project_id,
                    )
                    continue

                # Validate score
                if not isinstance(item.score, (int, float)) or not math.isfinite(item.score):
                    discarded_count += 1
                    logger.warning(
                        "Discarded malformed keyword candidate: point_id=%s, reason='non-finite score %s'",
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
                    KeywordSearchCandidate(
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
            "Keyword search completed: project_id='%s', queries=%d, returned=%d, discarded=%d, latency_ms=%.2f",
            clean_project_id,
            len(sparse_queries),
            len(candidates),
            discarded_count,
            elapsed_ms,
        )

        return candidates


def get_keyword_search_service(
    vector_store: Optional[BaseVectorStore] = None,
    sparse_encoder: Optional[BaseSparseEncoder] = None,
    config: Optional[KeywordSearchConfig] = None,
    settings: Optional[Settings] = None,
) -> KeywordSearchService:
    """Retrieve or initialize the singleton KeywordSearchService instance.

    Args:
        vector_store: Optional BaseVectorStore dependency override.
        sparse_encoder: Optional BaseSparseEncoder dependency override.
        config: Optional KeywordSearchConfig override.
        settings: Optional Settings override.

    Returns:
        KeywordSearchService: Configured keyword search service instance.
    """
    global _keyword_search_service
    if _keyword_search_service is None or any(
        arg is not None for arg in (vector_store, sparse_encoder, config, settings)
    ):
        service = KeywordSearchService(
            vector_store=vector_store,
            sparse_encoder=sparse_encoder,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (vector_store, sparse_encoder, config, settings)):
            _keyword_search_service = service
        return service
    return _keyword_search_service


def reset_keyword_search_service() -> None:
    """Reset the singleton KeywordSearchService instance (primarily for testing)."""
    global _keyword_search_service
    _keyword_search_service = None


async def search_keywords(
    retrieval_query_set: RetrievalQuerySet,
    project_id: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    service: Optional[KeywordSearchService] = None,
) -> list[KeywordSearchCandidate]:
    """Convenience function to execute project-isolated keyword search on a RetrievalQuerySet.

    Args:
        retrieval_query_set: Validated RetrievalQuerySet container from Query Transformation.
        project_id: Mandatory project/tenant ID strictly enforcing database-level isolation.
        top_k: Optional per-query result limit override.
        score_threshold: Optional similarity score threshold override.
        service: Optional KeywordSearchService instance override.

    Returns:
        list[KeywordSearchCandidate]: Scored and validated keyword search candidates.
    """
    active_service = service or get_keyword_search_service()
    return await active_service.search(
        retrieval_query_set=retrieval_query_set,
        project_id=project_id,
        top_k=top_k,
        score_threshold=score_threshold,
    )
