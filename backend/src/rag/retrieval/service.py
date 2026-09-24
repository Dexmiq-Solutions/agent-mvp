"""RAG Service orchestrating end-to-end project-scoped retrieval workflows."""

import asyncio
import time
from typing import Any, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.retrieval import (
    InvalidQueryError,
    ProjectBoundaryViolationError,
    RetrievalError,
)
from rag.retrieval.formatting import (
    ContextFormattingService,
    FormattedContext,
    get_context_formatting_service,
)
from rag.retrieval.assembly import (
    ContextAssemblyService,
    get_context_assembly_service,
)
from rag.retrieval.config import RetrievalConfig
from rag.retrieval.embedding import (
    QueryEmbeddingService,
    get_query_embedding_service,
)
from rag.retrieval.filtering import (
    MetadataFilteringService,
    get_metadata_filtering_service,
)
from rag.retrieval.fusion import (
    FusionService,
    get_fusion_service,
)
from rag.retrieval.hydration import (
    BaseChunkRepository,
    ChunkHydrationService,
    get_chunk_hydration_service,
    get_chunk_repository,
)
from rag.retrieval.keyword import (
    KeywordSearchService,
    get_keyword_search_service,
)
from rag.retrieval.models import (
    AssembledContext,
    EmbeddedQuerySet,
    FusedCandidate,
    HydratedCandidate,
    KeywordSearchCandidate,
    ProcessedQuery,
    RerankedCandidate,
    RetrievalAttemptMetadata,
    RetrievalExecutionMetadata,
    RetrievalQuerySet,
    RetrievalResult,
    RetrievedChunk,
    VectorSearchCandidate,
)
from rag.retrieval.preprocessing import (
    QueryPreprocessor,
    get_query_preprocessor,
)
from rag.retrieval.relevance import (
    BaseRelevanceChecker,
    get_relevance_checker,
)
from rag.retrieval.reranking import (
    RerankingService,
    get_reranking_service,
    resolve_candidate_text,
)
from rag.retrieval.transformation import (
    QueryTransformationService,
    get_query_transformation_service,
)
from rag.retrieval.vector import (
    VectorSearchService,
    get_vector_search_service,
)

logger = get_logger(__name__)

_rag_service: Optional["RAGService"] = None


class RAGService:
    """Application-facing orchestration service coordinating the modular RAG retrieval pipeline.

    Connects all specialized retrieval stages into one coherent, project-isolated workflow:
        User Query
            ↓
        Query Preprocessing
            ↓
        Optional Query Transformation
            ↓
        Query Embedding
            ↓
        Dense Retrieval + Sparse Retrieval (concurrent)
            ↓
        Hybrid Fusion (RRF)
            ↓
        Metadata Filtering
            ↓
        Cross-Encoder Reranking
            ↓
        Chunk Fetching / Hydration (batched from PostgreSQL)
            ↓
        Context Assembly
            ↓
        Optional Relevance Check / Bounded Fallback
            ↓
        Context Formatting
            ↓
        RetrievalResult
    """

    def __init__(
        self,
        query_preprocessor: Optional[QueryPreprocessor] = None,
        transformation_service: Optional[QueryTransformationService] = None,
        embedding_service: Optional[QueryEmbeddingService] = None,
        vector_search_service: Optional[VectorSearchService] = None,
        keyword_search_service: Optional[KeywordSearchService] = None,
        fusion_service: Optional[FusionService] = None,
        filtering_service: Optional[MetadataFilteringService] = None,
        reranking_service: Optional[RerankingService] = None,
        chunk_repository: Optional[BaseChunkRepository] = None,
        hydration_service: Optional[ChunkHydrationService] = None,
        assembly_service: Optional[ContextAssemblyService] = None,
        relevance_checker: Optional[BaseRelevanceChecker] = None,
        formatting_service: Optional[ContextFormattingService] = None,
        config: Optional[RetrievalConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize RAGService with modular retrieval stage dependencies.

        Args:
            query_preprocessor: Query preprocessor instance.
            transformation_service: Query transformation service instance.
            embedding_service: Query embedding service instance.
            vector_search_service: Dense vector search service instance.
            keyword_search_service: Sparse keyword search service instance.
            fusion_service: Hybrid fusion service instance.
            filtering_service: Metadata filtering service instance.
            reranking_service: Cross-encoder reranking service instance.
            chunk_repository: Chunk repository for content store lookups.
            hydration_service: Chunk hydration service instance.
            assembly_service: Context assembly service instance.
            relevance_checker: Relevance checker instance.
            formatting_service: Context formatting service instance.
            config: Default retrieval configuration.
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._config = config or RetrievalConfig.from_settings(self._settings)

        self._query_preprocessor = query_preprocessor or get_query_preprocessor(settings=self._settings)
        self._transformation_service = transformation_service or get_query_transformation_service()
        self._embedding_service = embedding_service or get_query_embedding_service(settings=self._settings)
        self._vector_search_service = vector_search_service or get_vector_search_service()
        self._keyword_search_service = keyword_search_service or get_keyword_search_service(settings=self._settings)
        self._fusion_service = fusion_service or get_fusion_service(settings=self._settings)
        self._filtering_service = filtering_service or get_metadata_filtering_service(settings=self._settings)
        self._reranking_service = reranking_service or get_reranking_service(settings=self._settings)
        self._chunk_repository = chunk_repository or get_chunk_repository()
        self._hydration_service = hydration_service or get_chunk_hydration_service(
            repository=self._chunk_repository,
            settings=self._settings,
        )
        self._assembly_service = assembly_service or get_context_assembly_service(settings=self._settings)
        self._relevance_checker = relevance_checker or get_relevance_checker(settings=self._settings)
        self._formatting_service = formatting_service or get_context_formatting_service(settings=self._settings)

    @property
    def config(self) -> RetrievalConfig:
        """Return active default retrieval configuration."""
        return self._config

    async def retrieve(
        self,
        project_id: str,
        query: str,
        config: Optional[RetrievalConfig] = None,
        session: Optional[AsyncSession] = None,
    ) -> RetrievalResult:
        """Execute the project-isolated RAG retrieval pipeline for a user query.

        Args:
            project_id: Mandatory project identifier strictly enforcing tenant isolation.
            query: Raw user query string.
            config: Optional per-request RetrievalConfig overrides.
            session: Optional active AsyncSession for PostgreSQL Content Store hydration.

        Returns:
            RetrievalResult: Structured retrieval outcome containing authoritative chunks,
                assembled and formatted context, and execution observability metadata.

        Raises:
            ProjectBoundaryViolationError: If project_id is invalid or missing.
            InvalidQueryError: If query is invalid or empty.
            RetrievalError: If pipeline execution fails.
        """
        overall_start_time = time.perf_counter()

        # 1. Validate Scope & Project Isolation
        if not isinstance(project_id, str) or not project_id.strip():
            raise ProjectBoundaryViolationError(
                "project_id must be a non-empty string to enforce project isolation."
            )
        clean_project_id = project_id.strip()

        # 2. Resolve Operational Configuration
        eff_config = config or self._config

        logger.info(
            "Starting RAG retrieval: project_id='%s', top_k=%d, sparse=%s, reranking=%s, relevance_check=%s",
            clean_project_id,
            eff_config.top_k,
            eff_config.enable_sparse,
            eff_config.enable_reranking,
            eff_config.enable_relevance_check,
        )

        # 3. Stage 1: Query Preprocessing
        t0 = time.perf_counter()
        processed_query: ProcessedQuery = self._query_preprocessor.preprocess(query)
        preprocessing_latency = (time.perf_counter() - t0) * 1000.0

        attempts_metadata: list[RetrievalAttemptMetadata] = []
        final_context: Optional[AssembledContext] = None
        final_hydrated_candidates: list[HydratedCandidate] = []
        final_retrieval_query: str = processed_query.processed_query

        max_attempts = max(1, eff_config.max_attempts)
        overall_stage_latencies: dict[str, float] = {
            "preprocessing_ms": preprocessing_latency,
        }

        # 4. Attempt Loop: Support Bounded Fallback
        for attempt in range(1, max_attempts + 1):
            attempt_start_time = time.perf_counter()
            stage_latencies: dict[str, float] = {}
            is_fallback_attempt = attempt > 1

            logger.debug(
                "Retrieval attempt %d/%d (fallback=%s) for project '%s'",
                attempt,
                max_attempts,
                is_fallback_attempt,
                clean_project_id,
            )

            # --- Stage 2: Query Transformation ---
            t0 = time.perf_counter()
            if eff_config.enable_transformation:
                query_set: RetrievalQuerySet = await self._transformation_service.transform(
                    query=processed_query,
                    is_fallback=is_fallback_attempt,
                    attempt=attempt,
                    project_id=clean_project_id,
                )
            else:
                query_set = RetrievalQuerySet(
                    original_query=processed_query.processed_query,
                    is_transformed=False,
                )
            stage_latencies["transformation_ms"] = (time.perf_counter() - t0) * 1000.0

            # Update active retrieval query for this attempt
            if query_set.has_transformed_query and query_set.transformed_query:
                current_retrieval_query = query_set.transformed_query
            else:
                current_retrieval_query = query_set.original_query
            final_retrieval_query = current_retrieval_query

            # --- Stage 3: Query Embedding ---
            t0 = time.perf_counter()
            embedded_query_set: EmbeddedQuerySet = await self._embedding_service.embed_query_set(query_set)
            stage_latencies["embedding_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 4: Dense + Sparse Retrieval (Concurrent Execution) ---
            t0 = time.perf_counter()
            dense_coro = self._vector_search_service.search(
                embedded_query_set=embedded_query_set,
                project_id=clean_project_id,
                top_k=eff_config.dense_top_k,
                score_threshold=eff_config.score_threshold,
            )

            if eff_config.enable_sparse:
                sparse_coro = self._keyword_search_service.search(
                    retrieval_query_set=query_set,
                    project_id=clean_project_id,
                    top_k=eff_config.sparse_top_k,
                    score_threshold=eff_config.score_threshold,
                )
                dense_candidates, sparse_candidates = await asyncio.gather(dense_coro, sparse_coro)
            else:
                dense_candidates = await dense_coro
                sparse_candidates = []

            stage_latencies["search_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 5: Hybrid Fusion (RRF) ---
            t0 = time.perf_counter()
            fused_candidates: list[FusedCandidate] = self._fusion_service.fuse(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                project_id=clean_project_id,
                top_k=eff_config.fusion_top_k,
            )
            stage_latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 6: Metadata Filtering ---
            t0 = time.perf_counter()
            filtered_candidates: list[FusedCandidate] = self._filtering_service.filter(
                candidates=fused_candidates,
                filter_spec=eff_config.metadata_filters,
                project_id=clean_project_id,
            )
            stage_latencies["filtering_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 7: Cross-Encoder Reranking ---
            t0 = time.perf_counter()
            effective_result_limit = min(eff_config.top_k, eff_config.rerank_result_limit)
            if eff_config.enable_reranking and filtered_candidates:
                # Bound candidates to candidate_limit before text resolution & reranking
                candidates_to_rerank = filtered_candidates[: eff_config.rerank_candidate_limit]

                # Resolve chunk texts for bounded candidates if not in candidate metadata
                needs_content = any(resolve_candidate_text(c) is None for c in candidates_to_rerank)
                content_lookup: Optional[dict[str, str]] = None

                if needs_content:
                    cids = [c.chunk_id for c in candidates_to_rerank]
                    records = await self._chunk_repository.fetch_chunks(
                        project_id=clean_project_id,
                        chunk_ids=cids,
                        session=session,
                    )
                    content_lookup = {cid: r.content for cid, r in records.items()}

                reranked_candidates = await self._reranking_service.rerank(
                    query=current_retrieval_query,
                    candidates=candidates_to_rerank,
                    project_id=clean_project_id,
                    content_lookup=content_lookup,
                    candidate_limit=eff_config.rerank_candidate_limit,
                    result_limit=effective_result_limit,
                )
            else:
                # If reranking disabled, slice filtered candidates
                reranked_candidates = filtered_candidates[:effective_result_limit]
            stage_latencies["reranking_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 8: Chunk Fetching / Hydration ---
            t0 = time.perf_counter()
            hydrated_candidates: list[HydratedCandidate] = await self._hydration_service.hydrate(
                candidates=reranked_candidates,
                project_id=clean_project_id,
                session=session,
            )
            stage_latencies["hydration_ms"] = (time.perf_counter() - t0) * 1000.0

            # --- Stage 9: Context Assembly ---
            t0 = time.perf_counter()
            assembled_context: AssembledContext = self._assembly_service.assemble(
                candidates=hydrated_candidates,
                project_id=clean_project_id,
                query=processed_query.original_query,
            )
            stage_latencies["assembly_ms"] = (time.perf_counter() - t0) * 1000.0

            attempt_latency = (time.perf_counter() - attempt_start_time) * 1000.0

            # --- Stage 10: Optional Relevance Check / Fallback ---
            is_relevant: Optional[bool] = None
            relevance_reason: Optional[str] = None

            if eff_config.enable_relevance_check:
                t0 = time.perf_counter()
                is_rel, reason = await self._relevance_checker.check_relevance(
                    query=processed_query.original_query,
                    context=assembled_context,
                )
                stage_latencies["relevance_check_ms"] = (time.perf_counter() - t0) * 1000.0
                is_relevant = is_rel
                relevance_reason = reason

                attempt_meta = RetrievalAttemptMetadata(
                    attempt=attempt,
                    query=current_retrieval_query,
                    is_transformed=query_set.is_transformed,
                    transformed_query=query_set.transformed_query,
                    strategy_used=query_set.strategy_used,
                    dense_candidates_count=len(dense_candidates),
                    sparse_candidates_count=len(sparse_candidates),
                    fused_candidates_count=len(fused_candidates),
                    filtered_candidates_count=len(filtered_candidates),
                    reranked_candidates_count=len(reranked_candidates),
                    hydrated_candidates_count=len(hydrated_candidates),
                    is_relevant=is_relevant,
                    relevance_reason=relevance_reason,
                    latency_ms=attempt_latency,
                    stage_latencies_ms=stage_latencies,
                )
                attempts_metadata.append(attempt_meta)

                if not is_relevant and attempt < max_attempts:
                    logger.info(
                        "Attempt %d context evaluated as not relevant ('%s'). Triggering fallback attempt %d.",
                        attempt,
                        relevance_reason,
                        attempt + 1,
                    )
                    continue

            else:
                attempt_meta = RetrievalAttemptMetadata(
                    attempt=attempt,
                    query=current_retrieval_query,
                    is_transformed=query_set.is_transformed,
                    transformed_query=query_set.transformed_query,
                    strategy_used=query_set.strategy_used,
                    dense_candidates_count=len(dense_candidates),
                    sparse_candidates_count=len(sparse_candidates),
                    fused_candidates_count=len(fused_candidates),
                    filtered_candidates_count=len(filtered_candidates),
                    reranked_candidates_count=len(reranked_candidates),
                    hydrated_candidates_count=len(hydrated_candidates),
                    is_relevant=None,
                    relevance_reason=None,
                    latency_ms=attempt_latency,
                    stage_latencies_ms=stage_latencies,
                )
                attempts_metadata.append(attempt_meta)

            # Accepted context found; break out of loop
            final_context = assembled_context
            final_hydrated_candidates = hydrated_candidates
            overall_stage_latencies.update(stage_latencies)
            break

        # If loop completed without context assignment (should not occur)
        if final_context is None:
            final_context = assembled_context
            final_hydrated_candidates = hydrated_candidates

        # --- Stage 11: Context Formatting ---
        t0 = time.perf_counter()
        formatted_context: FormattedContext = self._formatting_service.format(final_context)
        overall_stage_latencies["formatting_ms"] = (time.perf_counter() - t0) * 1000.0

        total_duration = (time.perf_counter() - overall_start_time) * 1000.0

        # --- Build Application-Facing Result Contract ---
        retrieved_chunks = tuple(
            RetrievedChunk.from_hydrated_candidate(hc) for hc in final_hydrated_candidates
        )

        execution_metadata = RetrievalExecutionMetadata(
            total_duration_ms=total_duration,
            attempts_count=len(attempts_metadata),
            fallback_triggered=len(attempts_metadata) > 1,
            attempts=tuple(attempts_metadata),
            stage_latencies_ms=overall_stage_latencies,
        )

        result = RetrievalResult(
            project_id=clean_project_id,
            original_query=processed_query.original_query,
            retrieval_query=final_retrieval_query,
            chunks=retrieved_chunks,
            assembled_context=final_context,
            formatted_context=formatted_context,
            execution_metadata=execution_metadata,
        )

        logger.info(
            "RAG retrieval completed: project_id='%s', chunks=%d, attempts=%d, fallback=%s, duration_ms=%.2f",
            clean_project_id,
            len(result),
            len(attempts_metadata),
            len(attempts_metadata) > 1,
            total_duration,
        )

        return result


RAGRetrievalService = RAGService


def get_rag_service(
    query_preprocessor: Optional[QueryPreprocessor] = None,
    transformation_service: Optional[QueryTransformationService] = None,
    embedding_service: Optional[QueryEmbeddingService] = None,
    vector_search_service: Optional[VectorSearchService] = None,
    keyword_search_service: Optional[KeywordSearchService] = None,
    fusion_service: Optional[FusionService] = None,
    filtering_service: Optional[MetadataFilteringService] = None,
    reranking_service: Optional[RerankingService] = None,
    chunk_repository: Optional[BaseChunkRepository] = None,
    hydration_service: Optional[ChunkHydrationService] = None,
    assembly_service: Optional[ContextAssemblyService] = None,
    relevance_checker: Optional[BaseRelevanceChecker] = None,
    formatting_service: Optional[ContextFormattingService] = None,
    config: Optional[RetrievalConfig] = None,
    settings: Optional[Settings] = None,
) -> RAGService:
    """Retrieve or initialize the singleton RAGService instance.

    Args:
        query_preprocessor: Optional query preprocessor override.
        transformation_service: Optional transformation service override.
        embedding_service: Optional query embedding service override.
        vector_search_service: Optional vector search service override.
        keyword_search_service: Optional keyword search service override.
        fusion_service: Optional fusion service override.
        filtering_service: Optional filtering service override.
        reranking_service: Optional reranking service override.
        chunk_repository: Optional chunk repository override.
        hydration_service: Optional hydration service override.
        assembly_service: Optional context assembly service override.
        relevance_checker: Optional relevance checker override.
        formatting_service: Optional context formatting service override.
        config: Optional retrieval configuration override.
        settings: Optional application settings override.

    Returns:
        RAGService: Configured RAG retrieval service instance.
    """
    global _rag_service
    has_overrides = any(
        arg is not None
        for arg in (
            query_preprocessor,
            transformation_service,
            embedding_service,
            vector_search_service,
            keyword_search_service,
            fusion_service,
            filtering_service,
            reranking_service,
            chunk_repository,
            hydration_service,
            assembly_service,
            relevance_checker,
            formatting_service,
            config,
            settings,
        )
    )

    if _rag_service is None or has_overrides:
        service = RAGService(
            query_preprocessor=query_preprocessor,
            transformation_service=transformation_service,
            embedding_service=embedding_service,
            vector_search_service=vector_search_service,
            keyword_search_service=keyword_search_service,
            fusion_service=fusion_service,
            filtering_service=filtering_service,
            reranking_service=reranking_service,
            chunk_repository=chunk_repository,
            hydration_service=hydration_service,
            assembly_service=assembly_service,
            relevance_checker=relevance_checker,
            formatting_service=formatting_service,
            config=config,
            settings=settings,
        )
        if not has_overrides:
            _rag_service = service
        return service

    return _rag_service


def reset_rag_service() -> None:
    """Reset the cached singleton RAGService instance (primarily for testing)."""
    global _rag_service
    _rag_service = None


async def retrieve(
    project_id: str,
    query: str,
    config: Optional[RetrievalConfig] = None,
    session: Optional[AsyncSession] = None,
) -> RetrievalResult:
    """Convenience functional entrypoint executing RAG retrieval via the default service.

    Args:
        project_id: Tenant/project ID enforcing isolation boundary.
        query: User query string.
        config: Optional RetrievalConfig overrides.
        session: Optional AsyncSession for database chunk hydration.

    Returns:
        RetrievalResult: Complete structured retrieval outcome.
    """
    service = get_rag_service()
    return await service.retrieve(
        project_id=project_id,
        query=query,
        config=config,
        session=session,
    )
