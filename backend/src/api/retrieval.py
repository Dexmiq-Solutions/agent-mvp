"""FastAPI router for project-scoped RAG retrieval operations."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_project_service, get_rag_service_dependency
from db.session import get_db_session
from rag.retrieval.config import RetrievalConfig
from rag.retrieval.service import RAGService
from schemas.retrieval import (
    RetrievalAttemptMetadataSchema,
    RetrievalExecutionMetadataSchema,
    RetrievalRequestSchema,
    RetrievalResponseSchema,
    RetrievedChunkSchema,
)
from services.project_service import ProjectService

router = APIRouter(prefix="/projects/{project_id}/retrieval", tags=["retrieval"])


@router.post(
    "",
    response_model=RetrievalResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Execute project-isolated knowledge retrieval",
    description=(
        "Executes the full RAG retrieval pipeline (preprocessing, transformation, embedding, "
        "dense/sparse retrieval, fusion, filtering, reranking, hydration, context assembly, and formatting) "
        "strictly scoped to the target project."
    ),
)
async def retrieve_project_knowledge(
    project_id: str,
    request: RetrievalRequestSchema,
    project_service: ProjectService = Depends(get_project_service),
    rag_service: RAGService = Depends(get_rag_service_dependency),
    db: AsyncSession = Depends(get_db_session),
) -> RetrievalResponseSchema:
    """Execute project-scoped retrieval query and return structured project knowledge."""
    # 1. Enforce Project Isolation & Existence Verification
    await project_service.get_project_by_id(project_id)

    # 2. Build Operational Config Overrides
    default_cfg = rag_service.config
    eff_config = RetrievalConfig(
        top_k=request.top_k if request.top_k is not None else default_cfg.top_k,
        dense_top_k=request.dense_top_k if request.dense_top_k is not None else default_cfg.dense_top_k,
        sparse_top_k=request.sparse_top_k if request.sparse_top_k is not None else default_cfg.sparse_top_k,
        fusion_top_k=default_cfg.fusion_top_k,
        rerank_candidate_limit=default_cfg.rerank_candidate_limit,
        rerank_result_limit=request.top_k if request.top_k is not None else default_cfg.rerank_result_limit,
        enable_transformation=(
            request.enable_transformation
            if request.enable_transformation is not None
            else default_cfg.enable_transformation
        ),
        enable_sparse=(
            request.enable_sparse
            if request.enable_sparse is not None
            else default_cfg.enable_sparse
        ),
        enable_reranking=(
            request.enable_reranking
            if request.enable_reranking is not None
            else default_cfg.enable_reranking
        ),
        enable_relevance_check=(
            request.enable_relevance_check
            if request.enable_relevance_check is not None
            else default_cfg.enable_relevance_check
        ),
        max_attempts=default_cfg.max_attempts,
        metadata_filters=request.metadata_filters,
        use_contextual_enrichment=default_cfg.use_contextual_enrichment,
        score_threshold=default_cfg.score_threshold,
    )

    # 3. Execute Retrieval Orchestration
    result = await rag_service.retrieve(
        project_id=project_id,
        query=request.query,
        config=eff_config,
        session=db,
    )

    # 4. Map Domain Model to Response Schema
    chunk_schemas = [
        RetrievedChunkSchema(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_version_id=c.document_version_id,
            content=c.content,
            rank=c.rank,
            score=c.score,
            heading=c.heading,
            section_path=list(c.section_path),
            contextual_content=c.contextual_content,
            metadata=c.metadata,
        )
        for c in result.chunks
    ]

    attempt_schemas = [
        RetrievalAttemptMetadataSchema(
            attempt=a.attempt,
            query=a.query,
            is_transformed=a.is_transformed,
            transformed_query=a.transformed_query,
            strategy_used=a.strategy_used,
            dense_candidates_count=a.dense_candidates_count,
            sparse_candidates_count=a.sparse_candidates_count,
            fused_candidates_count=a.fused_candidates_count,
            filtered_candidates_count=a.filtered_candidates_count,
            reranked_candidates_count=a.reranked_candidates_count,
            hydrated_candidates_count=a.hydrated_candidates_count,
            is_relevant=a.is_relevant,
            relevance_reason=a.relevance_reason,
            latency_ms=a.latency_ms,
        )
        for a in result.execution_metadata.attempts
    ]

    execution_schema = RetrievalExecutionMetadataSchema(
        total_duration_ms=result.execution_metadata.total_duration_ms,
        attempts_count=result.execution_metadata.attempts_count,
        fallback_triggered=result.execution_metadata.fallback_triggered,
        attempts=attempt_schemas,
        stage_latencies_ms=result.execution_metadata.stage_latencies_ms,
    )

    return RetrievalResponseSchema(
        project_id=result.project_id,
        original_query=result.original_query,
        retrieval_query=result.retrieval_query,
        chunk_count=len(chunk_schemas),
        chunks=chunk_schemas,
        formatted_context=result.formatted_text,
        execution_metadata=execution_schema,
    )
