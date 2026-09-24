"""Comprehensive test suite for RAG Service and Retrieval Orchestration."""

import asyncio
from typing import Any, Mapping, Optional, Sequence
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    ProjectBoundaryViolationError,
    QueryEmbeddingError,
    RetrievalError,
    VectorRetrievalError,
)
from models.chunk import ChunkModel
from rag.retrieval.formatting import FormattedContext, FormattedContextItem
from rag.retrieval.config import RetrievalConfig
from rag.retrieval.hydration.repository import BaseChunkRepository
from rag.retrieval.models import (
    AssembledContext,
    AssembledContextItem,
    EmbeddedQuery,
    EmbeddedQuerySet,
    FusedCandidate,
    HydratedCandidate,
    KeywordSearchCandidate,
    ProcessedQuery,
    RerankedCandidate,
    RetrievalQuerySet,
    RetrievalResult,
    RetrievedChunk,
    VectorSearchCandidate,
)
from rag.retrieval.relevance import BaseRelevanceChecker, HeuristicRelevanceChecker
from rag.retrieval.reranking.base import BaseReranker, ScoredDocument
from rag.retrieval.service import (
    RAGService,
    get_rag_service,
    reset_rag_service,
    retrieve,
)

# ------------------------------------------------------------------------------
# Test Fakes & Doubles
# ------------------------------------------------------------------------------


class FakeChunkRepository(BaseChunkRepository):
    """In-memory chunk repository for hydration testing."""

    def __init__(self, chunks: dict[str, ChunkModel] | None = None) -> None:
        self.chunks = chunks or {}
        self.fetch_calls: list[tuple[str, list[str]]] = []

    async def fetch_chunks(
        self,
        project_id: str,
        chunk_ids: Sequence[str],
        session: Any = None,
    ) -> dict[str, ChunkModel]:
        self.fetch_calls.append((project_id, list(chunk_ids)))
        return {
            cid: self.chunks[cid]
            for cid in chunk_ids
            if cid in self.chunks and self.chunks[cid].project_id == project_id
        }


class FakeReranker(BaseReranker):
    """Deterministic fake cross-encoder reranker."""

    def __init__(self, scores_by_text: dict[str, float] | None = None) -> None:
        self.scores_by_text = scores_by_text or {}
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return "test-reranker-2.5"

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        self.calls.append((query, list(documents)))
        scored = []
        for idx, doc in enumerate(documents):
            score = self.scores_by_text.get(doc, 0.5 - (idx * 0.05))
            scored.append(ScoredDocument(index=idx, score=score))
        # Sort descending by score
        scored.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored


class FakeRelevanceChecker(BaseRelevanceChecker):
    """Programmable relevance checker double."""

    def __init__(self, decisions: list[tuple[bool, str]] | None = None) -> None:
        self.decisions = list(decisions or [(True, "Relevant")])
        self.calls: list[tuple[str, AssembledContext]] = []

    async def check_relevance(
        self,
        query: str,
        context: AssembledContext,
    ) -> tuple[bool, str]:
        self.calls.append((query, context))
        if self.decisions:
            return self.decisions.pop(0)
        return True, "Default relevant"


def _make_chunk_record(
    chunk_id: str,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str = "ver-1",
    content: str = "Authoritative text",
    heading: str | None = "Section Header",
    section_path: list[str] | None = None,
    chunk_index: int = 0,
) -> ChunkModel:
    """Helper to construct ChunkModel records."""
    model = ChunkModel(
        chunk_id=chunk_id,
        project_id=project_id,
        document_id=document_id,
        document_version_id=document_version_id,
        content=content,
        heading=heading,
        section_path=section_path or ["Doc", "Section Header"],
        chunk_index=chunk_index,
        chunk_metadata={"source": "manual.pdf"},
    )
    return model


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_rag_singleton():
    """Reset the RAGService singleton before and after each test."""
    reset_rag_service()
    yield
    reset_rag_service()


@pytest.fixture
def mock_retrieval_environment():
    """Create a fully connected mock retrieval environment for RAGService testing."""
    records = {
        "c-1": _make_chunk_record("c-1", content="Chunk 1 content regarding deployment architecture"),
        "c-2": _make_chunk_record("c-2", content="Chunk 2 content on database sharding and pooling"),
        "c-3": _make_chunk_record("c-3", content="Chunk 3 content about authentication and tokens"),
    }
    repo = FakeChunkRepository(records)

    # Mock VectorSearchService
    vector_service = MagicMock()
    vector_service.search = AsyncMock(
        return_value=[
            VectorSearchCandidate("c-1", "doc-1", "proj-1", score=0.85),
            VectorSearchCandidate("c-2", "doc-1", "proj-1", score=0.75),
        ]
    )

    # Mock KeywordSearchService
    keyword_service = MagicMock()
    keyword_service.search = AsyncMock(
        return_value=[
            KeywordSearchCandidate("c-2", "doc-1", "proj-1", score=4.2),
            KeywordSearchCandidate("c-3", "doc-1", "proj-1", score=3.1),
        ]
    )

    # Mock QueryEmbeddingService
    embedding_service = MagicMock()
    embedding_service.embed_query_set = AsyncMock(
        return_value=EmbeddedQuerySet(
            original=EmbeddedQuery(query="deployment", vector=[0.1] * 1024, model="voyage-4")
        )
    )

    # Mock QueryTransformationService
    transformation_service = MagicMock()
    transformation_service.transform = AsyncMock(
        side_effect=lambda query, is_fallback=False, attempt=1, project_id=None: RetrievalQuerySet(
            original_query=query.processed_query if hasattr(query, "processed_query") else str(query),
            transformed_query=f"{query.processed_query if hasattr(query, 'processed_query') else str(query)} expanded" if is_fallback else None,
            is_transformed=is_fallback,
            strategy_used="llm_rewrite" if is_fallback else None,
        )
    )

    fake_reranker = FakeReranker(
        scores_by_text={
            "Chunk 1 content regarding deployment architecture": 0.95,
            "Chunk 2 content on database sharding and pooling": 0.88,
            "Chunk 3 content about authentication and tokens": 0.65,
        }
    )

    service = RAGService(
        chunk_repository=repo,
        vector_search_service=vector_service,
        keyword_search_service=keyword_service,
        embedding_service=embedding_service,
        transformation_service=transformation_service,
        reranking_service=None,  # will be initialized with settings/fake reranker below if passed
    )
    # Inject fake reranker into reranking service
    service._reranking_service._reranker = fake_reranker

    return {
        "service": service,
        "repo": repo,
        "vector_service": vector_service,
        "keyword_service": keyword_service,
        "embedding_service": embedding_service,
        "transformation_service": transformation_service,
        "reranker": fake_reranker,
    }


# ==============================================================================
# 1. End-to-End Retrieval Flow
# ==============================================================================


@pytest.mark.asyncio
async def test_end_to_end_retrieval_success(mock_retrieval_environment):
    """Verify complete end-to-end pipeline execution produces structured RetrievalResult."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="How is deployment configured?",
        config=RetrievalConfig(top_k=5, enable_sparse=True, enable_reranking=True),
    )

    assert isinstance(result, RetrievalResult)
    assert result.project_id == "proj-1"
    assert result.original_query == "How is deployment configured?"
    assert result.retrieval_query == "How is deployment configured?"
    assert not result.is_empty
    assert len(result.chunks) > 0

    # Chunks ordering dictated by reranker
    top_chunk: RetrievedChunk = result.chunks[0]
    assert top_chunk.chunk_id == "c-1"
    assert top_chunk.rank == 1
    assert top_chunk.score == 0.95
    assert "deployment architecture" in top_chunk.content
    assert top_chunk.section_path == ("Doc", "Section Header")

    # Formatted context checks
    assert isinstance(result.formatted_context, FormattedContext)
    assert "RETRIEVED CONTEXT" in result.formatted_text
    assert "deployment architecture" in result.formatted_text

    # Execution telemetry checks
    meta = result.execution_metadata
    assert meta.total_duration_ms > 0
    assert meta.attempts_count == 1
    assert not meta.fallback_triggered
    assert len(meta.attempts) == 1
    assert meta.attempts[0].fused_candidates_count >= 2


# ==============================================================================
# 2. Project Boundary Isolation Enforcement
# ==============================================================================


@pytest.mark.asyncio
async def test_project_boundary_validation():
    """Empty or invalid project IDs raise ProjectBoundaryViolationError."""
    service = RAGService()

    with pytest.raises(ProjectBoundaryViolationError):
        await service.retrieve(project_id="", query="Valid query")

    with pytest.raises(ProjectBoundaryViolationError):
        await service.retrieve(project_id="   ", query="Valid query")


@pytest.mark.asyncio
async def test_cross_tenant_chunks_defensively_isolated(mock_retrieval_environment):
    """Chunks belonging to another project are discarded during fusion, filtering, and hydration."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    # Infiltrate vector search results with a candidate from proj-2
    env["vector_service"].search = AsyncMock(
        return_value=[
            VectorSearchCandidate("c-1", "doc-1", "proj-1", score=0.9),
            VectorSearchCandidate("c-leak", "doc-leak", "proj-2", score=0.99),  # LEAK!
        ]
    )

    result = await service.retrieve(
        project_id="proj-1",
        query="Test cross-tenant leak prevention",
    )

    chunk_ids = [c.chunk_id for c in result.chunks]
    assert "c-leak" not in chunk_ids
    assert all(c.project_id == "proj-1" for c in result.chunks)


# ==============================================================================
# 3. Dense-Only vs Hybrid Retrieval
# ==============================================================================


@pytest.mark.asyncio
async def test_dense_only_retrieval_when_sparse_disabled(mock_retrieval_environment):
    """When enable_sparse=False, keyword search is completely bypassed."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="Dense search only test",
        config=RetrievalConfig(enable_sparse=False),
    )

    env["keyword_service"].search.assert_not_called()
    assert not result.is_empty
    attempt_meta = result.execution_metadata.attempts[0]
    assert attempt_meta.sparse_candidates_count == 0
    assert attempt_meta.dense_candidates_count > 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_fuses_both_branches(mock_retrieval_environment):
    """When enable_sparse=True, candidates from both branches are collected concurrently and fused."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="Hybrid search test",
        config=RetrievalConfig(enable_sparse=True),
    )

    env["vector_service"].search.assert_called_once()
    env["keyword_service"].search.assert_called_once()
    attempt_meta = result.execution_metadata.attempts[0]
    assert attempt_meta.dense_candidates_count > 0
    assert attempt_meta.sparse_candidates_count > 0


# ==============================================================================
# 4. Metadata Filtering Integration
# ==============================================================================


@pytest.mark.asyncio
async def test_metadata_filtering_applied(mock_retrieval_environment):
    """Metadata filters are passed to filtering service and filter non-matching candidates."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    # Attach metadata to candidates in vector/keyword search
    env["vector_service"].search = AsyncMock(
        return_value=[
            VectorSearchCandidate("c-1", "doc-1", "proj-1", score=0.9, metadata={"author": "Alice"}),
            VectorSearchCandidate("c-2", "doc-1", "proj-1", score=0.8, metadata={"author": "Bob"}),
        ]
    )
    env["keyword_service"].search = AsyncMock(return_value=[])

    result = await service.retrieve(
        project_id="proj-1",
        query="Filter by Alice",
        config=RetrievalConfig(
            enable_sparse=False,
            metadata_filters={"author": "Alice"},
        ),
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c-1"


# ==============================================================================
# 5. Reranking and Candidate Batch Text Resolution
# ==============================================================================


@pytest.mark.asyncio
async def test_reranker_bypassed_when_disabled(mock_retrieval_environment):
    """When enable_reranking=False, cross-encoder reranker is not invoked."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="No reranking test",
        config=RetrievalConfig(enable_reranking=False, top_k=2),
    )

    assert len(env["reranker"].calls) == 0
    assert len(result.chunks) <= 2


@pytest.mark.asyncio
async def test_reranker_candidate_text_resolution_from_repo(mock_retrieval_environment):
    """Reranking resolves candidate text from chunk repository when not in vector metadata."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="Deployment guide",
        config=RetrievalConfig(enable_reranking=True),
    )

    # Verify repository was queried to supply content_lookup for reranker
    assert len(env["repo"].fetch_calls) > 0
    assert len(env["reranker"].calls) == 1
    # Check that documents passed to reranker contained the authoritative chunk texts
    _, docs = env["reranker"].calls[0]
    assert any("deployment architecture" in d for d in docs)


# ==============================================================================
# 6. Batched Hydration (No N+1)
# ==============================================================================


@pytest.mark.asyncio
async def test_hydration_uses_single_batched_query(mock_retrieval_environment):
    """Hydration performs a single set-based repository lookup rather than individual queries."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    await service.retrieve(
        project_id="proj-1",
        query="Batch hydration test",
        config=RetrievalConfig(enable_reranking=False),
    )

    # With reranking disabled, only 1 fetch call occurs for hydration
    assert len(env["repo"].fetch_calls) == 1
    pid, requested_ids = env["repo"].fetch_calls[0]
    assert pid == "proj-1"
    assert len(requested_ids) > 1  # multiple chunks retrieved in 1 single call


# ==============================================================================
# 7. Query Transformation (Enabled vs Disabled)
# ==============================================================================


@pytest.mark.asyncio
async def test_query_transformation_disabled_passthrough(mock_retrieval_environment):
    """When enable_transformation=False, query passes through unchanged with zero LLM overhead."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    result = await service.retrieve(
        project_id="proj-1",
        query="Raw unchanged query",
        config=RetrievalConfig(enable_transformation=False),
    )

    env["transformation_service"].transform.assert_not_called()
    assert result.original_query == "Raw unchanged query"
    assert result.retrieval_query == "Raw unchanged query"


# ==============================================================================
# 8. Relevance Check & Bounded Fallback
# ==============================================================================


@pytest.mark.asyncio
async def test_relevance_check_passes_on_first_attempt(mock_retrieval_environment):
    """When relevance check returns True, no fallback is triggered."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    fake_checker = FakeRelevanceChecker([(True, "High relevance")])
    service._relevance_checker = fake_checker

    result = await service.retrieve(
        project_id="proj-1",
        query="Relevant query",
        config=RetrievalConfig(enable_relevance_check=True, max_attempts=2),
    )

    assert result.execution_metadata.attempts_count == 1
    assert not result.execution_metadata.fallback_triggered
    assert len(fake_checker.calls) == 1


@pytest.mark.asyncio
async def test_relevance_check_triggers_fallback_and_succeeds(mock_retrieval_environment):
    """When attempt 1 is irrelevant, triggers fallback transformation and attempt 2 succeeds."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    # Attempt 1: False -> Attempt 2: True
    fake_checker = FakeRelevanceChecker([
        (False, "Insufficient coverage on attempt 1"),
        (True, "Satisfactory coverage on attempt 2"),
    ])
    service._relevance_checker = fake_checker

    result = await service.retrieve(
        project_id="proj-1",
        query="Vague query needing fallback",
        config=RetrievalConfig(enable_relevance_check=True, max_attempts=2),
    )

    assert result.execution_metadata.attempts_count == 2
    assert result.execution_metadata.fallback_triggered
    assert len(fake_checker.calls) == 2

    # Attempt 1 recorded as not relevant
    att1 = result.execution_metadata.attempts[0]
    assert att1.attempt == 1
    assert att1.is_relevant is False

    # Attempt 2 recorded as relevant with transformed query
    att2 = result.execution_metadata.attempts[1]
    assert att2.attempt == 2
    assert att2.is_relevant is True
    assert "expanded" in att2.query


@pytest.mark.asyncio
async def test_relevance_check_bounded_fallback_stops_at_max(mock_retrieval_environment):
    """Fallback does not infinite loop and stops strictly at max_attempts."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    # Both attempts fail relevance
    fake_checker = FakeRelevanceChecker([
        (False, "Failed attempt 1"),
        (False, "Failed attempt 2"),
        (False, "Should never be called"),
    ])
    service._relevance_checker = fake_checker

    result = await service.retrieve(
        project_id="proj-1",
        query="Stubbornly irrelevant query",
        config=RetrievalConfig(enable_relevance_check=True, max_attempts=2),
    )

    assert result.execution_metadata.attempts_count == 2
    assert len(fake_checker.calls) == 2


# ==============================================================================
# 9. Error Handling and Propagation
# ==============================================================================


@pytest.mark.asyncio
async def test_empty_query_raises_empty_query_error(mock_retrieval_environment):
    """Empty or whitespace-only query raises EmptyQueryError."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    with pytest.raises(EmptyQueryError):
        await service.retrieve(project_id="proj-1", query="")

    with pytest.raises(EmptyQueryError):
        await service.retrieve(project_id="proj-1", query="   \n\t  ")


@pytest.mark.asyncio
async def test_embedding_provider_error_propagates(mock_retrieval_environment):
    """Failures in query embedding propagate cleanly as QueryEmbeddingError."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    env["embedding_service"].embed_query_set = AsyncMock(
        side_effect=QueryEmbeddingError("Voyage rate limit exceeded")
    )

    with pytest.raises(QueryEmbeddingError) as exc_info:
        await service.retrieve(project_id="proj-1", query="Test embedding fail")
    assert "Voyage rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_vector_search_failure_propagates(mock_retrieval_environment):
    """Failures in Qdrant dense search propagate cleanly as VectorRetrievalError."""
    env = mock_retrieval_environment
    service: RAGService = env["service"]

    env["vector_service"].search = AsyncMock(
        side_effect=VectorRetrievalError("Qdrant connection timeout")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.retrieve(project_id="proj-1", query="Test vector fail")
    assert "Qdrant connection timeout" in str(exc_info.value)


# ==============================================================================
# 10. API Endpoint Integration Test
# ==============================================================================


@pytest.mark.asyncio
async def test_retrieval_api_endpoint(mock_retrieval_environment):
    """Test retrieval FastAPI route integration."""
    from httpx import ASGITransport, AsyncClient
    from app.main import create_application
    from api.dependencies import get_rag_service_dependency, get_project_service
    from db.session import get_db_session
    from models.project import ProjectModel

    app = create_application()

    # Override database session dependency
    async def fake_get_db():
        yield AsyncMock()

    app.dependency_overrides[get_db_session] = fake_get_db

    # Override RAG service dependency
    app.dependency_overrides[get_rag_service_dependency] = lambda: mock_retrieval_environment["service"]

    # Mock ProjectService to confirm project existence
    mock_proj_service = MagicMock()
    mock_proj_service.get_project_by_id = AsyncMock(
        return_value=ProjectModel(id="proj-1", name="Test Project")
    )
    app.dependency_overrides[get_project_service] = lambda: mock_proj_service

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "What is the deployment topology?",
            "top_k": 3,
            "enable_sparse": True,
        }
        response = await client.post("/projects/proj-1/retrieval", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "proj-1"
        assert data["original_query"] == "What is the deployment topology?"
        assert data["chunk_count"] > 0
        assert len(data["chunks"]) <= 3
        assert "RETRIEVED CONTEXT" in data["formatted_context"]
        assert data["execution_metadata"]["total_duration_ms"] > 0
