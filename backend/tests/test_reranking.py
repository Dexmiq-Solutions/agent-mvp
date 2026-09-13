"""Unit tests for the Cross-Encoder Reranking stage in the retrieval pipeline."""

import asyncio
from typing import Any, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import voyageai.error as voyage_errors

from exceptions.retrieval import (
    RerankingConfigurationError,
    RerankingError,
    RerankingProviderError,
    RerankingTimeoutError,
    RerankingValidationError,
    RetrievalError,
)
from rag.retrieval import (
    BaseReranker as RagBaseReranker,
    RerankedCandidate as RagRerankedCandidate,
    RerankedSearchCandidate as RagRerankedSearchCandidate,
    RerankingConfig as RagRerankingConfig,
    RerankingError as RagRerankingError,
    RerankingService as RagRerankingService,
    ScoredDocument as RagScoredDocument,
    VoyageReranker as RagVoyageReranker,
    get_reranking_service as rag_get_reranking_service,
    rerank_candidates as rag_rerank_candidates,
    reset_reranking_service as rag_reset_reranking_service,
)
from retrieval import (
    BaseReranker,
    FusedCandidate,
    ProcessedQuery,
    RerankedCandidate,
    RerankedSearchCandidate,
    RerankingConfig,
    RerankingService,
    RetrievalQuerySet,
    ScoredDocument,
    VoyageReranker,
    get_reranking_service,
    rerank_candidates,
    reset_reranking_service,
    resolve_candidate_text,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Ensure singleton state is cleanly reset before and after each test."""
    reset_reranking_service()
    rag_reset_reranking_service()
    yield
    reset_reranking_service()
    rag_reset_reranking_service()


def _make_fused_candidate(
    chunk_id: str,
    score: float = 0.035,
    rank: int = 1,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
    dense_rank: int | None = 1,
    sparse_rank: int | None = 2,
    dense_score: float | None = 0.95,
    sparse_score: float | None = 18.0,
    metadata: dict | None = None,
    content: str | None = None,
) -> FusedCandidate:
    """Helper to construct a test FusedCandidate instance."""
    meta = dict(metadata or {})
    if content is not None and "content" not in meta:
        meta["content"] = content

    return FusedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        score=score,
        rank=rank,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=dense_score,
        sparse_score=sparse_score,
        document_version_id=document_version_id,
        metadata=meta,
    )


class FakeReranker(BaseReranker):
    """Deterministic fake reranker for testing without external API calls."""

    def __init__(
        self,
        scores_by_text: Optional[Mapping[str, float]] = None,
        default_score: float = 0.5,
        model_name: str = "test-reranker",
    ) -> None:
        self._scores_by_text = dict(scores_by_text or {})
        self._default_score = default_score
        self._model_name = model_name
        self.call_history: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[ScoredDocument]:
        self.call_history.append({"query": query, "documents": list(documents), "top_k": top_k})
        results: list[ScoredDocument] = []
        for idx, doc in enumerate(documents):
            score = self._scores_by_text.get(doc, self._default_score)
            results.append(ScoredDocument(index=idx, score=score))
        return results


# ==============================================================================
# 1. Basic Ranking & Reordering
# ==============================================================================

@pytest.mark.asyncio
async def test_basic_reranking_reorders_candidates_by_score():
    """Candidates are reordered strictly according to cross-encoder relevance scores."""
    c1 = _make_fused_candidate("chunk-1", rank=1, content="python intro")
    c2 = _make_fused_candidate("chunk-2", rank=2, content="advanced python")
    c3 = _make_fused_candidate("chunk-3", rank=3, content="unrelated topic")

    # Reranker assigns highest score to c2, then c1, then c3
    fake_reranker = FakeReranker(
        scores_by_text={
            "python intro": 0.65,
            "advanced python": 0.98,
            "unrelated topic": 0.12,
        }
    )
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(
        query="advanced python",
        candidates=[c1, c2, c3],
    )

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["chunk-2", "chunk-1", "chunk-3"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.rerank_score for r in results] == [0.98, 0.65, 0.12]
    assert [r.score for r in results] == [0.98, 0.65, 0.12]


# ==============================================================================
# 2. Score Preservation & Provenance
# ==============================================================================

@pytest.mark.asyncio
async def test_rerank_preserves_provenance_and_does_not_combine_scores():
    """Dense, sparse, and RRF fused scores are preserved without mathematical combination."""
    c1 = _make_fused_candidate(
        "chunk-1",
        score=0.033,  # RRF score
        dense_rank=2,
        sparse_rank=1,
        dense_score=0.88,
        sparse_score=14.2,
        content="some chunk content",
    )

    fake_reranker = FakeReranker(scores_by_text={"some chunk content": 0.92})
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(query="query", candidates=[c1])
    assert len(results) == 1
    r = results[0]

    # Reranker score establishes output ranking
    assert r.rerank_score == 0.92
    assert r.score == 0.92
    assert r.rank == 1

    # First-stage provenance preserved verbatim
    assert r.fusion_score == 0.033
    assert r.dense_rank == 2
    assert r.sparse_rank == 1
    assert r.dense_score == 0.88
    assert r.sparse_score == 14.2


# ==============================================================================
# 3. Candidate Identity Preservation
# ==============================================================================

@pytest.mark.asyncio
async def test_candidate_identity_and_metadata_preserved():
    """All identity fields and arbitrary metadata dictionaries remain intact."""
    c = _make_fused_candidate(
        chunk_id="chunk-xyz",
        project_id="proj-99",
        document_id="doc-abc",
        document_version_id="ver-3",
        metadata={"author": "alice", "content": "text payload", "tags": ["rag", "ai"]},
    )

    fake_reranker = FakeReranker(scores_by_text={"text payload": 0.85})
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(query="test", candidates=[c])
    assert len(results) == 1
    r = results[0]

    assert r.chunk_id == "chunk-xyz"
    assert r.project_id == "proj-99"
    assert r.document_id == "doc-abc"
    assert r.document_version_id == "ver-3"
    assert r.metadata["author"] == "alice"
    assert r.metadata["tags"] == ["rag", "ai"]

    # Test serialization
    as_dict = r.to_dict()
    assert as_dict["chunk_id"] == "chunk-xyz"
    assert as_dict["rerank_score"] == 0.85
    assert as_dict["metadata"]["author"] == "alice"


# ==============================================================================
# 4. Deduplication
# ==============================================================================

@pytest.mark.asyncio
async def test_deduplication_of_duplicate_candidate_chunks():
    """Duplicate candidate chunk IDs in input are deterministically deduplicated."""
    c1 = _make_fused_candidate("chunk-dup", score=0.04, content="duplicate chunk text")
    c2 = _make_fused_candidate("chunk-dup", score=0.02, content="duplicate chunk text")
    c3 = _make_fused_candidate("chunk-unique", score=0.01, content="unique chunk text")

    fake_reranker = FakeReranker(
        scores_by_text={"duplicate chunk text": 0.9, "unique chunk text": 0.8}
    )
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(query="query", candidates=[c1, c2, c3])

    # Only 2 distinct chunks sent to reranker and returned
    assert len(fake_reranker.call_history[0]["documents"]) == 2
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["chunk-dup", "chunk-unique"]


# ==============================================================================
# 5. Candidate and Result Limits
# ==============================================================================

@pytest.mark.asyncio
async def test_candidate_and_result_limits_enforced():
    """Only candidate_limit inputs are sent to reranker, and result_limit are returned."""
    candidates = [
        _make_fused_candidate(f"chunk-{i}", rank=i, content=f"text-{i}")
        for i in range(1, 21)
    ]

    # Scores make chunk-5 highest, chunk-10 second highest
    scores = {f"text-{i}": 0.1 for i in range(1, 21)}
    scores["text-5"] = 0.95
    scores["text-10"] = 0.90
    fake_reranker = FakeReranker(scores_by_text=scores)

    service = RerankingService(
        reranker=fake_reranker,
        config=RerankingConfig(candidate_limit=8, result_limit=3),
    )

    # With candidate_limit=8, chunk-10 is NOT included in the candidate window (1..8)
    results = await service.rerank(query="test", candidates=candidates)

    assert len(fake_reranker.call_history[0]["documents"]) == 8
    assert len(results) == 3
    assert results[0].chunk_id == "chunk-5"
    assert results[0].rank == 1
    # chunk-10 should not be in results because it exceeded candidate_limit
    assert all(r.chunk_id != "chunk-10" for r in results)


@pytest.mark.asyncio
async def test_dynamic_limit_overrides():
    """Per-call candidate_limit and result_limit override service configuration."""
    candidates = [
        _make_fused_candidate(f"chunk-{i}", content=f"doc-{i}") for i in range(1, 11)
    ]
    fake_reranker = FakeReranker()
    service = RerankingService(
        reranker=fake_reranker,
        config=RerankingConfig(candidate_limit=10, result_limit=5),
    )

    results = await service.rerank(
        query="query",
        candidates=candidates,
        candidate_limit=4,
        result_limit=2,
    )

    assert len(fake_reranker.call_history[0]["documents"]) == 4
    assert len(results) == 2


# ==============================================================================
# 6. Empty Candidate Handling
# ==============================================================================

@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_without_calling_reranker():
    """Empty candidate sequence returns empty list immediately without provider call."""
    fake_reranker = FakeReranker()
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(query="query", candidates=[])
    assert results == []
    assert len(fake_reranker.call_history) == 0


# ==============================================================================
# 7. Reranker Disabled Behavior
# ==============================================================================

@pytest.mark.asyncio
async def test_reranker_disabled_bypasses_provider_cleanly():
    """When disabled, provider is not invoked and candidates pass through preserving order."""
    c1 = _make_fused_candidate("chunk-1", rank=1, content="first")
    c2 = _make_fused_candidate("chunk-2", rank=2, content="second")
    c3 = _make_fused_candidate("chunk-3", rank=3, content="third")

    fake_reranker = FakeReranker()
    service = RerankingService(
        reranker=fake_reranker,
        config=RerankingConfig(enabled=False, result_limit=2),
    )

    results = await service.rerank(query="query", candidates=[c1, c2, c3])

    assert len(fake_reranker.call_history) == 0
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[1].chunk_id == "chunk-2"


# ==============================================================================
# 8. Project Isolation Enforcement
# ==============================================================================

@pytest.mark.asyncio
async def test_project_isolation_discards_leaked_candidates():
    """Candidates from other tenants are defensively discarded."""
    c1 = _make_fused_candidate("c-1", project_id="tenant-alpha", content="text 1")
    c2 = _make_fused_candidate("c-2", project_id="tenant-beta", content="text 2")  # Leaked!
    c3 = _make_fused_candidate("c-3", project_id="tenant-alpha", content="text 3")

    fake_reranker = FakeReranker(scores_by_text={"text 1": 0.7, "text 3": 0.8})
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(
        query="query",
        candidates=[c1, c2, c3],
        project_id="tenant-alpha",
    )

    # c2 must not have been sent to reranker or returned
    assert len(fake_reranker.call_history[0]["documents"]) == 2
    assert [r.chunk_id for r in results] == ["c-3", "c-1"]
    assert all(r.project_id == "tenant-alpha" for r in results)


# ==============================================================================
# 9. Deterministic Ordering with Tied Scores
# ==============================================================================

@pytest.mark.asyncio
async def test_deterministic_ordering_on_tied_scores():
    """Identical rerank scores are tie-broken deterministically by chunk_id."""
    c1 = _make_fused_candidate("chunk-zebra", content="t1")
    c2 = _make_fused_candidate("chunk-apple", content="t2")
    c3 = _make_fused_candidate("chunk-mango", content="t3")

    # All tied at score 0.75
    fake_reranker = FakeReranker(default_score=0.75)
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(query="query", candidates=[c1, c2, c3])

    assert len(results) == 3
    # Lexicographically sorted by chunk_id: apple, mango, zebra
    assert [r.chunk_id for r in results] == ["chunk-apple", "chunk-mango", "chunk-zebra"]
    assert [r.rank for r in results] == [1, 2, 3]


# ==============================================================================
# 10. Chunk Text Resolution
# ==============================================================================

@pytest.mark.asyncio
async def test_chunk_text_resolution_precedence():
    """Text is resolved from candidate attributes, metadata, or external content_lookup."""
    # 1. Via candidate attribute
    c_attr = _make_fused_candidate("c-attr")
    object.__setattr__(c_attr, "content", "attribute text")

    # 2. Via metadata dict
    c_meta = _make_fused_candidate("c-meta", metadata={"content": "metadata text"})

    # 3. Via external lookup mapping
    c_lookup = _make_fused_candidate("c-lookup", metadata={})
    lookup = {"c-lookup": "external lookup text"}

    fake_reranker = FakeReranker(
        scores_by_text={
            "attribute text": 0.9,
            "metadata text": 0.8,
            "external lookup text": 0.7,
        }
    )
    service = RerankingService(reranker=fake_reranker)

    results = await service.rerank(
        query="query",
        candidates=[c_attr, c_meta, c_lookup],
        content_lookup=lookup,
    )

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["c-attr", "c-meta", "c-lookup"]


@pytest.mark.asyncio
async def test_missing_chunk_text_raises_validation_error():
    """If a candidate's chunk text cannot be resolved, raises RerankingValidationError."""
    c = _make_fused_candidate("c-empty", metadata={})  # No content anywhere
    fake_reranker = FakeReranker()
    service = RerankingService(reranker=fake_reranker)

    with pytest.raises(RerankingValidationError) as exc_info:
        await service.rerank(query="query", candidates=[c])
    assert "Could not resolve non-empty chunk text" in str(exc_info.value)


# ==============================================================================
# 11. Query Type Inputs & Validation
# ==============================================================================

@pytest.mark.asyncio
async def test_query_types_and_validation():
    """Supports string, ProcessedQuery, and RetrievalQuerySet, and rejects empty query."""
    c = _make_fused_candidate("c-1", content="some text")
    fake_reranker = FakeReranker()
    service = RerankingService(reranker=fake_reranker)

    # ProcessedQuery
    pq = ProcessedQuery(original_query="raw", processed_query="normalized")
    res1 = await service.rerank(query=pq, candidates=[c])
    assert len(res1) == 1
    assert fake_reranker.call_history[-1]["query"] == "normalized"

    # RetrievalQuerySet
    rqs = RetrievalQuerySet(original_query="authoritative query")
    res2 = await service.rerank(query=rqs, candidates=[c])
    assert len(res2) == 1
    assert fake_reranker.call_history[-1]["query"] == "authoritative query"

    # Empty string query
    with pytest.raises(RerankingValidationError):
        await service.rerank(query="   ", candidates=[c])

    # Non-string invalid query type
    with pytest.raises(RerankingValidationError):
        await service.rerank(query=12345, candidates=[c])  # type: ignore


# ==============================================================================
# 12. VoyageReranker Concrete Provider Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_voyage_reranker_success_mapping():
    """VoyageReranker calls AsyncClient.rerank and parses results correctly."""
    mock_client = AsyncMock()

    # Create mock response mimicking Voyage RerankingObject
    mock_item0 = MagicMock(index=1, relevance_score=0.91)
    mock_item1 = MagicMock(index=0, relevance_score=0.42)
    mock_response = MagicMock(results=[mock_item0, mock_item1])
    mock_client.rerank.return_value = mock_response

    provider = VoyageReranker(model="rerank-2.5", client=mock_client)
    docs = ["doc A", "doc B"]

    results = await provider.rerank(query="search query", documents=docs)

    mock_client.rerank.assert_awaited_once_with(
        query="search query",
        documents=docs,
        model="rerank-2.5",
        top_k=None,
        truncation=True,
    )
    assert len(results) == 2
    assert results[0].index == 1
    assert results[0].score == 0.91
    assert results[1].index == 0
    assert results[1].score == 0.42


@pytest.mark.asyncio
async def test_voyage_reranker_timeout_error():
    """VoyageReranker translates timeout to RerankingTimeoutError."""
    mock_client = AsyncMock()

    async def slow_rerank(*args, **kwargs):
        await asyncio.sleep(0.5)

    mock_client.rerank.side_effect = slow_rerank

    provider = VoyageReranker(model="rerank-2.5", client=mock_client, timeout_seconds=0.05)

    with pytest.raises(RerankingTimeoutError):
        await provider.rerank(query="query", documents=["doc"])


@pytest.mark.asyncio
async def test_voyage_reranker_auth_error_translation():
    """VoyageReranker translates AuthenticationError to RerankingProviderError."""
    mock_client = AsyncMock()
    mock_client.rerank.side_effect = voyage_errors.AuthenticationError("Invalid API Key")

    provider = VoyageReranker(client=mock_client)

    with pytest.raises(RerankingProviderError) as exc_info:
        await provider.rerank(query="query", documents=["doc"])
    assert "Authentication failed with Voyage AI" in str(exc_info.value)


@pytest.mark.asyncio
async def test_voyage_reranker_rate_limit_error_translation():
    """VoyageReranker translates RateLimitError to RerankingProviderError."""
    mock_client = AsyncMock()
    mock_client.rerank.side_effect = voyage_errors.RateLimitError("Rate limit exceeded")

    provider = VoyageReranker(client=mock_client)

    with pytest.raises(RerankingProviderError) as exc_info:
        await provider.rerank(query="query", documents=["doc"])
    assert "rate limit exceeded" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_voyage_reranker_invalid_response_indices():
    """VoyageReranker rejects responses containing out-of-bounds document indices."""
    mock_client = AsyncMock()
    mock_item = MagicMock(index=99, relevance_score=0.8)  # Index 99 out of range for 1 doc
    mock_response = MagicMock(results=[mock_item])
    mock_client.rerank.return_value = mock_response

    provider = VoyageReranker(client=mock_client)

    with pytest.raises(RerankingValidationError) as exc_info:
        await provider.rerank(query="query", documents=["doc 0"])
    assert "invalid candidate index" in str(exc_info.value)


# ==============================================================================
# 13. Service Lifecycle, Facades, and Functional Entrypoints
# ==============================================================================

def test_service_lifecycle_and_rag_reexports():
    """Verifies get/reset lifecycle and confirms rag.retrieval re-exports identical classes."""
    svc1 = get_reranking_service()
    svc2 = rag_get_reranking_service()
    assert svc1 is svc2

    reset_reranking_service()
    svc3 = get_reranking_service()
    assert svc3 is not svc1

    # Verify RAG facade compatibility
    assert RagRerankingService is RerankingService
    assert RagBaseReranker is BaseReranker
    assert RagVoyageReranker is VoyageReranker
    assert RagScoredDocument is ScoredDocument
    assert RagRerankedCandidate is RerankedCandidate
    assert RagRerankedSearchCandidate is RerankedSearchCandidate
    assert RagRerankingConfig is RerankingConfig
    assert RagRerankingError is RerankingError


@pytest.mark.asyncio
async def test_functional_rerank_candidates_helper():
    """rerank_candidates functional helper coordinates with active service."""
    c = _make_fused_candidate("c-1", content="chunk body")
    fake = FakeReranker(scores_by_text={"chunk body": 0.88})

    results = await rerank_candidates(
        query="query",
        candidates=[c],
        service=RerankingService(reranker=fake),
    )
    assert len(results) == 1
    assert results[0].rerank_score == 0.88


# ==============================================================================
# 14. Configuration Validation
# ==============================================================================

def test_reranking_config_validation():
    """Invalid configuration values raise ValueError upon initialization."""
    with pytest.raises(ValueError):
        RerankingConfig(model="")

    with pytest.raises(ValueError):
        RerankingConfig(candidate_limit=0)

    with pytest.raises(ValueError):
        RerankingConfig(result_limit=-5)

    with pytest.raises(ValueError):
        RerankingConfig(timeout_seconds=0)

    # Valid config
    cfg = RerankingConfig(model="rerank-2.5", candidate_limit=40, result_limit=8, timeout_seconds=5.0)
    assert cfg.model == "rerank-2.5"
    assert cfg.candidate_limit == 40
    assert cfg.result_limit == 8
    assert cfg.timeout_seconds == 5.0
