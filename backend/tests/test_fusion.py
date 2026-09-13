"""Unit tests for Hybrid Retrieval Fusion (Reciprocal Rank Fusion) stage."""

import math
import pytest

from app.core.config import Settings
from exceptions.retrieval import FusionError, RetrievalError
from rag.retrieval import (
    BaseFusionStrategy as RagBaseFusionStrategy,
    FusedCandidate as RagFusedCandidate,
    FusedSearchCandidate as RagFusedSearchCandidate,
    FusionConfig as RagFusionConfig,
    FusionError as RagFusionError,
    FusionService as RagFusionService,
    ReciprocalRankFusionStrategy as RagReciprocalRankFusionStrategy,
    fuse_results as rag_fuse_results,
    get_fusion_service as rag_get_fusion_service,
    reset_fusion_service as rag_reset_fusion_service,
)
from retrieval import (
    BaseFusionStrategy,
    FusedCandidate,
    FusedSearchCandidate,
    FusionConfig,
    FusionError,
    FusionService,
    KeywordSearchCandidate,
    ReciprocalRankFusionStrategy,
    VectorSearchCandidate,
    fuse_results,
    get_fusion_service,
    reset_fusion_service,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Ensure singleton state is cleanly reset before and after each test."""
    reset_fusion_service()
    rag_reset_fusion_service()
    yield
    reset_fusion_service()
    rag_reset_fusion_service()


def _make_dense_candidate(
    chunk_id: str,
    score: float = 0.9,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
) -> VectorSearchCandidate:
    return VectorSearchCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        score=score,
        query_type="original",
        document_version_id=document_version_id,
    )


def _make_sparse_candidate(
    chunk_id: str,
    score: float = 15.5,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
) -> KeywordSearchCandidate:
    return KeywordSearchCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        score=score,
        query_type="original",
        document_version_id=document_version_id,
    )


# ==============================================================================
# 1. Combination and Deduplication
# ==============================================================================

def test_fuse_combines_and_deduplicates_dense_and_sparse():
    """Dense and sparse candidate lists are combined into a single unified list with duplicates merged."""
    dense = [
        _make_dense_candidate("chunk-A", score=0.95),
        _make_dense_candidate("chunk-B", score=0.85),
        _make_dense_candidate("chunk-C", score=0.75),
    ]
    sparse = [
        _make_sparse_candidate("chunk-C", score=20.0),
        _make_sparse_candidate("chunk-D", score=18.0),
        _make_sparse_candidate("chunk-A", score=15.0),
    ]

    results = fuse_results(dense_results=dense, sparse_results=sparse, top_k=None)

    chunk_ids = [c.chunk_id for c in results]
    assert len(chunk_ids) == 4
    assert set(chunk_ids) == {"chunk-A", "chunk-B", "chunk-C", "chunk-D"}


def test_candidates_in_both_lists_receive_dual_rrf_contributions():
    """Candidates appearing in both lists receive exact reciprocal rank contributions from both."""
    dense = [
        _make_dense_candidate("chunk-A", score=0.9),  # Dense rank 1
        _make_dense_candidate("chunk-B", score=0.8),  # Dense rank 2
    ]
    sparse = [
        _make_sparse_candidate("chunk-B", score=10.0),  # Sparse rank 1
        _make_sparse_candidate("chunk-A", score=5.0),   # Sparse rank 2
    ]

    strategy = ReciprocalRankFusionStrategy(rrf_k=60, default_top_k=None)
    results = strategy.fuse(dense, sparse)

    a_cand = next(c for c in results if c.chunk_id == "chunk-A")
    b_cand = next(c for c in results if c.chunk_id == "chunk-B")

    expected_a = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    expected_b = (1.0 / (60 + 2)) + (1.0 / (60 + 1))

    assert math.isclose(a_cand.score, expected_a, rel_tol=1e-9)
    assert math.isclose(b_cand.score, expected_b, rel_tol=1e-9)
    assert a_cand.dense_rank == 1
    assert a_cand.sparse_rank == 2
    assert b_cand.dense_rank == 2
    assert b_cand.sparse_rank == 1


def test_candidates_appearing_only_in_dense_remain():
    """Candidates returned only by dense search are preserved with single contribution and sparse_rank=None."""
    dense = [_make_dense_candidate("chunk-only-dense", score=0.88)]
    sparse = []

    results = fuse_results(dense, sparse, top_k=None)
    assert len(results) == 1
    cand = results[0]
    assert cand.chunk_id == "chunk-only-dense"
    assert cand.dense_rank == 1
    assert cand.sparse_rank is None
    assert cand.dense_score == 0.88
    assert cand.sparse_score is None
    assert math.isclose(cand.score, 1.0 / (60 + 1), rel_tol=1e-9)


def test_candidates_appearing_only_in_sparse_remain():
    """Candidates returned only by sparse search are preserved with single contribution and dense_rank=None."""
    dense = []
    sparse = [_make_sparse_candidate("chunk-only-sparse", score=12.5)]

    results = fuse_results(dense, sparse, top_k=None)
    assert len(results) == 1
    cand = results[0]
    assert cand.chunk_id == "chunk-only-sparse"
    assert cand.dense_rank is None
    assert cand.sparse_rank == 1
    assert cand.dense_score is None
    assert cand.sparse_score == 12.5
    assert math.isclose(cand.score, 1.0 / (60 + 1), rel_tol=1e-9)


# ==============================================================================
# 2. Mathematical Correctness & RRF Formulation
# ==============================================================================

def test_rrf_mathematical_calculation_exact():
    """Verify exact formula: RRF(d) = sum(1 / (k + rank(d)))."""
    k = 60
    dense = [
        _make_dense_candidate("A", score=0.9),  # rank 1
        _make_dense_candidate("B", score=0.8),  # rank 2
        _make_dense_candidate("C", score=0.7),  # rank 3
    ]
    sparse = [
        _make_sparse_candidate("C", score=30.0),  # rank 1
        _make_sparse_candidate("D", score=20.0),  # rank 2
        _make_sparse_candidate("A", score=10.0),  # rank 3
    ]

    strategy = ReciprocalRankFusionStrategy(rrf_k=k, default_top_k=None)
    results = strategy.fuse(dense, sparse)
    scores = {c.chunk_id: c.score for c in results}

    assert math.isclose(scores["A"], (1.0 / 61) + (1.0 / 63), rel_tol=1e-9)
    assert math.isclose(scores["B"], 1.0 / 62, rel_tol=1e-9)
    assert math.isclose(scores["C"], (1.0 / 63) + (1.0 / 61), rel_tol=1e-9)
    assert math.isclose(scores["D"], 1.0 / 62, rel_tol=1e-9)


def test_rrf_uses_ranks_not_raw_scores():
    """RRF must rank candidates by list rank, not by adding raw vector and keyword scores."""
    # Chunk-X: High raw scores (0.99 vector, 999.0 keyword) but placed at rank 10
    # Chunk-Y: Modest raw scores (0.01 vector, 1.0 keyword) but placed at rank 1
    dense = [
        _make_dense_candidate(f"pad-{i}", score=1.0 - (i * 0.05)) for i in range(1, 10)
    ] + [_make_dense_candidate("chunk-X", score=0.99)]
    # Insert chunk-Y at rank 1
    dense.insert(0, _make_dense_candidate("chunk-Y", score=0.01))

    sparse = [
        _make_sparse_candidate(f"spad-{i}", score=500.0 - i) for i in range(1, 10)
    ] + [_make_sparse_candidate("chunk-X", score=999.0)]
    sparse.insert(0, _make_sparse_candidate("chunk-Y", score=1.0))

    strategy = ReciprocalRankFusionStrategy(rrf_k=60, default_top_k=None)
    results = strategy.fuse(dense, sparse)

    y_cand = next(c for c in results if c.chunk_id == "chunk-Y")
    x_cand = next(c for c in results if c.chunk_id == "chunk-X")

    # chunk-Y was rank 1 in both (score = 1/61 + 1/61 = 0.03278)
    # chunk-X was rank 11 in both (score = 1/71 + 1/71 = 0.02816)
    assert y_cand.score > x_cand.score
    assert y_cand.rank < x_cand.rank


def test_rrf_k_parameter_is_respected():
    """Custom rrf_k constant changes fused score calculation accordingly."""
    dense = [_make_dense_candidate("chunk-1")]
    sparse = []

    strat_10 = ReciprocalRankFusionStrategy(rrf_k=10)
    strat_100 = ReciprocalRankFusionStrategy(rrf_k=100)

    res_10 = strat_10.fuse(dense, sparse)
    res_100 = strat_100.fuse(dense, sparse)

    assert math.isclose(res_10[0].score, 1.0 / (10 + 1), rel_tol=1e-9)
    assert math.isclose(res_100[0].score, 1.0 / (100 + 1), rel_tol=1e-9)


def test_first_result_uses_rank_1():
    """Candidate with highest score must have rank 1, followed by rank 2, etc."""
    dense = [
        _make_dense_candidate("first"),
        _make_dense_candidate("second"),
    ]
    sparse = []

    results = fuse_results(dense, sparse)
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_results_ordered_by_descending_fused_score():
    """Final unified candidate list must be ordered strictly by descending fused score."""
    dense = [
        _make_dense_candidate("rank-1-dense"),
        _make_dense_candidate("rank-2-dense"),
        _make_dense_candidate("rank-3-dense"),
    ]
    sparse = [
        _make_sparse_candidate("rank-1-dense"),  # boost top candidate
        _make_sparse_candidate("rank-1-sparse"),
    ]

    results = fuse_results(dense, sparse, top_k=None)
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_equal_scores_have_deterministic_ordering():
    """Candidates with equal RRF scores must be deterministically tie-broken by chunk_id."""
    # Chunk "B" and Chunk "A" both have identical rank 1 in their respective single branch
    dense = [_make_dense_candidate("chunk-B")]
    sparse = [_make_sparse_candidate("chunk-A")]

    strategy = ReciprocalRankFusionStrategy(rrf_k=60, default_top_k=None)
    results = strategy.fuse(dense, sparse)

    assert len(results) == 2
    assert results[0].score == results[1].score
    # Ascending lexicographical tie-break: "chunk-A" precedes "chunk-B"
    assert results[0].chunk_id == "chunk-A"
    assert results[0].rank == 1
    assert results[1].chunk_id == "chunk-B"
    assert results[1].rank == 2


# ==============================================================================
# 3. Empty Results & Edge Cases
# ==============================================================================

def test_empty_dense_results():
    """Empty dense results with valid sparse results returns valid fused candidates."""
    dense = []
    sparse = [
        _make_sparse_candidate("s-1"),
        _make_sparse_candidate("s-2"),
    ]
    results = fuse_results(dense, sparse)
    assert len(results) == 2
    assert results[0].chunk_id == "s-1"
    assert results[0].rank == 1
    assert results[1].chunk_id == "s-2"
    assert results[1].rank == 2


def test_empty_sparse_results():
    """Empty sparse results with valid dense results returns valid fused candidates."""
    dense = [
        _make_dense_candidate("d-1"),
        _make_dense_candidate("d-2"),
    ]
    sparse = []
    results = fuse_results(dense, sparse)
    assert len(results) == 2
    assert results[0].chunk_id == "d-1"
    assert results[0].rank == 1
    assert results[1].chunk_id == "d-2"
    assert results[1].rank == 2


def test_both_empty_results():
    """Both empty results return an empty list without error."""
    results = fuse_results([], [])
    assert results == []


def test_intra_list_duplicates_resilience():
    """Duplicate candidate chunks within a single list keep best rank without double counting."""
    dense = [
        _make_dense_candidate("dup-chunk", score=0.9),   # rank 1
        _make_dense_candidate("other-chunk", score=0.8), # rank 2
        _make_dense_candidate("dup-chunk", score=0.7),   # duplicate at rank 3
    ]
    sparse = []

    results = fuse_results(dense, sparse)
    assert len(results) == 2
    dup = next(c for c in results if c.chunk_id == "dup-chunk")
    assert dup.dense_rank == 1
    assert math.isclose(dup.score, 1.0 / (60 + 1), rel_tol=1e-9)


def test_discard_malformed_candidates(caplog):
    """Malformed candidates missing required identifiers are discarded and logged."""
    malformed_dense = [
        VectorSearchCandidate(
            chunk_id="",
            document_id="doc-1",
            project_id="proj-1",
            score=0.9,
        ),
        VectorSearchCandidate(
            chunk_id="valid-chunk",
            document_id="",
            project_id="proj-1",
            score=0.9,
        ),
        _make_dense_candidate("good-chunk"),
    ]
    results = fuse_results(malformed_dense, [])
    assert len(results) == 1
    assert results[0].chunk_id == "good-chunk"


# ==============================================================================
# 4. Project Isolation & Security
# ==============================================================================

def test_project_isolation_preservation():
    """Candidate project_id is preserved faithfully onto FusedCandidate."""
    dense = [_make_dense_candidate("chunk-1", project_id="tenant-alpha")]
    sparse = [_make_sparse_candidate("chunk-2", project_id="tenant-alpha")]

    results = fuse_results(dense, sparse)
    for c in results:
        assert c.project_id == "tenant-alpha"


def test_project_isolation_defensive_filtering():
    """When project_id is passed, cross-project leaked candidates are dropped defensively."""
    dense = [
        _make_dense_candidate("chunk-allowed", project_id="tenant-alpha"),
        _make_dense_candidate("chunk-leaked", project_id="tenant-beta"),
    ]
    sparse = [
        _make_sparse_candidate("chunk-allowed-2", project_id="tenant-alpha"),
        _make_sparse_candidate("chunk-leaked-2", project_id="tenant-gamma"),
    ]

    results = fuse_results(dense, sparse, project_id="tenant-alpha")
    chunk_ids = [c.chunk_id for c in results]
    assert set(chunk_ids) == {"chunk-allowed", "chunk-allowed-2"}
    assert all(c.project_id == "tenant-alpha" for c in results)


def test_invalid_project_id_raises():
    """Whitespace-only or non-string project_id raises FusionError."""
    with pytest.raises(FusionError, match="project_id must be a non-empty string"):
        fuse_results([], [], project_id="   ")


# ==============================================================================
# 5. Top-K Bounding
# ==============================================================================

def test_top_k_parameter_limits_output():
    """Explicit top_k limits the number of returned candidates."""
    dense = [_make_dense_candidate(f"chunk-{i}") for i in range(10)]
    sparse = [_make_sparse_candidate(f"chunk-{i}") for i in range(10, 20)]

    results = fuse_results(dense, sparse, top_k=5)
    assert len(results) == 5
    assert [c.rank for c in results] == [1, 2, 3, 4, 5]


def test_top_k_none_returns_all():
    """Passing top_k=None returns all unique fused candidates without truncation."""
    dense = [_make_dense_candidate(f"chunk-{i}") for i in range(10)]
    sparse = [_make_sparse_candidate(f"chunk-{i}") for i in range(10, 20)]

    results = fuse_results(dense, sparse, top_k=None)
    assert len(results) == 20


def test_invalid_top_k_raises():
    """Negative or zero top_k raises FusionError."""
    with pytest.raises(FusionError, match="top_k must be a positive integer or None"):
        fuse_results([], [], top_k=0)

    with pytest.raises(FusionError, match="top_k must be a positive integer or None"):
        fuse_results([], [], top_k=-5)


# ==============================================================================
# 6. Configuration & Models
# ==============================================================================

def test_fusion_config_validation():
    """FusionConfig parameters are strictly validated."""
    config = FusionConfig(rrf_k=60, top_k=10)
    assert config.rrf_k == 60
    assert config.top_k == 10

    with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
        FusionConfig(rrf_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        FusionConfig(top_k=-1)


def test_fusion_config_from_settings():
    """FusionConfig loads single-source settings from Settings instance."""
    settings = Settings(FUSION_RRF_K=45, FUSION_TOP_K=15)
    config = FusionConfig.from_settings(settings)
    assert config.rrf_k == 45
    assert config.top_k == 15


def test_fused_candidate_properties_and_serialization():
    """FusedCandidate serialization and properties function as expected."""
    cand = FusedCandidate(
        chunk_id="cid-1",
        document_id="did-1",
        project_id="pid-1",
        score=0.032,
        rank=1,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.95,
        sparse_score=15.0,
        document_version_id="vid-1",
    )

    assert cand.fused_score == 0.032
    d = cand.to_dict()
    assert d["chunk_id"] == "cid-1"
    assert d["document_id"] == "did-1"
    assert d["project_id"] == "pid-1"
    assert d["score"] == 0.032
    assert d["fused_score"] == 0.032
    assert d["rank"] == 1
    assert d["dense_rank"] == 1
    assert d["sparse_rank"] == 2
    assert d["dense_score"] == 0.95
    assert d["sparse_score"] == 15.0
    assert d["document_version_id"] == "vid-1"


# ==============================================================================
# 7. Service Lifecycle & Facades
# ==============================================================================

def test_service_singleton_lifecycle():
    """get_fusion_service returns singleton and reset_fusion_service clears it."""
    s1 = get_fusion_service()
    s2 = get_fusion_service()
    assert s1 is s2

    reset_fusion_service()
    s3 = get_fusion_service()
    assert s3 is not s1


def test_service_dependency_injection():
    """FusionService accepts custom strategy and config."""
    custom_strategy = ReciprocalRankFusionStrategy(rrf_k=30, default_top_k=3)
    service = FusionService(strategy=custom_strategy)
    assert service.strategy is custom_strategy
    assert service.strategy.strategy_name == "rrf"

    dense = [_make_dense_candidate(f"c-{i}") for i in range(10)]
    fused = service.fuse(dense, [])
    assert len(fused) == 3


def test_invalid_input_sequences_raise_fusion_error():
    """Non-sequence inputs raise FusionError."""
    service = get_fusion_service()
    with pytest.raises(FusionError, match="Expected Sequence for dense_results"):
        service.fuse("not-a-sequence", [])  # type: ignore

    with pytest.raises(FusionError, match="Expected Sequence for sparse_results"):
        service.fuse([], 12345)  # type: ignore


# ==============================================================================
# 8. RAG Retrieval Alias Re-exports
# ==============================================================================

def test_rag_retrieval_re_exports():
    """Verify that rag.retrieval re-exports all fusion contracts identically."""
    assert RagBaseFusionStrategy is BaseFusionStrategy
    assert RagReciprocalRankFusionStrategy is ReciprocalRankFusionStrategy
    assert RagFusionService is FusionService
    assert RagFusionConfig is FusionConfig
    assert RagFusedCandidate is FusedCandidate
    assert RagFusedSearchCandidate is FusedSearchCandidate
    assert RagFusionError is FusionError
    assert issubclass(RagFusionError, RetrievalError)
    assert rag_fuse_results is fuse_results
    assert rag_get_fusion_service is get_fusion_service
    assert rag_reset_fusion_service is reset_fusion_service
