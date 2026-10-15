"""Unit tests for Metadata Filtering stage in the RAG retrieval pipeline."""

from datetime import date, datetime
import pytest

from exceptions.retrieval import (
    FilterEvaluationError,
    InvalidFilterError,
    MetadataFilteringError,
    RetrievalError,
)
from rag.retrieval import (
    BaseFilterCondition as RagBaseFilterCondition,
    BooleanCondition as RagBooleanCondition,
    CandidateMetadataResolver as RagCandidateMetadataResolver,
    ComparisonOperator as RagComparisonOperator,
    CompoundCondition as RagCompoundCondition,
    ExactMatchCondition as RagExactMatchCondition,
    LogicalOperator as RagLogicalOperator,
    MetadataConditionEvaluator as RagMetadataConditionEvaluator,
    MetadataFilter as RagMetadataFilter,
    MetadataFilteringConfig as RagMetadataFilteringConfig,
    MetadataFilteringError as RagMetadataFilteringError,
    MetadataFilteringService as RagMetadataFilteringService,
    RangeCondition as RagRangeCondition,
    SetCondition as RagSetCondition,
    filter_candidates as rag_filter_candidates,
    get_metadata_filtering_service as rag_get_metadata_filtering_service,
    reset_metadata_filtering_service as rag_reset_metadata_filtering_service,
    to_qdrant_condition as rag_to_qdrant_condition,
    to_qdrant_filter as rag_to_qdrant_filter,
)
from rag.retrieval import (
    BaseFilterCondition,
    BooleanCondition,
    CandidateMetadataResolver,
    ComparisonOperator,
    CompoundCondition,
    ExactMatchCondition,
    FusedCandidate,
    LogicalOperator,
    MetadataConditionEvaluator,
    MetadataFilter,
    MetadataFilteringConfig,
    MetadataFilteringError,
    MetadataFilteringService,
    RangeCondition,
    SetCondition,
    filter_candidates,
    get_metadata_filtering_service,
    reset_metadata_filtering_service,
    to_qdrant_condition,
    to_qdrant_filter,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Ensure singleton state is cleanly reset before and after each test."""
    reset_metadata_filtering_service()
    rag_reset_metadata_filtering_service()
    yield
    reset_metadata_filtering_service()
    rag_reset_metadata_filtering_service()


def _make_candidate(
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
) -> FusedCandidate:
    """Helper creating test FusedCandidate instance."""
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
        metadata=dict(metadata) if metadata is not None else {},
    )


# ==============================================================================
# 1. Exact-Match Filtering
# ==============================================================================

def test_exact_match_filtering_success():
    """Candidates matching exact string, integer, or attribute values are retained."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements", "author": "Alice"})
    c2 = _make_candidate("c2", metadata={"document_type": "architecture", "author": "Bob"})
    c3 = _make_candidate("c3", metadata={"document_type": "requirements", "author": "Charlie"})

    filter_spec = MetadataFilter(must=[ExactMatchCondition("document_type", "requirements")])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c3"]


def test_exact_match_no_fuzzy_matching():
    """Exact-match does not perform fuzzy or substring matching."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements"})
    c2 = _make_candidate("c2", metadata={"document_type": "requirement"})
    c3 = _make_candidate("c3", metadata={"document_type": "Requirements"})
    c4 = _make_candidate("c4", metadata={"document_type": "system_requirements"})

    filter_spec = MetadataFilter(must=[ExactMatchCondition("document_type", "requirements")])
    results = filter_candidates([c1, c2, c3, c4], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1"]


def test_exact_match_on_top_level_candidate_attributes():
    """Exact-match works on candidate attributes like document_id and document_version_id."""
    c1 = _make_candidate("c1", document_id="D123", document_version_id="V4")
    c2 = _make_candidate("c2", document_id="D999", document_version_id="V4")
    c3 = _make_candidate("c3", document_id="D123", document_version_id="V5")

    filter_spec = MetadataFilter(must=[
        ExactMatchCondition("document_id", "D123"),
        ExactMatchCondition("document_version_id", "V4"),
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1"]


# ==============================================================================
# 2. Multiple-Value / Set Filtering
# ==============================================================================

def test_set_filtering_scalar_value():
    """Candidate is retained when scalar metadata value matches any permitted set element."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements"})
    c2 = _make_candidate("c2", metadata={"document_type": "specification"})
    c3 = _make_candidate("c3", metadata={"document_type": "notes"})

    filter_spec = MetadataFilter(must=[
        SetCondition("document_type", ["requirements", "specification"])
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c2"]


def test_set_filtering_collection_value():
    """Candidate is retained when list/tuple metadata value intersects permitted set."""
    c1 = _make_candidate("c1", metadata={"section_path": ["Architecture", "Security"]})
    c2 = _make_candidate("c2", metadata={"section_path": ["Architecture", "Database"]})
    c3 = _make_candidate("c3", metadata={"section_path": ["Frontend", "UI"]})

    filter_spec = MetadataFilter(must=[
        SetCondition("section_path", ["Security", "Compliance"])
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1"]


# ==============================================================================
# 3. Range Filtering
# ==============================================================================

def test_range_filtering_numerical():
    """Numeric range conditions evaluate with strict boundary behavior."""
    c1 = _make_candidate("c1", metadata={"character_count": 50})
    c2 = _make_candidate("c2", metadata={"character_count": 100})
    c3 = _make_candidate("c3", metadata={"character_count": 250})
    c4 = _make_candidate("c4", metadata={"character_count": 500})
    c5 = _make_candidate("c5", metadata={"character_count": 600})

    # Range [100, 500] inclusive
    filter_spec = MetadataFilter(must=[
        RangeCondition("character_count", gte=100, lte=500)
    ])
    results = filter_candidates([c1, c2, c3, c4, c5], filter_spec=filter_spec)
    assert [c.chunk_id for c in results] == ["c2", "c3", "c4"]

    # Strict inequality (100, 500) exclusive
    exclusive_filter = MetadataFilter(must=[
        RangeCondition("character_count", gt=100, lt=500)
    ])
    ex_results = filter_candidates([c1, c2, c3, c4, c5], filter_spec=exclusive_filter)
    assert [c.chunk_id for c in ex_results] == ["c3"]


def test_range_filtering_dates_and_boundaries():
    """Date comparisons parse ISO format strings and evaluate boundary values chronologically."""
    c1 = _make_candidate("c1", metadata={"date": "2025-12-31"})
    c2 = _make_candidate("c2", metadata={"date": "2026-01-01"})
    c3 = _make_candidate("c3", metadata={"date": "2026-02-15"})
    c4 = _make_candidate("c4", metadata={"date": "2026-03-31"})
    c5 = _make_candidate("c5", metadata={"date": "2026-04-01"})

    filter_spec = MetadataFilter(must=[
        RangeCondition("date", gte="2026-01-01", lte="2026-03-31")
    ])
    results = filter_candidates([c1, c2, c3, c4, c5], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c2", "c3", "c4"]


def test_range_filtering_datetime_objects():
    """Date range filtering correctly compares datetime and date objects."""
    c1 = _make_candidate("c1", metadata={"created_at": datetime(2026, 1, 1, 10, 0)})
    c2 = _make_candidate("c2", metadata={"created_at": date(2026, 1, 1)})
    c3 = _make_candidate("c3", metadata={"created_at": datetime(2025, 12, 31, 23, 59)})

    filter_spec = MetadataFilter(must=[
        RangeCondition("created_at", gte="2026-01-01")
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c2"]


# ==============================================================================
# 4. Boolean Filtering
# ==============================================================================

def test_boolean_filtering_strictness():
    """Boolean filters only match genuine bool values and reject pseudo-boolean strings."""
    c1 = _make_candidate("c1", metadata={"has_code": True})
    c2 = _make_candidate("c2", metadata={"has_code": False})
    c3 = _make_candidate("c3", metadata={"has_code": "true"})
    c4 = _make_candidate("c4", metadata={"has_code": 1})

    filter_spec = MetadataFilter(must=[BooleanCondition("has_code", True)])
    results = filter_candidates([c1, c2, c3, c4], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1"]


# ==============================================================================
# 5. Compound Filtering
# ==============================================================================

def test_compound_and_condition():
    """CompoundCondition with AND requires all sub-conditions to be satisfied."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements", "has_code": True})
    c2 = _make_candidate("c2", metadata={"document_type": "requirements", "has_code": False})
    c3 = _make_candidate("c3", metadata={"document_type": "specification", "has_code": True})

    filter_spec = MetadataFilter(must=[
        CompoundCondition(LogicalOperator.AND, [
            ExactMatchCondition("document_type", "requirements"),
            BooleanCondition("has_code", True),
        ])
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1"]


def test_compound_or_condition():
    """CompoundCondition with OR requires at least one sub-condition to be satisfied."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements", "has_code": False})
    c2 = _make_candidate("c2", metadata={"document_type": "notes", "has_code": True})
    c3 = _make_candidate("c3", metadata={"document_type": "notes", "has_code": False})

    filter_spec = MetadataFilter(must=[
        CompoundCondition(LogicalOperator.OR, [
            ExactMatchCondition("document_type", "requirements"),
            BooleanCondition("has_code", True),
        ])
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c2"]


def test_compound_not_condition():
    """CompoundCondition with NOT excludes candidates matching sub-conditions."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements"})
    c2 = _make_candidate("c2", metadata={"document_type": "deprecated"})
    c3 = _make_candidate("c3", metadata={"document_type": "specification"})

    filter_spec = MetadataFilter(must_not=[
        ExactMatchCondition("document_type", "deprecated")
    ])
    results = filter_candidates([c1, c2, c3], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c3"]


def test_nested_compound_conditions():
    """Nested compound expressions (A AND (B OR C)) evaluate correctly."""
    c1 = _make_candidate("c1", metadata={"dept": "engineering", "level": "senior", "active": True})
    c2 = _make_candidate("c2", metadata={"dept": "engineering", "level": "staff", "active": True})
    c3 = _make_candidate("c3", metadata={"dept": "engineering", "level": "junior", "active": True})
    c4 = _make_candidate("c4", metadata={"dept": "marketing", "level": "senior", "active": True})

    # dept == engineering AND (level == senior OR level == staff)
    filter_spec = MetadataFilter(must=[
        ExactMatchCondition("dept", "engineering"),
        CompoundCondition(LogicalOperator.OR, [
            ExactMatchCondition("level", "senior"),
            ExactMatchCondition("level", "staff"),
        ]),
    ])
    results = filter_candidates([c1, c2, c3, c4], filter_spec=filter_spec)

    assert [c.chunk_id for c in results] == ["c1", "c2"]


# ==============================================================================
# 6. Missing Metadata Handling
# ==============================================================================

def test_missing_metadata_never_satisfies_constraints():
    """Missing metadata must evaluate to False and never silently satisfy constraints."""
    c1 = _make_candidate("c1", metadata={"document_type": "requirements"})
    c2 = _make_candidate("c2", metadata={})  # Missing document_type
    c3 = _make_candidate("c3", metadata={"other_field": 123})

    # Exact match on missing field
    exact_res = filter_candidates([c1, c2, c3], MetadataFilter(must=[ExactMatchCondition("document_type", "requirements")]))
    assert [c.chunk_id for c in exact_res] == ["c1"]

    # Range on missing field
    range_res = filter_candidates([c1, c2, c3], MetadataFilter(must=[RangeCondition("character_count", gte=0)]))
    assert len(range_res) == 0

    # Set on missing field
    set_res = filter_candidates([c1, c2, c3], MetadataFilter(must=[SetCondition("document_type", ["requirements"])]))
    assert [c.chunk_id for c in set_res] == ["c1"]

    # Boolean on missing field
    bool_res = filter_candidates([c1, c2, c3], MetadataFilter(must=[BooleanCondition("has_code", False)]))
    assert len(bool_res) == 0


# ==============================================================================
# 7. Multiple Simultaneous Constraints (Realistic Combination)
# ==============================================================================

def test_multiple_simultaneous_constraints_realistic():
    """document_type = requirements AND document_version_id = V4 AND date >= 2026-01-01."""
    # Satisfies all
    c_pass = _make_candidate(
        "c_pass",
        document_version_id="V4",
        metadata={"document_type": "requirements", "date": "2026-01-15", "has_code": True},
    )
    # Fails document_type
    c_fail_type = _make_candidate(
        "c_fail_type",
        document_version_id="V4",
        metadata={"document_type": "architecture", "date": "2026-01-15", "has_code": True},
    )
    # Fails document_version_id
    c_fail_ver = _make_candidate(
        "c_fail_ver",
        document_version_id="V3",
        metadata={"document_type": "requirements", "date": "2026-01-15", "has_code": True},
    )
    # Fails date
    c_fail_date = _make_candidate(
        "c_fail_date",
        document_version_id="V4",
        metadata={"document_type": "requirements", "date": "2025-12-15", "has_code": True},
    )
    # Fails multiple constraints
    c_fail_multi = _make_candidate(
        "c_fail_multi",
        document_version_id="V1",
        metadata={"document_type": "unknown", "date": "2020-01-01"},
    )

    filter_spec = MetadataFilter(must=[
        ExactMatchCondition("document_type", "requirements"),
        ExactMatchCondition("document_version_id", "V4"),
        RangeCondition("date", gte="2026-01-01"),
    ])

    results = filter_candidates(
        [c_pass, c_fail_type, c_fail_ver, c_fail_date, c_fail_multi],
        filter_spec=filter_spec,
    )

    assert [c.chunk_id for c in results] == ["c_pass"]


# ==============================================================================
# 8. Score, Rank, and Identity Invariant Preservation
# ==============================================================================

def test_preserves_fusion_score_rank_and_identity():
    """Metadata filtering strictly preserves chunk_id, fused_score, rank, and provenance."""
    orig = _make_candidate(
        chunk_id="chk-stable-1",
        score=0.042857,
        rank=3,
        project_id="proj-alpha",
        document_id="doc-beta",
        document_version_id="v-gamma",
        dense_rank=2,
        sparse_rank=4,
        dense_score=0.88,
        sparse_score=24.5,
        metadata={"target": "yes"},
    )

    results = filter_candidates([orig], filter_spec={"target": "yes"})
    assert len(results) == 1
    retained = results[0]

    assert retained.chunk_id == "chk-stable-1"
    assert retained.score == orig.score
    assert retained.fused_score == orig.fused_score
    assert retained.rank == orig.rank
    assert retained.dense_rank == orig.dense_rank
    assert retained.sparse_rank == orig.sparse_rank
    assert retained.dense_score == orig.dense_score
    assert retained.sparse_score == orig.sparse_score
    assert retained.project_id == orig.project_id
    assert retained.document_id == orig.document_id
    assert retained.document_version_id == orig.document_version_id


# ==============================================================================
# 9. Project Isolation
# ==============================================================================

def test_project_isolation_defensively_discards_foreign_candidates():
    """Candidates from another project are discarded even if metadata matches."""
    c1 = _make_candidate("c1", project_id="proj-A", metadata={"category": "finance"})
    c2 = _make_candidate("c2", project_id="proj-B", metadata={"category": "finance"})

    results = filter_candidates([c1, c2], filter_spec={"category": "finance"}, project_id="proj-A")

    assert [c.chunk_id for c in results] == ["c1"]


# ==============================================================================
# 10. Empty Candidates & Empty Filter Behavior
# ==============================================================================

def test_empty_candidates_returns_empty_list():
    """Empty candidate sequence returns empty list without error."""
    assert filter_candidates([], filter_spec={"a": "b"}) == []


def test_empty_filter_passes_all_candidates_unchanged():
    """None or empty filter passes all candidates preserving order."""
    c1 = _make_candidate("c1", rank=1, score=0.03)
    c2 = _make_candidate("c2", rank=2, score=0.02)

    assert filter_candidates([c1, c2], filter_spec=None) == [c1, c2]
    assert filter_candidates([c1, c2], filter_spec={}) == [c1, c2]
    assert filter_candidates([c1, c2], filter_spec=MetadataFilter()) == [c1, c2]


# ==============================================================================
# 11. External Metadata Lookup Resolution
# ==============================================================================

def test_external_metadata_lookup():
    """Metadata provided via external dict or callable is resolved when absent from candidate."""
    c1 = _make_candidate("c1", metadata={})
    c2 = _make_candidate("c2", metadata={})

    lookup = {
        "c1": {"classification": "confidential"},
        "c2": {"classification": "public"},
    }

    results = filter_candidates(
        [c1, c2],
        filter_spec={"classification": "confidential"},
        metadata_lookup=lookup,
    )

    assert [c.chunk_id for c in results] == ["c1"]


# ==============================================================================
# 12. Dictionary Parsing & Syntactic Sugar
# ==============================================================================

def test_metadata_filter_from_dict_flat_and_structured():
    """MetadataFilter.from_dict correctly parses flat and structured dictionaries."""
    # Flat dict
    flat = MetadataFilter.from_dict({
        "doc_type": "specs",
        "tags": ["a", "b"],
        "count": {"gte": 10, "lte": 50},
        "is_active": True,
    })
    assert len(flat.must) == 4

    # Structured dict with must and should
    structured = MetadataFilter.from_dict({
        "must": [{"field": "doc_type", "value": "specs"}],
        "should": [
            {"field": "priority", "value": "high"},
            {"field": "priority", "value": "urgent"},
        ],
        "must_not": [{"field": "status", "value": "archived"}],
    })
    assert len(structured.must) == 1
    assert len(structured.should) == 2
    assert len(structured.must_not) == 1


# ==============================================================================
# 13. Native Qdrant Filter Translation
# ==============================================================================

def test_to_qdrant_filter_translation():
    """to_qdrant_filter compiles domain filter to Qdrant models.Filter with project_id."""
    domain_filter = MetadataFilter(
        must=[
            ExactMatchCondition("doc_type", "spec"),
            SetCondition("tags", ["python", "ai"]),
            RangeCondition("score", gte=0.5, lte=1.0),
            BooleanCondition("is_active", True),
        ],
        should=[
            ExactMatchCondition("tier", "gold"),
        ],
        must_not=[
            ExactMatchCondition("status", "draft"),
        ],
    )

    qdrant_filt = to_qdrant_filter(domain_filter, project_id="proj-999")

    # Project condition + 4 must conditions = 5
    assert len(qdrant_filt.must) == 5
    assert qdrant_filt.must[0].key == "project_id"
    assert qdrant_filt.must[0].match.value == "proj-999"
    assert len(qdrant_filt.should) == 1
    assert len(qdrant_filt.must_not) == 1


# ==============================================================================
# 14. Service Facades, Re-exports, and Lifecycle
# ==============================================================================

def test_service_lifecycle_and_rag_reexports():
    """Verifies get/reset lifecycle and confirms rag.retrieval re-exports identical classes."""
    svc1 = get_metadata_filtering_service()
    svc2 = rag_get_metadata_filtering_service()
    assert svc1 is svc2

    reset_metadata_filtering_service()
    svc3 = get_metadata_filtering_service()
    assert svc3 is not svc1

    # Verify RAG facade compatibility
    assert RagMetadataFilteringService is MetadataFilteringService
    assert RagMetadataFilter is MetadataFilter
    assert RagExactMatchCondition is ExactMatchCondition
    assert RagSetCondition is SetCondition
    assert RagRangeCondition is RangeCondition
    assert RagBooleanCondition is BooleanCondition
    assert RagCompoundCondition is CompoundCondition
    assert RagMetadataConditionEvaluator is MetadataConditionEvaluator
    assert RagCandidateMetadataResolver is CandidateMetadataResolver
    assert RagComparisonOperator is ComparisonOperator
    assert RagLogicalOperator is LogicalOperator
    assert RagMetadataFilteringConfig is MetadataFilteringConfig
    assert RagMetadataFilteringError is MetadataFilteringError


# ==============================================================================
# 15. Invalid Inputs and Error Handling
# ==============================================================================

def test_invalid_filter_inputs_raise_errors():
    """Invalid condition arguments raise InvalidFilterError."""
    with pytest.raises(InvalidFilterError):
        ExactMatchCondition("", "val")

    with pytest.raises(InvalidFilterError):
        ExactMatchCondition("field", None)

    with pytest.raises(InvalidFilterError):
        SetCondition("field", [])

    with pytest.raises(InvalidFilterError):
        RangeCondition("field")  # No bounds specified

    with pytest.raises(InvalidFilterError):
        BooleanCondition("field", "not_a_bool")  # type: ignore

    with pytest.raises(InvalidFilterError):
        to_qdrant_filter(None, project_id="")  # Empty project_id
