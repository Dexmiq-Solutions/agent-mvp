"""Unit tests for the Chunk Fetching / Hydration stage in the retrieval pipeline."""

import asyncio
from typing import Any, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exceptions.retrieval import (
    ChunkHydrationError,
    ChunkHydrationValidationError,
    ChunkNotFoundError,
    DatabaseRetrievalError,
    RetrievalError,
)
from models.chunk import ChunkModel
from rag.retrieval import (
    BaseChunkRepository as RagBaseChunkRepository,
    ChunkHydrationConfig as RagChunkHydrationConfig,
    ChunkHydrationError as RagChunkHydrationError,
    ChunkHydrationService as RagChunkHydrationService,
    ChunkHydrationValidationError as RagChunkHydrationValidationError,
    ChunkNotFoundError as RagChunkNotFoundError,
    DatabaseRetrievalError as RagDatabaseRetrievalError,
    HydratedCandidate as RagHydratedCandidate,
    HydratedChunk as RagHydratedChunk,
    HydratedSearchCandidate as RagHydratedSearchCandidate,
    SQLAlchemyChunkRepository as RagSQLAlchemyChunkRepository,
    get_chunk_hydration_service as rag_get_chunk_hydration_service,
    get_chunk_repository as rag_get_chunk_repository,
    hydrate_candidates as rag_hydrate_candidates,
    reset_chunk_hydration_service as rag_reset_chunk_hydration_service,
    reset_chunk_repository as rag_reset_chunk_repository,
)
from retrieval import (
    BaseChunkRepository,
    ChunkHydrationConfig,
    ChunkHydrationService,
    FusedCandidate,
    HydratedCandidate,
    HydratedChunk,
    HydratedSearchCandidate,
    RerankedCandidate,
    SQLAlchemyChunkRepository,
    get_chunk_hydration_service,
    get_chunk_repository,
    hydrate_candidates,
    reset_chunk_hydration_service,
    reset_chunk_repository,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Reset hydration singletons before and after each test."""
    reset_chunk_hydration_service()
    reset_chunk_repository()
    rag_reset_chunk_hydration_service()
    rag_reset_chunk_repository()
    yield
    reset_chunk_hydration_service()
    reset_chunk_repository()
    rag_reset_chunk_hydration_service()
    rag_reset_chunk_repository()


def _make_reranked_candidate(
    chunk_id: str,
    rerank_score: float = 0.95,
    rank: int = 1,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
    fusion_score: float | None = 0.035,
    dense_rank: int | None = 1,
    sparse_rank: int | None = 2,
    dense_score: float | None = 0.88,
    sparse_score: float | None = 15.4,
    metadata: dict | None = None,
) -> RerankedCandidate:
    """Helper to construct a test RerankedCandidate instance."""
    return RerankedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        rerank_score=rerank_score,
        rank=rank,
        fusion_score=fusion_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=dense_score,
        sparse_score=sparse_score,
        document_version_id=document_version_id,
        metadata=dict(metadata or {}),
    )


def _make_fused_candidate(
    chunk_id: str,
    score: float = 0.035,
    rank: int = 1,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
    metadata: dict | None = None,
) -> FusedCandidate:
    """Helper to construct a test FusedCandidate instance."""
    return FusedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        score=score,
        rank=rank,
        document_version_id=document_version_id,
        metadata=dict(metadata or {}),
    )


def _make_chunk_model(
    chunk_id: str,
    content: str = "Authoritative stored chunk text",
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
    contextual_content: str | None = None,
    chunk_index: int = 0,
    heading: str | None = "Introduction",
    heading_level: int | None = 1,
    section_path: list[str] | None = None,
    parent_element_id: str | None = "elem-0",
    parent_chunk_id: str | None = None,
    element_types: list[str] | None = None,
    metadata: dict | None = None,
) -> ChunkModel:
    """Helper to construct a ChunkModel instance."""
    return ChunkModel(
        chunk_id=chunk_id,
        project_id=project_id,
        document_id=document_id,
        document_version_id=document_version_id,
        content=content,
        contextual_content=contextual_content,
        chunk_index=chunk_index,
        heading=heading,
        heading_level=heading_level,
        section_path=section_path or ["Overview", "Introduction"],
        parent_element_id=parent_element_id,
        parent_chunk_id=parent_chunk_id,
        element_types=element_types or ["heading", "paragraph"],
        chunk_metadata=dict(metadata or {"category": "technical", "source": "guide.pdf"}),
    )


class FakeChunkRepository(BaseChunkRepository):
    """In-memory mock chunk repository tracking queries and supporting deterministic returns."""

    def __init__(
        self,
        records: Optional[Sequence[ChunkModel]] = None,
        exception_to_raise: Optional[Exception] = None,
    ) -> None:
        self.records_by_id: dict[str, ChunkModel] = {r.chunk_id: r for r in (records or [])}
        self.exception_to_raise = exception_to_raise
        self.call_count = 0
        self.last_project_id: Optional[str] = None
        self.last_chunk_ids: list[str] = []

    async def fetch_chunks(
        self,
        project_id: str,
        chunk_ids: Sequence[str],
        session: Optional[AsyncSession] = None,
    ) -> dict[str, ChunkModel]:
        self.call_count += 1
        self.last_project_id = project_id
        self.last_chunk_ids = list(chunk_ids)

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        # Scope strictly to project_id and requested chunk_ids
        matched: dict[str, ChunkModel] = {}
        for cid in chunk_ids:
            record = self.records_by_id.get(cid)
            if record and record.project_id == project_id:
                matched[cid] = record

        return matched


# ==============================================================================
# 1. Normal Flow & Field Resolution Tests
# ==============================================================================


@pytest.mark.anyio
async def test_normal_hydration_flow():
    """Verify valid ranked candidates resolve to HydratedCandidate records with stored content."""
    candidates = [
        _make_reranked_candidate("chk-1", rerank_score=0.96, rank=1),
        _make_reranked_candidate("chk-2", rerank_score=0.89, rank=2),
    ]

    records = [
        _make_chunk_model("chk-1", content="Chunk 1 authoritative text", chunk_index=0),
        _make_chunk_model("chk-2", content="Chunk 2 authoritative text", chunk_index=1),
    ]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates=candidates, project_id="proj-1")

    assert len(results) == 2
    assert repo.call_count == 1
    assert repo.last_chunk_ids == ["chk-1", "chk-2"]

    r1, r2 = results
    assert isinstance(r1, HydratedCandidate)
    assert r1.chunk_id == "chk-1"
    assert r1.content == "Chunk 1 authoritative text"
    assert r1.text == "Chunk 1 authoritative text"
    assert r1.chunk_text == "Chunk 1 authoritative text"
    assert r1.rerank_score == 0.96
    assert r1.score == 0.96
    assert r1.rank == 1
    assert r1.fusion_score == 0.035
    assert r1.dense_rank == 1
    assert r1.sparse_rank == 2
    assert r1.dense_score == 0.88
    assert r1.sparse_score == 15.4
    assert r1.heading == "Introduction"
    assert r1.section_path == ("Overview", "Introduction")

    assert r2.chunk_id == "chk-2"
    assert r2.content == "Chunk 2 authoritative text"
    assert r2.rerank_score == 0.89
    assert r2.rank == 2


@pytest.mark.anyio
async def test_fused_candidate_input_hydration():
    """Verify FusedCandidate (when reranking is bypassed) hydrates cleanly preserving fused scores."""
    candidates = [
        _make_fused_candidate("chk-10", score=0.045, rank=1),
    ]
    records = [
        _make_chunk_model("chk-10", content="Fused chunk content"),
    ]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates=candidates)

    assert len(results) == 1
    h = results[0]
    assert h.chunk_id == "chk-10"
    assert h.content == "Fused chunk content"
    assert h.fusion_score == 0.045
    assert h.score == 0.045
    assert h.rerank_score is None
    assert h.rank == 1


# ==============================================================================
# 2. Batched Lookup & N+1 Prevention Tests
# ==============================================================================


@pytest.mark.anyio
async def test_batched_lookup_single_database_query():
    """Verify that multiple candidates produce exactly one batched repository call."""
    cands = [
        _make_reranked_candidate(f"chk-{i}", rerank_score=1.0 - (i * 0.05), rank=i)
        for i in range(1, 11)
    ]
    records = [
        _make_chunk_model(f"chk-{i}", content=f"Text for chunk {i}", chunk_index=i)
        for i in range(1, 11)
    ]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates=cands, project_id="proj-1")

    # Critical: exactly 1 database call for all 10 chunks (No N+1 queries)
    assert repo.call_count == 1
    assert len(results) == 10
    assert len(repo.last_chunk_ids) == 10


# ==============================================================================
# 3. Reranking Order Preservation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_rerank_order_preserved_when_db_returns_out_of_order():
    """Verify hydrated results preserve original reranking order even when DB returns out-of-order."""
    # Reranking order: chk-47, chk-101, chk-19, chk-832
    candidates = [
        _make_reranked_candidate("chk-47", rerank_score=0.92, rank=1),
        _make_reranked_candidate("chk-101", rerank_score=0.87, rank=2),
        _make_reranked_candidate("chk-19", rerank_score=0.81, rank=3),
        _make_reranked_candidate("chk-832", rerank_score=0.75, rank=4),
    ]

    # Database returns them in completely different/scrambled order
    db_records = [
        _make_chunk_model("chk-19", content="Content 19"),
        _make_chunk_model("chk-832", content="Content 832"),
        _make_chunk_model("chk-47", content="Content 47"),
        _make_chunk_model("chk-101", content="Content 101"),
    ]

    repo = FakeChunkRepository(db_records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates=candidates, project_id="proj-1")

    # Output order MUST strictly match reranking input order
    assert [r.chunk_id for r in results] == ["chk-47", "chk-101", "chk-19", "chk-832"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert [r.content for r in results] == ["Content 47", "Content 101", "Content 19", "Content 832"]


# ==============================================================================
# 4. Contextual Content & Metadata Preservation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_contextual_content_attached():
    """Verify persisted contextual content is attached to HydratedCandidate."""
    candidate = _make_reranked_candidate("chk-ctx", rerank_score=0.90)
    record = _make_chunk_model(
        "chk-ctx",
        content="Primary text",
        contextual_content="Context prefix description\n\nPrimary text",
    )

    repo = FakeChunkRepository([record])
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate([candidate], project_id="proj-1")

    assert len(results) == 1
    assert results[0].content == "Primary text"
    assert results[0].contextual_content == "Context prefix description\n\nPrimary text"


@pytest.mark.anyio
async def test_metadata_merging_and_provenance_preservation():
    """Verify candidate retrieval metadata and stored chunk metadata are merged without loss."""
    candidate = _make_reranked_candidate(
        "chk-meta",
        rerank_score=0.91,
        metadata={"query_type": "original", "retrieval_branch": "dense"},
    )
    record = _make_chunk_model(
        "chk-meta",
        metadata={"char_count": 450, "token_count": 95, "has_table": False},
    )

    repo = FakeChunkRepository([record])
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate([candidate], project_id="proj-1")

    assert len(results) == 1
    meta = results[0].metadata
    # Stored record metadata preserved
    assert meta["char_count"] == 450
    assert meta["has_table"] is False
    # Retrieval provenance metadata preserved
    assert meta["query_type"] == "original"
    assert meta["retrieval_branch"] == "dense"


# ==============================================================================
# 5. Project Isolation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_cross_project_candidate_discarded():
    """Verify candidates from another project are defensively discarded before query."""
    candidates = [
        _make_reranked_candidate("chk-ok", project_id="proj-tenant-a"),
        _make_reranked_candidate("chk-leaked", project_id="proj-tenant-b"),
    ]
    records = [
        _make_chunk_model("chk-ok", project_id="proj-tenant-a"),
        _make_chunk_model("chk-leaked", project_id="proj-tenant-b"),
    ]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates, project_id="proj-tenant-a")

    # Leaked candidate discarded; repository only queried with 'chk-ok'
    assert len(results) == 1
    assert results[0].chunk_id == "chk-ok"
    assert repo.last_chunk_ids == ["chk-ok"]
    assert repo.last_project_id == "proj-tenant-a"


@pytest.mark.anyio
async def test_cannot_hydrate_another_projects_record():
    """Verify repository scopes query to active project so cross-tenant record is never returned."""
    candidate = _make_reranked_candidate("chk-cross", project_id="proj-1")
    # Record in DB belongs to proj-2
    record_other = _make_chunk_model("chk-cross", project_id="proj-2")

    repo = FakeChunkRepository([record_other])
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate([candidate], project_id="proj-1")

    # Matched count is 0 because DB lookup scoped to proj-1 returns empty
    assert len(results) == 0


# ==============================================================================
# 6. Document & Version Identity Verification Tests
# ==============================================================================


@pytest.mark.anyio
async def test_document_id_mismatch_skipped():
    """Verify chunk is discarded if stored document_id does not match candidate document_id."""
    candidate = _make_reranked_candidate("chk-doc", document_id="doc-correct")
    record = _make_chunk_model("chk-doc", document_id="doc-corrupted")

    repo = FakeChunkRepository([record])
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate([candidate], project_id="proj-1")

    assert len(results) == 0


@pytest.mark.anyio
async def test_version_identity_mismatch_skipped_when_configured():
    """Verify chunk is discarded if document_version_id mismatches when verify_version_identity=True."""
    candidate = _make_reranked_candidate("chk-ver", document_version_id="v-2.0")
    record = _make_chunk_model("chk-ver", document_version_id="v-1.0")

    repo = FakeChunkRepository([record])
    config = ChunkHydrationConfig(verify_version_identity=True)
    service = ChunkHydrationService(repository=repo, config=config)

    results = await service.hydrate([candidate], project_id="proj-1")

    assert len(results) == 0


@pytest.mark.anyio
async def test_version_identity_check_can_be_disabled():
    """Verify version mismatch is accepted if verify_version_identity=False."""
    candidate = _make_reranked_candidate("chk-ver", document_version_id="v-2.0")
    record = _make_chunk_model("chk-ver", document_version_id="v-1.0")

    repo = FakeChunkRepository([record])
    config = ChunkHydrationConfig(verify_version_identity=False)
    service = ChunkHydrationService(repository=repo, config=config)

    results = await service.hydrate([candidate], project_id="proj-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chk-ver"


# ==============================================================================
# 7. Empty Input & Deduplication Tests
# ==============================================================================


@pytest.mark.anyio
async def test_empty_candidates_zero_db_calls():
    """Verify empty input returns empty list immediately with zero database interactions."""
    repo = FakeChunkRepository()
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates=[], project_id="proj-1")

    assert results == []
    assert repo.call_count == 0


@pytest.mark.anyio
async def test_duplicate_chunk_ids_deduplicated():
    """Verify duplicate candidate chunk IDs are defensively deduplicated preserving higher rank."""
    candidates = [
        _make_reranked_candidate("chk-dup", rerank_score=0.95, rank=1),
        _make_reranked_candidate("chk-dup", rerank_score=0.70, rank=5),
    ]
    records = [
        _make_chunk_model("chk-dup", content="Deduplicated text"),
    ]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await service.hydrate(candidates, project_id="proj-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chk-dup"
    assert results[0].rank == 1
    assert results[0].rerank_score == 0.95
    assert repo.last_chunk_ids == ["chk-dup"]


# ==============================================================================
# 8. Missing Chunks & Partial Results Tests
# ==============================================================================


@pytest.mark.anyio
async def test_partial_results_permissive_mode():
    """Verify missing chunks are omitted while surviving chunks are returned in order."""
    candidates = [
        _make_reranked_candidate("chk-exists-1", rank=1),
        _make_reranked_candidate("chk-missing", rank=2),
        _make_reranked_candidate("chk-exists-2", rank=3),
    ]
    records = [
        _make_chunk_model("chk-exists-1", content="Text 1"),
        _make_chunk_model("chk-exists-2", content="Text 2"),
    ]

    repo = FakeChunkRepository(records)
    config = ChunkHydrationConfig(fail_on_missing=False)
    service = ChunkHydrationService(repository=repo, config=config)

    results = await service.hydrate(candidates, project_id="proj-1")

    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["chk-exists-1", "chk-exists-2"]
    assert [r.rank for r in results] == [1, 3]


@pytest.mark.anyio
async def test_missing_chunks_strict_mode_raises():
    """Verify ChunkNotFoundError is raised when fail_on_missing=True."""
    candidates = [
        _make_reranked_candidate("chk-exists", rank=1),
        _make_reranked_candidate("chk-missing-1", rank=2),
        _make_reranked_candidate("chk-missing-2", rank=3),
    ]
    records = [
        _make_chunk_model("chk-exists"),
    ]

    repo = FakeChunkRepository(records)
    config = ChunkHydrationConfig(fail_on_missing=True)
    service = ChunkHydrationService(repository=repo, config=config)

    with pytest.raises(ChunkNotFoundError) as exc_info:
        await service.hydrate(candidates, project_id="proj-1")

    err = exc_info.value
    assert "missing" in str(err).lower()
    assert set(err.missing_chunk_ids) == {"chk-missing-1", "chk-missing-2"}
    assert err.project_id == "proj-1"


# ==============================================================================
# 9. Database Error Handling Tests
# ==============================================================================


@pytest.mark.anyio
async def test_database_failure_raises_database_retrieval_error():
    """Verify database errors are not silently swallowed into empty results."""
    candidates = [_make_reranked_candidate("chk-1")]
    db_error = OperationalError("SELECT * FROM chunks", {}, Exception("Connection refused"))

    repo = FakeChunkRepository(exception_to_raise=db_error)
    service = ChunkHydrationService(repository=repo)

    with pytest.raises(DatabaseRetrievalError) as exc_info:
        await service.hydrate(candidates, project_id="proj-1")

    assert "Connection refused" in str(exc_info.value) or "OperationalError" in str(exc_info.value)
    assert exc_info.value.original_error is db_error


# ==============================================================================
# 10. Input Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_invalid_input_types_raise_validation_error():
    """Verify non-sequence or invalid project_id inputs raise ChunkHydrationValidationError."""
    service = ChunkHydrationService(repository=FakeChunkRepository())

    # String instead of sequence
    with pytest.raises(ChunkHydrationValidationError):
        await service.hydrate("invalid_string_candidate")  # type: ignore

    # None candidates
    with pytest.raises(ChunkHydrationValidationError):
        await service.hydrate(None)  # type: ignore

    # Empty project_id string
    with pytest.raises(ChunkHydrationValidationError):
        await service.hydrate([_make_reranked_candidate("chk-1")], project_id="   ")

    # Missing project_id everywhere
    cand_no_pid = RerankedCandidate(
        chunk_id="chk-1",
        document_id="doc-1",
        project_id="",
        rerank_score=0.9,
        rank=1,
    )
    with pytest.raises(ChunkHydrationValidationError):
        await service.hydrate([cand_no_pid], project_id=None)


# ==============================================================================
# 11. SQLAlchemy Repository Direct Unit Tests
# ==============================================================================


@pytest.mark.anyio
async def test_sqlalchemy_chunk_repository_fetch():
    """Verify SQLAlchemyChunkRepository executes batched SELECT and maps results."""
    mock_record_1 = _make_chunk_model("chk-1", content="Text 1")
    mock_record_2 = _make_chunk_model("chk-2", content="Text 2")

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [mock_record_1, mock_record_2]

    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__.return_value = mock_session
    mock_maker.return_value.__aexit__.return_value = None

    repo = SQLAlchemyChunkRepository(session_maker=mock_maker)

    result_map = await repo.fetch_chunks(
        project_id="proj-1",
        chunk_ids=["chk-1", "chk-2"],
    )

    assert len(result_map) == 2
    assert result_map["chk-1"].content == "Text 1"
    assert result_map["chk-2"].content == "Text 2"
    mock_session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_sqlalchemy_chunk_repository_empty_ids():
    """Verify SQLAlchemyChunkRepository short-circuits on empty IDs."""
    mock_maker = MagicMock()
    repo = SQLAlchemyChunkRepository(session_maker=mock_maker)

    result_map = await repo.fetch_chunks(project_id="proj-1", chunk_ids=[])
    assert result_map == {}
    mock_maker.assert_not_called()


@pytest.mark.anyio
async def test_sqlalchemy_chunk_repository_db_exception():
    """Verify SQLAlchemyChunkRepository wraps SQLAlchemyError into DatabaseRetrievalError."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__.return_value = mock_session
    mock_maker.return_value.__aexit__.return_value = None

    repo = SQLAlchemyChunkRepository(session_maker=mock_maker)

    with pytest.raises(DatabaseRetrievalError) as exc_info:
        await repo.fetch_chunks(project_id="proj-1", chunk_ids=["chk-1"])

    assert "Database connection lost" in str(exc_info.value)


# ==============================================================================
# 12. Functional Convenience Entrypoint Tests
# ==============================================================================


@pytest.mark.anyio
async def test_hydrate_candidates_functional_entrypoint():
    """Verify hydrate_candidates functional helper coordinates with service."""
    candidates = [_make_reranked_candidate("chk-func", rerank_score=0.88, rank=1)]
    records = [_make_chunk_model("chk-func", content="Functional text")]

    repo = FakeChunkRepository(records)
    service = ChunkHydrationService(repository=repo)

    results = await hydrate_candidates(candidates, project_id="proj-1", service=service)

    assert len(results) == 1
    assert results[0].chunk_id == "chk-func"
    assert results[0].content == "Functional text"


# ==============================================================================
# 13. Package Re-export Parity Tests
# ==============================================================================


def test_package_reexport_parity():
    """Verify all hydration models, exceptions, and helpers are re-exported consistently."""
    assert HydratedCandidate is RagHydratedCandidate
    assert HydratedSearchCandidate is RagHydratedSearchCandidate
    assert HydratedChunk is RagHydratedChunk
    assert ChunkHydrationConfig is RagChunkHydrationConfig
    assert BaseChunkRepository is RagBaseChunkRepository
    assert SQLAlchemyChunkRepository is RagSQLAlchemyChunkRepository
    assert ChunkHydrationService is RagChunkHydrationService
    assert get_chunk_hydration_service is rag_get_chunk_hydration_service
    assert hydrate_candidates is rag_hydrate_candidates
    assert reset_chunk_hydration_service is rag_reset_chunk_hydration_service
    assert get_chunk_repository is rag_get_chunk_repository
    assert reset_chunk_repository is rag_reset_chunk_repository
    assert ChunkHydrationError is RagChunkHydrationError
    assert ChunkHydrationValidationError is RagChunkHydrationValidationError
    assert ChunkNotFoundError is RagChunkNotFoundError
    assert DatabaseRetrievalError is RagDatabaseRetrievalError
    assert issubclass(ChunkHydrationError, RetrievalError)
