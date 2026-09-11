"""Unit tests for Vector Search component of the RAG retrieval pipeline."""

import math
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.config import Settings
from exceptions.retrieval import RetrievalError, VectorRetrievalError
from exceptions.vector import (
    CollectionNotFoundError,
    VectorSearchError,
    VectorStoreAuthenticationError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from rag.retrieval import (
    EmbeddedQuery as RagEmbeddedQuery,
    EmbeddedQuerySet as RagEmbeddedQuerySet,
    VectorRetrievalError as RagVectorRetrievalError,
    VectorSearchCandidate as RagVectorSearchCandidate,
    VectorSearchConfig as RagVectorSearchConfig,
    VectorSearchService as RagVectorSearchService,
    get_vector_search_service as rag_get_vector_search_service,
    reset_vector_search_service as rag_reset_vector_search_service,
    search_vectors as rag_search_vectors,
)
from retrieval import (
    EmbeddedQuery,
    EmbeddedQuerySet,
    VectorRetrievalError,
    VectorSearchCandidate,
    VectorSearchConfig,
    VectorSearchService,
    get_vector_search_service,
    reset_vector_search_service,
    search_vectors,
)
from storage.vector import BaseVectorStore, VectorPayload, VectorSearchResult


class MockVectorStore(BaseVectorStore):
    """Deterministic mock vector store implementing BaseVectorStore."""

    def __init__(
        self,
        collection_name: str = "document_chunks",
        canned_batch_results: list[list[VectorSearchResult]] | None = None,
        should_fail_with: Exception | None = None,
    ) -> None:
        self._collection_name = collection_name
        self.canned_batch_results = canned_batch_results or []
        self.should_fail_with = should_fail_with
        self.search_calls: list[dict] = []
        self.search_batch_calls: list[dict] = []

    @property
    def collection_name(self) -> str:
        return self._collection_name

    async def ensure_collection_exists(self, vector_size=None, distance="Cosine") -> bool:
        return True

    async def upsert(self, records) -> int:
        return len(records)

    async def delete(self, point_ids) -> int:
        return len(point_ids)

    async def delete_by_filter(self, project_id, document_id=None, document_version_id=None) -> bool:
        return True

    async def search(
        self,
        query_vector,
        project_id,
        limit=10,
        filter_metadata=None,
        score_threshold=None,
        with_vectors=False,
    ) -> list[VectorSearchResult]:
        self.search_calls.append({
            "query_vector": query_vector,
            "project_id": project_id,
            "limit": limit,
            "filter_metadata": filter_metadata,
            "score_threshold": score_threshold,
        })
        if self.should_fail_with:
            raise self.should_fail_with
        if self.canned_batch_results:
            return self.canned_batch_results[0]
        return []

    async def search_batch(
        self,
        query_vectors,
        project_id,
        limit=10,
        filter_metadata=None,
        score_threshold=None,
        with_vectors=False,
    ) -> list[list[VectorSearchResult]]:
        self.search_batch_calls.append({
            "query_vectors": query_vectors,
            "project_id": project_id,
            "limit": limit,
            "filter_metadata": filter_metadata,
            "score_threshold": score_threshold,
        })
        if self.should_fail_with:
            raise self.should_fail_with
        if self.canned_batch_results:
            return self.canned_batch_results
        return [[] for _ in query_vectors]


@pytest.fixture(autouse=True)
def reset_service():
    """Reset singleton vector search service between tests."""
    reset_vector_search_service()
    yield
    reset_vector_search_service()


def make_sample_candidate(
    point_id: str = "p1",
    score: float = 0.92,
    project_id: str = "proj_test",
    document_id: str = "doc_1",
    chunk_id: str = "chunk_1",
    document_version_id: str | None = "v1",
) -> VectorSearchResult:
    """Helper to build a lower-level storage VectorSearchResult."""
    return VectorSearchResult(
        id=point_id,
        score=score,
        payload=VectorPayload(
            project_id=project_id,
            document_id=document_id,
            chunk_id=chunk_id,
            document_version_id=document_version_id,
        ),
    )


# ==============================================================================
# 1. Basic Vector Search Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_basic_success():
    """Verify basic vector search returns expected VectorSearchCandidate objects."""
    item = make_sample_candidate(chunk_id="chk_100", score=0.89)
    mock_store = MockVectorStore(canned_batch_results=[[item]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="what is rag?", vector=[0.1, 0.2, 0.3], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")

    assert len(results) == 1
    candidate = results[0]
    assert isinstance(candidate, VectorSearchCandidate)
    assert candidate.chunk_id == "chk_100"
    assert candidate.document_id == "doc_1"
    assert candidate.project_id == "proj_test"
    assert candidate.score == 0.89
    assert candidate.query_type == "original"
    assert candidate.document_version_id == "v1"

    # Verify serialization
    d = candidate.to_dict()
    assert d["chunk_id"] == "chk_100"
    assert d["score"] == 0.89
    assert d["query_type"] == "original"


@pytest.mark.anyio
async def test_search_single_original_query():
    """Verify that when only original query is present, it is searched correctly."""
    item = make_sample_candidate(chunk_id="chk_orig")
    mock_store = MockVectorStore(canned_batch_results=[[item]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="user query", vector=[0.5, 0.6], query_type="original"),
        transformed=None,
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")

    assert len(results) == 1
    assert results[0].chunk_id == "chk_orig"
    assert results[0].query_type == "original"

    # Verify mock store was called with single query vector
    assert len(mock_store.search_batch_calls) == 1
    call = mock_store.search_batch_calls[0]
    assert len(call["query_vectors"]) == 1
    assert call["query_vectors"][0] == [0.5, 0.6]
    assert call["project_id"] == "proj_test"


@pytest.mark.anyio
async def test_search_original_and_transformed_queries():
    """Verify original and transformed queries are both searched and retain distinct provenance."""
    item_orig = make_sample_candidate(point_id="p1", chunk_id="chk_1", score=0.85)
    item_trans = make_sample_candidate(point_id="p2", chunk_id="chk_2", score=0.91)

    mock_store = MockVectorStore(canned_batch_results=[[item_orig], [item_trans]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q_orig", vector=[0.1, 0.2], query_type="original"),
        transformed=EmbeddedQuery(query="q_trans", vector=[0.3, 0.4], query_type="transformed"),
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")

    assert len(results) == 2
    assert results[0].chunk_id == "chk_1"
    assert results[0].query_type == "original"
    assert results[0].score == 0.85

    assert results[1].chunk_id == "chk_2"
    assert results[1].query_type == "transformed"
    assert results[1].score == 0.91

    # Verify both vectors were searched in a single batch call
    assert len(mock_store.search_batch_calls) == 1
    assert len(mock_store.search_batch_calls[0]["query_vectors"]) == 2


@pytest.mark.anyio
async def test_search_explicit_query_provenance():
    """Verify query provenance is explicitly attached to candidates without relying on Qdrant ordering."""
    c1 = make_sample_candidate(chunk_id="chunk_alpha")
    c2 = make_sample_candidate(chunk_id="chunk_beta")
    mock_store = MockVectorStore(canned_batch_results=[[c1], [c2]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="original", vector=[0.1], query_type="original"),
        transformed=EmbeddedQuery(query="transformed", vector=[0.2], query_type="transformed"),
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")

    orig_results = [r for r in results if r.query_type == "original"]
    trans_results = [r for r in results if r.query_type == "transformed"]

    assert len(orig_results) == 1
    assert orig_results[0].chunk_id == "chunk_alpha"
    assert len(trans_results) == 1
    assert trans_results[0].chunk_id == "chunk_beta"


# ==============================================================================
# 2. Project Isolation Tests (Mandatory Invariant)
# ==============================================================================


@pytest.mark.anyio
async def test_project_isolation_empty_project_id_rejected():
    """Verify empty, whitespace, or invalid project_id raises VectorRetrievalError immediately."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="")
    assert "project_id" in str(exc_info.value).lower()

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="   ")
    assert "project_id" in str(exc_info.value).lower()

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id=None)  # type: ignore
    assert "project_id" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_project_isolation_passed_to_store():
    """Verify that the requested project_id is passed directly to the vector store."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1, 0.2], query_type="original")
    )

    await service.search(embedded_query_set=query_set, project_id="tenant_xyz")

    assert len(mock_store.search_batch_calls) == 1
    assert mock_store.search_batch_calls[0]["project_id"] == "tenant_xyz"


@pytest.mark.anyio
async def test_project_isolation_defensive_filtering():
    """Verify that if a vector point unexpectedly leaks from another project, it is dropped."""
    leaked = make_sample_candidate(chunk_id="leaked_chunk", project_id="foreign_tenant")
    valid = make_sample_candidate(chunk_id="valid_chunk", project_id="my_tenant")

    mock_store = MockVectorStore(canned_batch_results=[[leaked, valid]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="my_tenant")

    assert len(results) == 1
    assert results[0].chunk_id == "valid_chunk"
    assert results[0].project_id == "my_tenant"


# ==============================================================================
# 3. Configuration & Parameterization Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_default_top_k():
    """Verify default top_k from VectorSearchConfig is used when not explicitly passed."""
    mock_store = MockVectorStore()
    config = VectorSearchConfig(top_k=7)
    service = VectorSearchService(vector_store=mock_store, config=config)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    await service.search(embedded_query_set=query_set, project_id="proj_1")

    assert mock_store.search_batch_calls[0]["limit"] == 7


@pytest.mark.anyio
async def test_search_explicit_top_k_override():
    """Verify explicit top_k override takes precedence over config default."""
    mock_store = MockVectorStore()
    config = VectorSearchConfig(top_k=10)
    service = VectorSearchService(vector_store=mock_store, config=config)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    await service.search(embedded_query_set=query_set, project_id="proj_1", top_k=25)

    assert mock_store.search_batch_calls[0]["limit"] == 25


@pytest.mark.anyio
async def test_search_invalid_top_k_rejected():
    """Verify non-positive top_k raises VectorRetrievalError."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="proj_1", top_k=0)
    assert "top_k" in str(exc_info.value).lower()

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="proj_1", top_k=-5)
    assert "top_k" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_search_score_threshold_forwarded():
    """Verify score_threshold is forwarded from config and can be overridden."""
    mock_store = MockVectorStore()
    config = VectorSearchConfig(score_threshold=0.65)
    service = VectorSearchService(vector_store=mock_store, config=config)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    # 1. Config default
    await service.search(embedded_query_set=query_set, project_id="p1")
    assert mock_store.search_batch_calls[0]["score_threshold"] == 0.65

    # 2. Explicit override
    await service.search(embedded_query_set=query_set, project_id="p1", score_threshold=0.85)
    assert mock_store.search_batch_calls[1]["score_threshold"] == 0.85


def test_vector_search_config_validation():
    """Verify VectorSearchConfig validates its fields."""
    with pytest.raises(ValueError):
        VectorSearchConfig(top_k=0)

    with pytest.raises(ValueError):
        VectorSearchConfig(score_threshold=float("inf"))

    with pytest.raises(ValueError):
        VectorSearchConfig(score_threshold="invalid")  # type: ignore

    cfg = VectorSearchConfig.from_settings(Settings(VECTOR_SEARCH_TOP_K=15))
    assert cfg.top_k == 15


# ==============================================================================
# 4. Result Validation & Observability Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_empty_results_valid():
    """Verify empty result set from vector store returns empty list without error."""
    mock_store = MockVectorStore(canned_batch_results=[[]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_1")
    assert results == []


@pytest.mark.anyio
async def test_search_malformed_missing_chunk_id_dropped():
    """Verify points with missing or empty chunk_id are dropped defensively."""
    bad = make_sample_candidate(chunk_id="")
    good = make_sample_candidate(chunk_id="chk_good")

    mock_store = MockVectorStore(canned_batch_results=[[bad, good]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")
    assert len(results) == 1
    assert results[0].chunk_id == "chk_good"


@pytest.mark.anyio
async def test_search_malformed_missing_document_id_dropped():
    """Verify points with missing or empty document_id are dropped defensively."""
    bad = make_sample_candidate(document_id="")
    good = make_sample_candidate(document_id="doc_good")

    mock_store = MockVectorStore(canned_batch_results=[[bad, good]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")
    assert len(results) == 1
    assert results[0].document_id == "doc_good"


@pytest.mark.anyio
async def test_search_malformed_non_finite_score_dropped():
    """Verify points with NaN or Inf scores are dropped defensively."""
    bad_nan = make_sample_candidate(score=float("nan"))
    bad_inf = make_sample_candidate(score=float("inf"))
    good = make_sample_candidate(score=0.9)

    mock_store = MockVectorStore(canned_batch_results=[[bad_nan, bad_inf, good]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_test")
    assert len(results) == 1
    assert results[0].score == 0.9


# ==============================================================================
# 5. Error Handling & Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_vector_store_error_translated():
    """Verify VectorStoreError is translated into VectorRetrievalError with original_error."""
    original = VectorSearchError("Qdrant search timeout")
    mock_store = MockVectorStore(should_fail_with=original)
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="proj_1")

    assert "Vector search failed" in str(exc_info.value)
    assert exc_info.value.original_error is original
    assert isinstance(exc_info.value, RetrievalError)


@pytest.mark.anyio
async def test_search_collection_not_found_translated():
    """Verify CollectionNotFoundError is translated into VectorRetrievalError."""
    original = CollectionNotFoundError("Collection 'chunks' not found")
    mock_store = MockVectorStore(should_fail_with=original)
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="proj_1")

    assert exc_info.value.original_error is original


@pytest.mark.anyio
async def test_search_never_swallows_failure_into_empty():
    """Verify that infrastructure failure never silently returns empty list."""
    original = VectorStoreConnectionError("Cannot connect to Qdrant cluster")
    mock_store = MockVectorStore(should_fail_with=original)
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError):
        await service.search(embedded_query_set=query_set, project_id="proj_1")


@pytest.mark.anyio
async def test_search_unexpected_exception_translated():
    """Verify unexpected non-vector-store exception is translated into VectorRetrievalError."""
    original = RuntimeError("Unexpected internal crash")
    mock_store = MockVectorStore(should_fail_with=original)
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="proj_1")

    assert exc_info.value.original_error is original


# ==============================================================================
# 6. Input Structure Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_invalid_embedded_query_set():
    """Verify non-EmbeddedQuerySet raises VectorRetrievalError."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set="not an embedded query set", project_id="p1")  # type: ignore
    assert "EmbeddedQuerySet" in str(exc_info.value)


@pytest.mark.anyio
async def test_search_empty_original_vector():
    """Verify EmbeddedQuerySet with empty original vector raises VectorRetrievalError."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[], query_type="original")
    )

    with pytest.raises(VectorRetrievalError) as exc_info:
        await service.search(embedded_query_set=query_set, project_id="p1")
    assert "vector is missing or empty" in str(exc_info.value).lower()


# ==============================================================================
# 7. Multi-Query Batch Efficiency & Architectural Boundaries Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_multi_query_batch_efficiency():
    """Verify multiple query representations use a single search_batch call."""
    mock_store = MockVectorStore()
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q1", vector=[0.1, 0.2], query_type="original"),
        transformed=EmbeddedQuery(query="q2", vector=[0.3, 0.4], query_type="transformed"),
    )

    await service.search(embedded_query_set=query_set, project_id="proj_1")

    assert len(mock_store.search_batch_calls) == 1
    assert len(mock_store.search_batch_calls[0]["query_vectors"]) == 2


@pytest.mark.anyio
async def test_search_architectural_boundaries():
    """Verify Vector Search does NOT perform embedding, transformation, fusion, reranking, or SQL chunk fetching."""
    # 1. Candidate does not have full text payload or sql session
    item1 = make_sample_candidate(chunk_id="chunk_1", score=0.8, project_id="proj_1")
    item2 = make_sample_candidate(chunk_id="chunk_1", score=0.9, project_id="proj_1")  # duplicate chunk_id across branches
    mock_store = MockVectorStore(canned_batch_results=[[item1], [item2]])
    service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q1", vector=[0.1], query_type="original"),
        transformed=EmbeddedQuery(query="q2", vector=[0.2], query_type="transformed"),
    )

    results = await service.search(embedded_query_set=query_set, project_id="proj_1")

    # Vector Search preserves candidates from both branches without fusion or deduplication
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_1"
    assert results[0].query_type == "original"
    assert results[1].chunk_id == "chunk_1"
    assert results[1].query_type == "transformed"

    # Does not have chunk text (only relational coordinates)
    assert not hasattr(results[0], "content")
    assert not hasattr(results[0], "text")


# ==============================================================================
# 8. Functional Entrypoints & Package Exports Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_vectors_convenience_function():
    """Verify search_vectors convenience entrypoint works."""
    item = make_sample_candidate(chunk_id="c_conv", project_id="p1")
    mock_store = MockVectorStore(canned_batch_results=[[item]])
    custom_service = VectorSearchService(vector_store=mock_store)

    query_set = EmbeddedQuerySet(
        original=EmbeddedQuery(query="q", vector=[0.1], query_type="original")
    )

    results = await search_vectors(
        embedded_query_set=query_set,
        project_id="p1",
        service=custom_service,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c_conv"


def test_get_and_reset_vector_search_service():
    """Verify get_vector_search_service returns singleton and reset clears it."""
    s1 = get_vector_search_service()
    s2 = get_vector_search_service()
    assert s1 is s2

    reset_vector_search_service()
    s3 = get_vector_search_service()
    assert s3 is not s1


def test_package_re_exports():
    """Verify vector search symbols are accessible from both retrieval and rag.retrieval."""
    assert VectorSearchCandidate is RagVectorSearchCandidate
    assert VectorSearchConfig is RagVectorSearchConfig
    assert VectorSearchService is RagVectorSearchService
    assert get_vector_search_service is rag_get_vector_search_service
    assert reset_vector_search_service is rag_reset_vector_search_service
    assert search_vectors is rag_search_vectors
    assert VectorRetrievalError is RagVectorRetrievalError
