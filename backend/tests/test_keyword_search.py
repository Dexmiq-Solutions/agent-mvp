"""Unit tests for KeywordSearchService (lexical/sparse retrieval branch)."""

import math
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from exceptions.retrieval import KeywordRetrievalError, SparseEncodingError
from exceptions.vector import (
    CollectionNotFoundError,
    VectorSearchError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from retrieval.config import KeywordSearchConfig
from retrieval.keyword import (
    KeywordSearchService,
    get_keyword_search_service,
    reset_keyword_search_service,
    search_keywords,
)
from retrieval.keyword.encoder.base import BaseSparseEncoder
from retrieval.models import (
    KeywordSearchCandidate,
    RetrievalQuerySet,
    SparseVector,
)
from storage.vector import BaseVectorStore
from storage.vector.models import VectorPayload, VectorSearchResult


class MockSparseVectorStore(BaseVectorStore):
    """In-memory mock of BaseVectorStore for Keyword Search unit tests."""

    def __init__(
        self,
        canned_batch_results: list[list[VectorSearchResult]] | None = None,
        should_fail_with: Exception | None = None,
    ) -> None:
        self._canned_batch_results = canned_batch_results or []
        self._should_fail_with = should_fail_with
        self.recorded_calls: list[dict] = []

    @property
    def collection_name(self) -> str:
        return "test_rag_documents"

    async def ensure_collection_exists(self, vector_size=None, distance="Cosine") -> bool:
        return False

    async def upsert(self, records):
        return len(records)

    async def delete(self, point_ids):
        return len(point_ids)

    async def delete_by_filter(self, project_id, document_id=None, document_version_id=None, filter_metadata=None):
        return True

    async def search(self, query_vector, project_id, limit=10, filter_metadata=None, score_threshold=None, with_vectors=False):
        return []

    async def search_batch(self, query_vectors, project_id, limit=10, filter_metadata=None, score_threshold=None, with_vectors=False):
        return []

    async def search_sparse_batch(
        self,
        query_sparse_vectors: list[SparseVector],
        project_id: str,
        limit: int = 10,
        filter_metadata=None,
        score_threshold=None,
        vector_name: str = "sparse",
    ) -> list[list[VectorSearchResult]]:
        if self._should_fail_with:
            raise self._should_fail_with

        self.recorded_calls.append(
            {
                "query_sparse_vectors": query_sparse_vectors,
                "project_id": project_id,
                "limit": limit,
                "score_threshold": score_threshold,
                "vector_name": vector_name,
            }
        )
        return self._canned_batch_results


class MockSparseEncoder(BaseSparseEncoder):
    """Deterministic mock sparse encoder."""

    def __init__(self, should_fail_with: Exception | None = None) -> None:
        self._should_fail_with = should_fail_with
        self.encoded_queries: list[str] = []

    @property
    def strategy_name(self) -> str:
        return "mock_sparse"

    @property
    def version(self) -> str:
        return "1.0"

    def encode_document(self, text: str) -> SparseVector:
        return SparseVector(indices=(1,), values=(1.0,))

    def encode_query(self, query: str) -> SparseVector:
        if self._should_fail_with:
            raise self._should_fail_with
        self.encoded_queries.append(query)
        # Produce simple deterministic index based on length
        return SparseVector(indices=(len(query),), values=(1.0,))


def make_sample_result(
    chunk_id: str,
    project_id: str = "proj_1",
    document_id: str = "doc_1",
    score: float = 0.85,
    document_version_id: str | None = "v1",
) -> VectorSearchResult:
    """Helper creating a valid VectorSearchResult for testing."""
    payload = VectorPayload(
        project_id=project_id,
        document_id=document_id,
        chunk_id=chunk_id,
        document_version_id=document_version_id,
    )
    return VectorSearchResult(
        id=f"pt-{chunk_id}",
        score=score,
        payload=payload,
    )


@pytest.fixture(autouse=True)
def cleanup_service():
    """Reset singleton instance before and after each test."""
    reset_keyword_search_service()
    yield
    reset_keyword_search_service()


# ==============================================================================
# Search Execution & Query Provenance Tests
# ==============================================================================

@pytest.mark.anyio
async def test_search_single_original_query():
    """Verify search with only original query sets query_type='original'."""
    res1 = make_sample_result("c1", score=0.9)
    res2 = make_sample_result("c2", score=0.8)
    store = MockSparseVectorStore(canned_batch_results=[[res1, res2]])
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(original_query="What is ERR-401?", is_transformed=False)
    candidates = await service.search(retrieval_query_set=query_set, project_id="proj_1")

    assert len(candidates) == 2
    assert candidates[0].chunk_id == "c1"
    assert candidates[0].query_type == "original"
    assert candidates[0].score == 0.9
    assert candidates[1].chunk_id == "c2"
    assert candidates[1].query_type == "original"
    assert candidates[1].score == 0.8

    assert len(store.recorded_calls) == 1
    call = store.recorded_calls[0]
    assert call["project_id"] == "proj_1"
    assert len(call["query_sparse_vectors"]) == 1


@pytest.mark.anyio
async def test_search_original_and_transformed_queries_provenance():
    """Verify search preserves distinct provenance for original and transformed queries."""
    res_orig = make_sample_result("c_orig", score=0.85)
    res_trans = make_sample_result("c_trans", score=0.92)
    store = MockSparseVectorStore(canned_batch_results=[[res_orig], [res_trans]])
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(
        original_query="ERR-401",
        transformed_query="error 401 unauthorized resolution",
        is_transformed=True,
    )
    candidates = await service.search(retrieval_query_set=query_set, project_id="proj_1")

    assert len(candidates) == 2
    assert candidates[0].chunk_id == "c_orig"
    assert candidates[0].query_type == "original"
    assert candidates[0].score == 0.85

    assert candidates[1].chunk_id == "c_trans"
    assert candidates[1].query_type == "transformed"
    assert candidates[1].score == 0.92

    # Verify single batched vector store call executed with both queries
    assert len(store.recorded_calls) == 1
    assert len(store.recorded_calls[0]["query_sparse_vectors"]) == 2


# ==============================================================================
# Operational Parameter Resolution Tests
# ==============================================================================

@pytest.mark.anyio
async def test_search_top_k_and_threshold_forwarded():
    """Verify explicit top_k and score_threshold are forwarded to vector store."""
    store = MockSparseVectorStore(canned_batch_results=[[]])
    encoder = MockSparseEncoder()
    config = KeywordSearchConfig(top_k=10, score_threshold=0.5)
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder, config=config)

    query_set = RetrievalQuerySet(original_query="test query")

    # 1. Config defaults
    await service.search(retrieval_query_set=query_set, project_id="proj_1")
    assert store.recorded_calls[0]["limit"] == 10
    assert store.recorded_calls[0]["score_threshold"] == 0.5

    # 2. Explicit overrides
    await service.search(
        retrieval_query_set=query_set,
        project_id="proj_1",
        top_k=25,
        score_threshold=0.75,
    )
    assert store.recorded_calls[1]["limit"] == 25
    assert store.recorded_calls[1]["score_threshold"] == 0.75


# ==============================================================================
# Validation and Tenant Isolation Tests
# ==============================================================================

@pytest.mark.anyio
async def test_empty_or_invalid_project_id_rejected():
    """Verify empty or non-string project_id raises KeywordRetrievalError."""
    store = MockSparseVectorStore()
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)
    query_set = RetrievalQuerySet(original_query="test")

    with pytest.raises(KeywordRetrievalError, match="project_id must be a non-empty string"):
        await service.search(retrieval_query_set=query_set, project_id="")

    with pytest.raises(KeywordRetrievalError, match="project_id must be a non-empty string"):
        await service.search(retrieval_query_set=query_set, project_id="   ")

    with pytest.raises(KeywordRetrievalError, match="project_id must be a non-empty string"):
        await service.search(retrieval_query_set=query_set, project_id=None)  # type: ignore


@pytest.mark.anyio
async def test_invalid_query_set_rejected():
    """Verify invalid query_set or empty original_query raises KeywordRetrievalError."""
    store = MockSparseVectorStore()
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    with pytest.raises(KeywordRetrievalError, match="Expected RetrievalQuerySet instance"):
        await service.search(retrieval_query_set="not a query set", project_id="p1")  # type: ignore

    with pytest.raises(KeywordRetrievalError, match="original_query is missing or empty"):
        await service.search(retrieval_query_set=RetrievalQuerySet(original_query=""), project_id="p1")


@pytest.mark.anyio
async def test_invalid_top_k_or_threshold_rejected():
    """Verify invalid top_k or non-finite threshold raises KeywordRetrievalError."""
    store = MockSparseVectorStore()
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)
    query_set = RetrievalQuerySet(original_query="test")

    with pytest.raises(KeywordRetrievalError, match="top_k must be a positive integer"):
        await service.search(retrieval_query_set=query_set, project_id="p1", top_k=0)

    with pytest.raises(KeywordRetrievalError, match="score_threshold must be a finite float"):
        await service.search(retrieval_query_set=query_set, project_id="p1", score_threshold=float("nan"))


@pytest.mark.anyio
async def test_defensive_tenant_isolation_discards_mismatched_project():
    """Verify candidates with mismatched project_id are dropped defensively."""
    good_res = make_sample_result("c_good", project_id="proj_target")
    leaked_res = make_sample_result("c_bad", project_id="proj_OTHER_TENANT")

    store = MockSparseVectorStore(canned_batch_results=[[good_res, leaked_res]])
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(original_query="secure query")
    candidates = await service.search(retrieval_query_set=query_set, project_id="proj_target")

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c_good"
    assert candidates[0].project_id == "proj_target"


@pytest.mark.anyio
async def test_malformed_candidate_records_dropped():
    """Verify candidates with missing chunk_id, document_id, or non-finite scores are discarded."""
    valid_res = make_sample_result("c_ok", score=0.8)

    no_chunk = make_sample_result("", score=0.8)
    no_doc = make_sample_result("c_no_doc", score=0.8)
    object.__setattr__(no_doc.payload, "document_id", "")
    bad_score = make_sample_result("c_nan", score=float("nan"))

    store = MockSparseVectorStore(canned_batch_results=[[valid_res, no_chunk, no_doc, bad_score]])
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(original_query="check validation")
    candidates = await service.search(retrieval_query_set=query_set, project_id="proj_1")

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c_ok"


# ==============================================================================
# Error Handling & Infrastructure Translation Tests
# ==============================================================================

@pytest.mark.anyio
async def test_vector_store_error_translated_to_keyword_retrieval_error():
    """Verify VectorStoreError is wrapped into KeywordRetrievalError."""
    store = MockSparseVectorStore(should_fail_with=VectorSearchError("Sparse index timeout"))
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(original_query="timeout test")
    with pytest.raises(KeywordRetrievalError, match="Keyword search failed"):
        await service.search(retrieval_query_set=query_set, project_id="p1")


@pytest.mark.anyio
async def test_sparse_encoder_error_translated_to_keyword_retrieval_error():
    """Verify SparseEncodingError is wrapped into KeywordRetrievalError."""
    store = MockSparseVectorStore()
    encoder = MockSparseEncoder(should_fail_with=SparseEncodingError("Encoding failure"))
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(original_query="encoder failure test")
    with pytest.raises(KeywordRetrievalError, match="Failed to encode query text"):
        await service.search(retrieval_query_set=query_set, project_id="p1")


# ==============================================================================
# Architecture Boundaries & Convenience Function Tests
# ==============================================================================

@pytest.mark.anyio
async def test_architectural_boundaries():
    """Verify KeywordSearchCandidate preserves minimal relational fields without bloated SQL state."""
    res1 = make_sample_result("chunk_alpha", score=0.77)
    res2 = make_sample_result("chunk_alpha", score=0.88)  # duplicate chunk across query branches
    store = MockSparseVectorStore(canned_batch_results=[[res1], [res2]])
    encoder = MockSparseEncoder()
    service = KeywordSearchService(vector_store=store, sparse_encoder=encoder)

    query_set = RetrievalQuerySet(
        original_query="query 1",
        transformed_query="query 2",
        is_transformed=True,
    )
    candidates = await service.search(retrieval_query_set=query_set, project_id="proj_1")

    # Keyword search does NOT deduplicate across query branches (Fusion does that)
    assert len(candidates) == 2
    for cand in candidates:
        assert isinstance(cand, KeywordSearchCandidate)
        assert hasattr(cand, "chunk_id")
        assert hasattr(cand, "document_id")
        assert hasattr(cand, "project_id")
        assert hasattr(cand, "score")
        assert hasattr(cand, "query_type")
        # Ensure no SQL session or chunk body payload is attached
        assert not hasattr(cand, "content")
        assert not hasattr(cand, "session")


@pytest.mark.anyio
async def test_convenience_function_and_singleton():
    """Verify search_keywords and get_keyword_search_service work seamlessly."""
    res = make_sample_result("c_conv")
    store = MockSparseVectorStore(canned_batch_results=[[res]])
    encoder = MockSparseEncoder()

    service = get_keyword_search_service(vector_store=store, sparse_encoder=encoder)
    query_set = RetrievalQuerySet(original_query="convenience test")

    candidates = await search_keywords(
        retrieval_query_set=query_set,
        project_id="proj_1",
        service=service,
    )

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c_conv"
