"""Unit tests for Qdrant vector store, client, models, and error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from app.core.config import Settings
from exceptions.vector import (
    CollectionConfigurationError,
    CollectionNotFoundError,
    VectorDeletionError,
    VectorInputValidationError,
    VectorSearchError,
    VectorStoreAuthenticationError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from storage.vector import (
    BaseVectorStore,
    QdrantVectorStore,
    VectorPayload,
    VectorRecord,
    VectorSearchResult,
    close_async_qdrant_client,
    get_async_qdrant_client,
    get_vector_store,
    reset_async_qdrant_client,
    reset_vector_store,
)


@pytest.fixture(autouse=True)
def cleanup_vector_store():
    """Reset cached singleton client and provider before and after each test."""
    reset_async_qdrant_client()
    reset_vector_store()
    yield
    reset_async_qdrant_client()
    reset_vector_store()


@pytest.fixture
def mock_qdrant_settings():
    """Sample settings with Qdrant configuration."""
    return Settings(
        QDRANT_URL="https://mock-cluster.qdrant.tech:6333",
        QDRANT_API_KEY="mock-qdrant-key",
        QDRANT_COLLECTION_NAME="test_chunks",
        QDRANT_TIMEOUT=30,
        QDRANT_VECTOR_SIZE=1024,
    )


@pytest.fixture
def mock_qdrant_client():
    """Mock AsyncQdrantClient instance."""
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock(return_value=True)
    client.get_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.query_points = AsyncMock()
    client.query_batch_points = AsyncMock()
    client.close = AsyncMock()
    return client


# ==============================================================================
# 1. Configuration & Client Management Tests
# ==============================================================================


def test_get_async_qdrant_client_missing_url():
    """Verify get_async_qdrant_client raises VectorStoreConfigurationError when URL is missing."""
    empty_settings = Settings(QDRANT_URL=None)
    with pytest.raises(VectorStoreConfigurationError) as exc_info:
        get_async_qdrant_client(settings=empty_settings)
    assert "QDRANT_URL" in str(exc_info.value)


def test_get_async_qdrant_client_caching(mock_qdrant_settings):
    """Verify get_async_qdrant_client returns cached singleton client."""
    client1 = get_async_qdrant_client(settings=mock_qdrant_settings)
    client2 = get_async_qdrant_client(settings=mock_qdrant_settings)

    assert client1 is client2


def test_reset_async_qdrant_client(mock_qdrant_settings):
    """Verify reset_async_qdrant_client clears cached client instance."""
    client1 = get_async_qdrant_client(settings=mock_qdrant_settings)
    reset_async_qdrant_client()
    client2 = get_async_qdrant_client(settings=mock_qdrant_settings)

    assert client1 is not client2


@pytest.mark.anyio
async def test_close_async_qdrant_client(mock_qdrant_settings):
    """Verify close_async_qdrant_client closes client and clears instance."""
    with patch("storage.vector.client.AsyncQdrantClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_client_cls.return_value = mock_instance

        client = get_async_qdrant_client(settings=mock_qdrant_settings)
        await close_async_qdrant_client()

        mock_instance.close.assert_awaited_once()


# ==============================================================================
# 2. Model & Payload Serialization Tests
# ==============================================================================


def test_vector_payload_serialization():
    """Verify VectorPayload round-trip serialization to and from dictionary."""
    payload = VectorPayload(
        project_id="proj_123",
        document_id="doc_456",
        chunk_id="chunk_789",
        document_version_id="ver_001",
        metadata={"page": 3, "section": "Introduction"},
    )

    data = payload.to_dict()
    assert data["project_id"] == "proj_123"
    assert data["document_id"] == "doc_456"
    assert data["chunk_id"] == "chunk_789"
    assert data["document_version_id"] == "ver_001"
    assert data["page"] == 3
    assert data["section"] == "Introduction"

    restored = VectorPayload.from_dict(data)
    assert restored.project_id == "proj_123"
    assert restored.document_id == "doc_456"
    assert restored.chunk_id == "chunk_789"
    assert restored.document_version_id == "ver_001"
    assert restored.metadata == {"page": 3, "section": "Introduction"}


def test_vector_models_structure():
    """Verify VectorRecord and VectorSearchResult structure."""
    payload = VectorPayload(project_id="proj_1", document_id="doc_1", chunk_id="chk_1")
    rec = VectorRecord(id="point_1", vector=[0.1, 0.2, 0.3], payload=payload)
    assert rec.id == "point_1"
    assert len(rec.vector) == 3

    search_res = VectorSearchResult(id="point_1", score=0.95, payload=payload, vector=[0.1, 0.2, 0.3])
    assert search_res.score == 0.95
    assert search_res.payload.project_id == "proj_1"


# ==============================================================================
# 3. Collection Management Tests
# ==============================================================================


@pytest.mark.anyio
async def test_ensure_collection_exists_creates_when_missing(
    mock_qdrant_settings, mock_qdrant_client
):
    """Verify ensure_collection_exists creates collection when missing."""
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)

    created = await store.ensure_collection_exists(vector_size=1024, distance="Cosine")
    assert created is True
    mock_qdrant_client.create_collection.assert_awaited_once()
    call_kwargs = mock_qdrant_client.create_collection.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_chunks"
    assert call_kwargs["vectors_config"].size == 1024
    assert call_kwargs["vectors_config"].distance == models.Distance.COSINE


@pytest.mark.anyio
async def test_ensure_collection_exists_already_present_compatible(
    mock_qdrant_settings, mock_qdrant_client
):
    """Verify ensure_collection_exists returns False when compatible collection exists."""
    mock_qdrant_client.collection_exists.return_value = True
    info = MagicMock()
    info.config.params.vectors = models.VectorParams(size=1024, distance=models.Distance.COSINE)
    mock_qdrant_client.get_collection.return_value = info

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    created = await store.ensure_collection_exists(vector_size=1024)

    assert created is False
    mock_qdrant_client.create_collection.assert_not_awaited()


@pytest.mark.anyio
async def test_ensure_collection_exists_dimension_mismatch(
    mock_qdrant_settings, mock_qdrant_client
):
    """Verify ensure_collection_exists raises CollectionConfigurationError on dimension mismatch."""
    mock_qdrant_client.collection_exists.return_value = True
    info = MagicMock()
    info.config.params.vectors = models.VectorParams(size=768, distance=models.Distance.COSINE)
    mock_qdrant_client.get_collection.return_value = info

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    with pytest.raises(CollectionConfigurationError) as exc_info:
        await store.ensure_collection_exists(vector_size=1024)

    assert "768" in str(exc_info.value)
    assert "1024" in str(exc_info.value)


# ==============================================================================
# 4. Input Validation Tests
# ==============================================================================


def test_validate_project_id(mock_qdrant_settings):
    """Verify project_id validation catches empty or non-string values."""
    store = QdrantVectorStore(settings=mock_qdrant_settings)
    with pytest.raises(VectorInputValidationError):
        store._validate_project_id("")

    with pytest.raises(VectorInputValidationError):
        store._validate_project_id("   ")

    with pytest.raises(VectorInputValidationError):
        store._validate_project_id(None)


def test_validate_records_invalid_inputs(mock_qdrant_settings):
    """Verify record validation catches empty batches, invalid IDs, and missing references."""
    store = QdrantVectorStore(settings=mock_qdrant_settings)

    with pytest.raises(VectorInputValidationError):
        store._validate_records([])

    with pytest.raises(VectorInputValidationError):
        store._validate_records("not a list")

    payload_valid = VectorPayload(project_id="p1", document_id="d1", chunk_id="c1")

    # Invalid ID
    with pytest.raises(VectorInputValidationError):
        store._validate_records([VectorRecord(id="", vector=[0.1], payload=payload_valid)])

    # Empty vector
    with pytest.raises(VectorInputValidationError):
        store._validate_records([VectorRecord(id="pt1", vector=[], payload=payload_valid)])

    # Missing chunk_id in payload
    payload_invalid = VectorPayload(project_id="p1", document_id="d1", chunk_id="")
    with pytest.raises(VectorInputValidationError):
        store._validate_records([VectorRecord(id="pt1", vector=[0.1], payload=payload_invalid)])


# ==============================================================================
# 5. Upsert Operations Tests
# ==============================================================================


@pytest.mark.anyio
async def test_upsert_batch_success(mock_qdrant_settings, mock_qdrant_client):
    """Verify upsert converts VectorRecord items into PointStruct and returns count."""
    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)

    records = [
        VectorRecord(
            id="point_1",
            vector=[0.1, 0.2, 0.3],
            payload=VectorPayload(project_id="p1", document_id="d1", chunk_id="c1"),
        ),
        VectorRecord(
            id="point_2",
            vector=[0.4, 0.5, 0.6],
            payload=VectorPayload(project_id="p1", document_id="d1", chunk_id="c2"),
        ),
    ]

    count = await store.upsert(records)
    assert count == 2
    mock_qdrant_client.upsert.assert_awaited_once()
    call_kwargs = mock_qdrant_client.upsert.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_chunks"
    points = call_kwargs["points"]
    assert len(points) == 2
    assert points[0].id == "point_1"
    assert points[0].payload["project_id"] == "p1"
    assert points[1].id == "point_2"


# ==============================================================================
# 6. Deletion Operations Tests
# ==============================================================================


@pytest.mark.anyio
async def test_delete_by_point_ids_success(mock_qdrant_settings, mock_qdrant_client):
    """Verify delete removes points by ID list."""
    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)

    deleted_count = await store.delete(["point_1", "point_2"])
    assert deleted_count == 2
    mock_qdrant_client.delete.assert_awaited_once_with(
        collection_name="test_chunks",
        points_selector=["point_1", "point_2"],
        wait=True,
    )


@pytest.mark.anyio
async def test_delete_by_filter_success(mock_qdrant_settings, mock_qdrant_client):
    """Verify delete_by_filter enforces project isolation filter."""
    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)

    res = await store.delete_by_filter(project_id="p1", document_id="d1")
    assert res is True
    mock_qdrant_client.delete.assert_awaited_once()
    call_kwargs = mock_qdrant_client.delete.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_chunks"
    selector = call_kwargs["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    must_conditions = selector.filter.must
    assert len(must_conditions) == 2
    assert must_conditions[0].key == "project_id"
    assert must_conditions[0].match.value == "p1"
    assert must_conditions[1].key == "document_id"
    assert must_conditions[1].match.value == "d1"


# ==============================================================================
# 7. Similarity Search & Project Isolation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_enforces_project_isolation(mock_qdrant_settings, mock_qdrant_client):
    """Verify search queries include project_id filter condition and map results."""
    mock_point = MagicMock()
    mock_point.id = "point_1"
    mock_point.score = 0.92
    mock_point.payload = {
        "project_id": "proj_target",
        "document_id": "doc_1",
        "chunk_id": "chk_1",
        "page": 2,
    }
    mock_point.vector = None

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_qdrant_client.query_points.return_value = mock_response

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    results = await store.search(
        query_vector=[0.1, 0.2, 0.3],
        project_id="proj_target",
        limit=5,
        filter_metadata={"section": "Summary"},
        score_threshold=0.8,
    )

    assert len(results) == 1
    assert results[0].id == "point_1"
    assert results[0].score == 0.92
    assert results[0].payload.project_id == "proj_target"
    assert results[0].payload.metadata["page"] == 2

    mock_qdrant_client.query_points.assert_awaited_once()
    call_kwargs = mock_qdrant_client.query_points.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_chunks"
    assert call_kwargs["query"] == [0.1, 0.2, 0.3]
    assert call_kwargs["limit"] == 5
    assert call_kwargs["score_threshold"] == 0.8

    q_filter = call_kwargs["query_filter"]
    assert len(q_filter.must) == 2
    assert q_filter.must[0].key == "project_id"
    assert q_filter.must[0].match.value == "proj_target"
    assert q_filter.must[1].key == "section"
    assert q_filter.must[1].match.value == "Summary"


@pytest.mark.anyio
async def test_search_filters_out_mismatched_project_id_defensively(
    mock_qdrant_settings, mock_qdrant_client
):
    """Verify search drops points if project_id unexpectedly does not match."""
    leaked_point = MagicMock()
    leaked_point.id = "leaked_1"
    leaked_point.score = 0.99
    leaked_point.payload = {"project_id": "other_project", "document_id": "d2", "chunk_id": "c2"}
    leaked_point.vector = None

    mock_response = MagicMock()
    mock_response.points = [leaked_point]
    mock_qdrant_client.query_points.return_value = mock_response

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    results = await store.search(query_vector=[0.1, 0.2], project_id="my_project")

    # Mismatched result should be filtered out
    assert len(results) == 0


# ==============================================================================
# 8. Error Handling & Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_qdrant_authentication_error_translation(mock_qdrant_settings, mock_qdrant_client):
    """Verify 401 Unauthorized raises VectorStoreAuthenticationError."""
    http_error = qdrant_exceptions.UnexpectedResponse(
        status_code=401,
        reason_phrase="Unauthorized",
        content=b"Invalid API key",
        headers={},
    )
    mock_qdrant_client.upsert.side_effect = http_error

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    records = [
        VectorRecord(
            id="p1",
            vector=[0.1],
            payload=VectorPayload(project_id="pr1", document_id="d1", chunk_id="c1"),
        )
    ]

    with pytest.raises(VectorStoreAuthenticationError) as exc_info:
        await store.upsert(records)
    assert "Authentication failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_qdrant_collection_not_found_translation(mock_qdrant_settings, mock_qdrant_client):
    """Verify 404 Not Found raises CollectionNotFoundError."""
    http_error = qdrant_exceptions.UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b"Collection not found",
        headers={},
    )
    mock_qdrant_client.query_points.side_effect = http_error

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    with pytest.raises(CollectionNotFoundError) as exc_info:
        await store.search(query_vector=[0.1, 0.2], project_id="p1")
    assert "not found" in str(exc_info.value).lower()


# ==============================================================================
# 9. Factory Function Tests
# ==============================================================================


def test_get_vector_store_factory(mock_qdrant_settings):
    """Verify get_vector_store returns BaseVectorStore with singleton caching."""
    store1 = get_vector_store(settings=mock_qdrant_settings)
    store2 = get_vector_store(settings=mock_qdrant_settings)

    assert isinstance(store1, BaseVectorStore)
    assert store1 is store2
    assert store1.collection_name == "test_chunks"

    # Custom collection name returns a new instance
    custom_store = get_vector_store(
        collection_name="custom_collection", settings=mock_qdrant_settings
    )
    assert custom_store.collection_name == "custom_collection"
    assert custom_store is not store1


# ==============================================================================
# 10. Batch Similarity Search Tests
# ==============================================================================


@pytest.mark.anyio
async def test_search_batch_single_query_delegates_to_search(mock_qdrant_settings, mock_qdrant_client):
    """Verify search_batch with 1 query vector delegates directly to search() without batch overhead."""
    mock_point = MagicMock()
    mock_point.id = "p_single"
    mock_point.score = 0.95
    mock_point.payload = {"project_id": "proj_1", "document_id": "doc_1", "chunk_id": "c_1"}
    mock_point.vector = None

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_qdrant_client.query_points.return_value = mock_response

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    results = await store.search_batch(
        query_vectors=[[0.1, 0.2]],
        project_id="proj_1",
        limit=5,
    )

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].id == "p_single"
    assert results[0][0].score == 0.95
    assert results[0][0].payload.chunk_id == "c_1"

    mock_qdrant_client.query_points.assert_awaited_once()
    mock_qdrant_client.query_batch_points.assert_not_awaited()


@pytest.mark.anyio
async def test_search_batch_multiple_queries_native_batch(mock_qdrant_settings, mock_qdrant_client):
    """Verify search_batch with multiple vectors executes a single query_batch_points call."""
    mock_pt1 = MagicMock()
    mock_pt1.id = "pt_1"
    mock_pt1.score = 0.88
    mock_pt1.payload = {"project_id": "proj_alpha", "document_id": "doc_1", "chunk_id": "c_1"}
    mock_pt1.vector = None

    mock_pt2 = MagicMock()
    mock_pt2.id = "pt_2"
    mock_pt2.score = 0.92
    mock_pt2.payload = {"project_id": "proj_alpha", "document_id": "doc_2", "chunk_id": "c_2"}
    mock_pt2.vector = None

    resp1 = MagicMock()
    resp1.points = [mock_pt1]
    resp2 = MagicMock()
    resp2.points = [mock_pt2]
    mock_qdrant_client.query_batch_points.return_value = [resp1, resp2]

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    results = await store.search_batch(
        query_vectors=[[0.1, 0.2], [0.3, 0.4]],
        project_id="proj_alpha",
        limit=10,
        score_threshold=0.7,
    )

    assert len(results) == 2
    assert len(results[0]) == 1
    assert results[0][0].id == "pt_1"
    assert results[0][0].score == 0.88
    assert results[1][0].id == "pt_2"
    assert results[1][0].score == 0.92

    mock_qdrant_client.query_batch_points.assert_awaited_once()
    call_kwargs = mock_qdrant_client.query_batch_points.await_args.kwargs
    assert call_kwargs["collection_name"] == "test_chunks"
    requests = call_kwargs["requests"]
    assert len(requests) == 2
    assert requests[0].query == [0.1, 0.2]
    assert requests[0].limit == 10
    assert requests[0].score_threshold == 0.7
    assert requests[1].query == [0.3, 0.4]

    # Verify project isolation filter inside both requests
    for req in requests:
        assert req.filter is not None
        assert any(
            cond.key == "project_id" and cond.match.value == "proj_alpha"
            for cond in req.filter.must
        )


@pytest.mark.anyio
async def test_search_batch_drops_mismatched_project_defensively(mock_qdrant_settings, mock_qdrant_client):
    """Verify search_batch defensively discards any returned point with a mismatched project_id."""
    leaked_pt = MagicMock()
    leaked_pt.id = "leaked_point"
    leaked_pt.score = 0.99
    leaked_pt.payload = {"project_id": "foreign_project", "document_id": "doc_x", "chunk_id": "c_x"}
    leaked_pt.vector = None

    valid_pt = MagicMock()
    valid_pt.id = "valid_point"
    valid_pt.score = 0.85
    valid_pt.payload = {"project_id": "target_project", "document_id": "doc_y", "chunk_id": "c_y"}
    valid_pt.vector = None

    resp1 = MagicMock()
    resp1.points = [leaked_pt, valid_pt]
    mock_qdrant_client.query_batch_points.return_value = [resp1]

    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)
    results = await store.search_batch(
        query_vectors=[[0.1, 0.2], [0.3, 0.4]],
        project_id="target_project",
    )

    # Only the valid point should remain; leaked point discarded
    assert len(results[0]) == 1
    assert results[0][0].id == "valid_point"


@pytest.mark.anyio
async def test_search_batch_input_validation(mock_qdrant_settings, mock_qdrant_client):
    """Verify search_batch validates inputs strictly."""
    store = QdrantVectorStore(client=mock_qdrant_client, settings=mock_qdrant_settings)

    with pytest.raises(VectorInputValidationError):
        await store.search_batch(query_vectors=[], project_id="p1")

    with pytest.raises(VectorInputValidationError):
        await store.search_batch(query_vectors=[[]], project_id="p1")

    with pytest.raises(VectorInputValidationError):
        await store.search_batch(query_vectors=[[0.1]], project_id="")

    with pytest.raises(VectorInputValidationError):
        await store.search_batch(query_vectors=[[0.1]], project_id="p1", limit=0)
