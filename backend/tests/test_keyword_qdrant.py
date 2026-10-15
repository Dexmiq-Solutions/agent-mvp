"""Unit tests for Qdrant sparse vector integration and lexical search."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from qdrant_client.http import models

from core.config import Settings
from exceptions.vector import (
    CollectionConfigurationError,
    VectorInputValidationError,
    VectorSearchError,
)
from storage.vector.models import SparseVector, VectorPayload, VectorRecord
from storage.vector.qdrant import QdrantVectorStore


@pytest.fixture
def test_settings():
    """Settings instance for Qdrant testing."""
    return Settings(
        QDRANT_URL="https://mock-cluster.qdrant.tech:6333",
        QDRANT_API_KEY="mock-api-key",
        QDRANT_COLLECTION_NAME="test_lexical_collection",
        QDRANT_TIMEOUT=30,
        QDRANT_VECTOR_SIZE=4,
        SPARSE_INDEXING_ENABLED=True,
        SPARSE_VECTOR_NAME="sparse",
    )


@pytest.fixture
def mock_qdrant_client():
    """Mock AsyncQdrantClient with required methods."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.get_collection = AsyncMock()
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock()
    client.query_batch_points = AsyncMock()
    return client


class TestQdrantSparseCollection:
    """Tests for collection lifecycle with sparse vectors."""

    @pytest.mark.anyio
    async def test_ensure_collection_creates_with_sparse_vectors_config(
        self, test_settings, mock_qdrant_client
    ):
        """Verify collection creation configures both dense and sparse vectors."""
        mock_qdrant_client.collection_exists.return_value = False
        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )

        created = await store.ensure_collection_exists(vector_size=4)
        assert created is True

        mock_qdrant_client.create_collection.assert_awaited_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_lexical_collection"
        assert "sparse_vectors_config" in call_kwargs
        sparse_config = call_kwargs["sparse_vectors_config"]
        assert "sparse" in sparse_config
        assert isinstance(sparse_config["sparse"], models.SparseVectorParams)
        assert sparse_config["sparse"].modifier == models.Modifier.IDF

    @pytest.mark.anyio
    async def test_ensure_collection_detects_missing_sparse_vector(
        self, test_settings, mock_qdrant_client
    ):
        """Verify error raised if existing collection lacks required sparse vector configuration."""
        mock_qdrant_client.collection_exists.return_value = True
        # Mock existing collection with only dense vector, no sparse vector
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = models.VectorParams(
            size=4, distance=models.Distance.COSINE
        )
        mock_collection_info.config.params.sparse_vectors = None
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )

        with pytest.raises(CollectionConfigurationError, match="missing required sparse vector"):
            await store.ensure_collection_exists(vector_size=4)

    @pytest.mark.anyio
    async def test_ensure_collection_succeeds_with_existing_sparse_vector(
        self, test_settings, mock_qdrant_client
    ):
        """Verify existing collection with sparse vector is validated successfully."""
        mock_qdrant_client.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors = models.VectorParams(
            size=4, distance=models.Distance.COSINE
        )
        mock_collection_info.config.params.sparse_vectors = {
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )

        created = await store.ensure_collection_exists(vector_size=4)
        assert created is False


class TestQdrantSparseUpsert:
    """Tests for upserting dual dense + sparse records."""

    @pytest.mark.anyio
    async def test_upsert_dual_vectors(self, test_settings, mock_qdrant_client):
        """Verify upsert constructs PointStruct with both dense and sparse vectors."""
        mock_qdrant_client.upsert.return_value = MagicMock(
            status=models.UpdateStatus.COMPLETED
        )
        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )

        payload = VectorPayload(
            project_id="proj_1",
            document_id="doc_1",
            chunk_id="chunk_1",
        )
        sparse_vec = SparseVector(indices=(10, 20), values=(1.5, 2.5))
        record = VectorRecord(
            id="point-1",
            vector=[0.1, 0.2, 0.3, 0.4],
            payload=payload,
            sparse_vector=sparse_vec,
        )

        count = await store.upsert([record])
        assert count == 1

        mock_qdrant_client.upsert.assert_awaited_once()
        points = mock_qdrant_client.upsert.call_args.kwargs["points"]
        assert len(points) == 1
        pt = points[0]
        assert isinstance(pt.vector, dict)
        assert pt.vector[""] == [0.1, 0.2, 0.3, 0.4]
        assert isinstance(pt.vector["sparse"], models.SparseVector)
        assert pt.vector["sparse"].indices == [10, 20]
        assert pt.vector["sparse"].values == [1.5, 2.5]


class TestQdrantSparseSearch:
    """Tests for single and batch sparse search in Qdrant."""

    @pytest.mark.anyio
    async def test_search_sparse_empty_vector_returns_empty_immediately(
        self, test_settings, mock_qdrant_client
    ):
        """Verify empty sparse vector returns empty list without calling Qdrant."""
        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )
        empty_sv = SparseVector()
        results = await store.search_sparse(
            query_sparse_vector=empty_sv,
            project_id="proj_1",
        )
        assert results == []
        mock_qdrant_client.query_points.assert_not_awaited()

    @pytest.mark.anyio
    async def test_search_sparse_executes_with_project_filter(
        self, test_settings, mock_qdrant_client
    ):
        """Verify search_sparse executes query_points with SparseVector and project filter."""
        mock_point = MagicMock()
        mock_point.id = "p-1"
        mock_point.score = 0.88
        mock_point.payload = {
            "project_id": "proj_1",
            "document_id": "doc_1",
            "chunk_id": "chunk_1",
        }
        mock_qdrant_client.query_points.return_value = MagicMock(points=[mock_point])

        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )
        sv = SparseVector(indices=(100, 200), values=(1.0, 2.0))
        results = await store.search_sparse(
            query_sparse_vector=sv,
            project_id="proj_1",
            limit=5,
            score_threshold=0.1,
        )

        assert len(results) == 1
        res = results[0]
        assert res.id == "p-1"
        assert res.score == 0.88
        assert res.payload.chunk_id == "chunk_1"

        mock_qdrant_client.query_points.assert_awaited_once()
        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["using"] == "sparse"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["score_threshold"] == 0.1
        assert isinstance(call_kwargs["query"], models.SparseVector)
        assert call_kwargs["query"].indices == [100, 200]

    @pytest.mark.anyio
    async def test_search_sparse_batch_mixed_empty_and_non_empty(
        self, test_settings, mock_qdrant_client
    ):
        """Verify search_sparse_batch handles empty and non-empty queries gracefully."""
        mock_point = MagicMock()
        mock_point.id = "p-non-empty"
        mock_point.score = 0.95
        mock_point.payload = {
            "project_id": "proj_test",
            "document_id": "doc_test",
            "chunk_id": "c_test",
        }
        mock_resp = MagicMock(points=[mock_point])
        mock_qdrant_client.query_batch_points.return_value = [mock_resp]

        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )

        sv1 = SparseVector(indices=(10,), values=(1.0,))
        sv_empty = SparseVector()  # empty query

        results = await store.search_sparse_batch(
            query_sparse_vectors=[sv1, sv_empty],
            project_id="proj_test",
            limit=10,
        )

        assert len(results) == 2
        # First query returned matches
        assert len(results[0]) == 1
        assert results[0][0].id == "p-non-empty"
        # Second empty query returned empty list
        assert len(results[1]) == 0

        # Qdrant client was only queried for the non-empty query
        mock_qdrant_client.query_batch_points.assert_awaited_once()
        requests = mock_qdrant_client.query_batch_points.call_args.kwargs["requests"]
        assert len(requests) == 1

    @pytest.mark.anyio
    async def test_search_sparse_batch_all_empty(
        self, test_settings, mock_qdrant_client
    ):
        """Verify search_sparse_batch when all vectors are empty returns lists without Qdrant call."""
        store = QdrantVectorStore(
            collection_name="test_lexical_collection",
            settings=test_settings,
            client=mock_qdrant_client,
        )
        results = await store.search_sparse_batch(
            query_sparse_vectors=[SparseVector(), SparseVector()],
            project_id="proj_test",
        )
        assert results == [[], []]
        mock_qdrant_client.query_batch_points.assert_not_awaited()
