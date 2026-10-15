"""Focused tests verifying the Cohere Embed v4 1024-dimension enforcement.

Covers:
  - Requirement 2 & 3: output_dimension=1024 is explicitly set on all Cohere embed requests (document & query)
  - Requirement 6: Unit tests asserting exact request payloads and returned 1024-dimensional vectors
  - Requirement 7: Document chunks index into 1024d Qdrant collection without dimensionality mismatch
  - Requirement 8: Query embeddings generate 1024d vectors compatible with Qdrant retrieval
"""

from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from core.config import Settings
from rag.embeddings.cohere import (
    DEFAULT_COHERE_EMBEDDING_DIM,
    CohereEmbeddingProvider,
)
from rag.embeddings.cohere_client import (
    get_async_cohere_client,
    reset_async_cohere_client,
)
from rag.indexing.models import IndexableRecord
from rag.retrieval.embedding.service import QueryEmbeddingService
from rag.retrieval.models import RetrievalQuerySet
from storage.vector.models import VectorPayload
from storage.vector.qdrant import QdrantVectorStore


@pytest.fixture(autouse=True)
def cleanup():
    """Reset singletons before and after test execution."""
    reset_async_cohere_client()
    yield
    reset_async_cohere_client()


@pytest.fixture
def mock_cohere_settings():
    return Settings(
        COHERE_API_KEY="test-cohere-key-abc",
        EMBEDDING_PROVIDER="cohere",
        EMBEDDING_MODEL="embed-v4.0",
        QDRANT_VECTOR_SIZE=1024,
    )


def _make_mock_embed_response(count: int, dim: int = 1024, total_tokens: int = 25):
    resp = MagicMock()
    resp.embeddings = [[0.05 * (j + 1) for j in range(dim)] for _ in range(count)]
    resp.meta = MagicMock()
    resp.meta.tokens = MagicMock()
    resp.meta.tokens.input_tokens = total_tokens
    resp.meta.billed_units = None
    return resp


# ==============================================================================
# 1. Cohere Provider Explicit output_dimension=1024 Unit Tests
# ==============================================================================


@pytest.mark.anyio
async def test_cohere_provider_document_embedding_explicit_1024_dimension(mock_cohere_settings):
    """Verify document embedding explicitly sets output_dimension=1024 and returns 1024d vector."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    assert provider.model_name == "embed-v4.0"
    assert provider.output_dimension == 1024
    assert provider.expected_dim == 1024

    doc_text = "This is a document paragraph describing system architecture."
    vector = await provider.embed_text(doc_text, input_type="document")

    # Vector length must be 1024
    assert len(vector) == 1024

    # Crucial assertion: output_dimension=1024 MUST be explicitly provided to Cohere
    mock_client.embed.assert_awaited_once_with(
        texts=[doc_text],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=1024,
    )


@pytest.mark.anyio
async def test_cohere_provider_query_embedding_explicit_1024_dimension(mock_cohere_settings):
    """Verify query embedding explicitly sets output_dimension=1024 and returns 1024d vector."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    assert provider.output_dimension == 1024

    query_text = "What is the system architecture?"
    vector = await provider.embed_query(query_text)

    # Vector length must be 1024
    assert len(vector) == 1024

    # Crucial assertion: output_dimension=1024 MUST be explicitly provided to Cohere
    mock_client.embed.assert_awaited_once_with(
        texts=[query_text],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=1024,
    )


@pytest.mark.anyio
async def test_cohere_provider_batch_documents_explicit_1024_dimension(mock_cohere_settings):
    """Verify batched document embedding passes output_dimension=1024 on all slices."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=3, dim=1024, total_tokens=90))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    chunks = [f"Chunk content section {i}" for i in range(3)]

    result = await provider.embed_batch(chunks, input_type="document")

    assert len(result) == 3
    assert result.dimension == 1024
    for vec in result.embeddings:
        assert len(vec) == 1024

    mock_client.embed.assert_awaited_once_with(
        texts=chunks,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=1024,
    )


@pytest.mark.anyio
async def test_cohere_provider_batch_queries_explicit_1024_dimension(mock_cohere_settings):
    """Verify batched query embedding passes output_dimension=1024 on all slices."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=2, dim=1024, total_tokens=40))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    queries = ["First retrieval query", "Second transformed query"]

    result = await provider.embed_queries(queries)

    assert len(result) == 2
    assert result.dimension == 1024
    for vec in result.embeddings:
        assert len(vec) == 1024

    mock_client.embed.assert_awaited_once_with(
        texts=queries,
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=1024,
    )


# ==============================================================================
# 2. End-to-End Qdrant Integration Tests (1024d Compatibility)
# ==============================================================================


@pytest.mark.anyio
async def test_qdrant_store_accepts_1024_dimension_cohere_vectors(mock_cohere_settings):
    """Verify QdrantVectorStore upserts and queries 1024-dimensional vectors successfully."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    doc_vec = await provider.embed_text("Sample document text", input_type="document")
    assert len(doc_vec) == 1024

    # Setup mock vector store
    mock_qdrant = AsyncMock()
    mock_qdrant.upsert = AsyncMock(return_value=None)

    store = QdrantVectorStore(client=mock_qdrant, settings=mock_cohere_settings)

    test_project_id = str(uuid.uuid4())
    test_doc_id = str(uuid.uuid4())
    test_chunk_id = str(uuid.uuid4())

    from storage.vector.models import VectorRecord

    record = VectorRecord(
        id=test_chunk_id,
        vector=doc_vec,
        payload=VectorPayload(
            project_id=test_project_id,
            document_id=test_doc_id,
            chunk_id=test_chunk_id,
        ),
    )

    # Upsert into vector store
    await store.upsert([record])

    # Assert upsert was called with the 1024-dimensional vector
    assert mock_qdrant.upsert.await_count == 1
    call_kwargs = mock_qdrant.upsert.call_args.kwargs
    points = call_kwargs["points"]
    assert len(points) == 1
    assert len(points[0].vector) == 1024
    assert points[0].payload["project_id"] == test_project_id


@pytest.mark.anyio
async def test_query_embedding_service_produces_1024d_compatible_with_retrieval(mock_cohere_settings):
    """Verify QueryEmbeddingService produces 1024d embedded query set compatible with retrieval."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    service = QueryEmbeddingService(provider=provider, settings=mock_cohere_settings)

    query_set = RetrievalQuerySet(original_query="Architecture overview query")
    embedded_set = await service.embed_query_set(query_set)

    assert len(embedded_set.original.vector) == 1024
    assert embedded_set.original.model == "embed-v4.0"
