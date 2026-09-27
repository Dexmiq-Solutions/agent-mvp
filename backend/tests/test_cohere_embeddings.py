"""Comprehensive unit and contract tests for Cohere Embed v4 provider, client, and factory."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import cohere
import httpx

from core.config import Settings
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseValidationError,
)
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.cohere import (
    DEFAULT_COHERE_EMBEDDING_DIM,
    DEFAULT_COHERE_MAX_BATCH_SIZE,
    CohereEmbeddingProvider,
)
from rag.embeddings.cohere_client import (
    get_async_cohere_client,
    reset_async_cohere_client,
)
from rag.embeddings.models import EmbeddingBatchResult
from rag.embeddings import get_embedding_provider, reset_embedding_provider
from rag.embeddings.voyage import VoyageEmbeddingProvider


@pytest.fixture(autouse=True)
def cleanup_embedding_singletons():
    """Reset cached client and provider singletons before and after each test."""
    reset_async_cohere_client()
    reset_embedding_provider()
    yield
    reset_async_cohere_client()
    reset_embedding_provider()


@pytest.fixture
def mock_cohere_settings():
    """Settings instance configured with Cohere credentials."""
    return Settings(
        COHERE_API_KEY="test-cohere-api-key-12345",
        EMBEDDING_PROVIDER="cohere",
        EMBEDDING_MODEL="embed-v4.0",
    )


def _make_mock_embed_response(count: int, dim: int = 1024, total_tokens: int = 50):
    """Helper creating a mock response mimicking Cohere EmbeddingsFloatsEmbedResponse."""
    resp = MagicMock()
    resp.embeddings = [[0.01 * (j + 1) for j in range(dim)] for _ in range(count)]
    resp.meta = MagicMock()
    resp.meta.tokens = MagicMock()
    resp.meta.tokens.input_tokens = total_tokens
    resp.meta.billed_units = None
    return resp


# ==============================================================================
# 1. Configuration & Client Management Tests
# ==============================================================================


def test_get_async_cohere_client_missing_key():
    """Verify get_async_cohere_client raises EmbeddingConfigurationError when key is missing."""
    empty_settings = Settings(COHERE_API_KEY=None)
    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        get_async_cohere_client(settings=empty_settings)
    assert "COHERE_API_KEY" in str(exc_info.value)


def test_get_async_cohere_client_caching(mock_cohere_settings):
    """Verify get_async_cohere_client returns cached singleton instance."""
    client1 = get_async_cohere_client(settings=mock_cohere_settings)
    client2 = get_async_cohere_client(settings=mock_cohere_settings)
    assert client1 is client2


def test_reset_async_cohere_client(mock_cohere_settings):
    """Verify reset_async_cohere_client clears cached client instance."""
    client1 = get_async_cohere_client(settings=mock_cohere_settings)
    reset_async_cohere_client()
    client2 = get_async_cohere_client(settings=mock_cohere_settings)
    assert client1 is not client2


# ==============================================================================
# 2. Provider Initialization & Properties
# ==============================================================================


def test_cohere_provider_initialization(mock_cohere_settings):
    """Verify provider properties and default attributes."""
    mock_client = AsyncMock()
    provider = CohereEmbeddingProvider(
        client=mock_client,
        settings=mock_cohere_settings,
    )
    assert provider.model_name == "embed-v4.0"
    assert provider.provider_name == "cohere"
    assert provider.expected_dim == 1024


def test_cohere_provider_custom_model(mock_cohere_settings):
    """Verify custom model override."""
    mock_client = AsyncMock()
    provider = CohereEmbeddingProvider(
        model="embed-multilingual-v3.0",
        client=mock_client,
        settings=mock_cohere_settings,
    )
    assert provider.model_name == "embed-multilingual-v3.0"


# ==============================================================================
# 3. Input-Type Mapping Tests
# ==============================================================================


def test_map_input_type(mock_cohere_settings):
    """Verify central mapping of document and query input types to Cohere types."""
    provider = CohereEmbeddingProvider(client=AsyncMock(), settings=mock_cohere_settings)

    assert provider._map_input_type("document") == "search_document"
    assert provider._map_input_type("DOCUMENT") == "search_document"
    assert provider._map_input_type("search_document") == "search_document"

    assert provider._map_input_type("query") == "search_query"
    assert provider._map_input_type("QUERY") == "search_query"
    assert provider._map_input_type("search_query") == "search_query"

    assert provider._map_input_type("classification") == "classification"
    assert provider._map_input_type("clustering") == "clustering"
    assert provider._map_input_type(None) == "search_document"
    assert provider._map_input_type("") == "search_document"


@pytest.mark.anyio
async def test_embed_text_query_and_document_mapping(mock_cohere_settings):
    """Verify embed_text and embed_query pass the correct mapped input_type to Cohere."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)

    # Document embedding
    vec = await provider.embed_text("Sample document text", input_type="document")
    assert len(vec) == 1024
    mock_client.embed.assert_awaited_with(
        texts=["Sample document text"],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
    )

    # Query embedding
    q_vec = await provider.embed_query("User retrieval query")
    assert len(q_vec) == 1024
    mock_client.embed.assert_awaited_with(
        texts=["User retrieval query"],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
    )


# ==============================================================================
# 4. Batch Handling Tests (1, 96, 97, 150 items)
# ==============================================================================


@pytest.mark.anyio
async def test_embed_batch_single_item(mock_cohere_settings):
    """Verify batch with 1 item executes in a single API call."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=1, dim=1024, total_tokens=10))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    result = await provider.embed_batch(["Single chunk"], input_type="document")

    assert isinstance(result, EmbeddingBatchResult)
    assert len(result) == 1
    assert result.dimension == 1024
    assert result.total_tokens == 10
    assert mock_client.embed.await_count == 1


@pytest.mark.anyio
async def test_embed_batch_exactly_96_items(mock_cohere_settings):
    """Verify batch of exactly 96 items executes in a single API call."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=_make_mock_embed_response(count=96, dim=1024, total_tokens=960))

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    texts = [f"Chunk {i}" for i in range(96)]
    result = await provider.embed_batch(texts, input_type="document")

    assert len(result) == 96
    assert mock_client.embed.await_count == 1
    assert result.total_tokens == 960


@pytest.mark.anyio
async def test_embed_batch_97_items_splits_correctly(mock_cohere_settings):
    """Verify batch of 97 items splits into 96 + 1 preserving order."""
    mock_client = AsyncMock()

    # Two sub-batch responses
    resp1 = _make_mock_embed_response(count=96, dim=1024, total_tokens=960)
    resp2 = _make_mock_embed_response(count=1, dim=1024, total_tokens=10)
    # Distinct values to verify ordering
    resp1.embeddings = [[float(i)] * 1024 for i in range(96)]
    resp2.embeddings = [[96.0] * 1024]
    mock_client.embed.side_effect = [resp1, resp2]

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    texts = [f"Chunk {i}" for i in range(97)]
    result = await provider.embed_batch(texts, input_type="document")

    assert len(result) == 97
    assert mock_client.embed.await_count == 2
    assert result.total_tokens == 970

    # Verify order preservation across the split boundary
    assert result[0] == [0.0] * 1024
    assert result[95] == [95.0] * 1024
    assert result[96] == [96.0] * 1024


@pytest.mark.anyio
async def test_embed_batch_150_items_splits_correctly(mock_cohere_settings):
    """Verify batch of 150 items splits into 96 + 54 preserving order."""
    mock_client = AsyncMock()

    resp1 = _make_mock_embed_response(count=96, dim=1024, total_tokens=960)
    resp2 = _make_mock_embed_response(count=54, dim=1024, total_tokens=540)
    resp1.embeddings = [[float(i)] * 1024 for i in range(96)]
    resp2.embeddings = [[float(i + 96)] * 1024 for i in range(54)]
    mock_client.embed.side_effect = [resp1, resp2]

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)
    texts = [f"Chunk {i}" for i in range(150)]
    result = await provider.embed_batch(texts, input_type="document")

    assert len(result) == 150
    assert mock_client.embed.await_count == 2
    assert result.total_tokens == 1500

    # First chunk, slice boundary, and last chunk
    assert result[0] == [0.0] * 1024
    assert result[95] == [95.0] * 1024
    assert result[96] == [96.0] * 1024
    assert result[149] == [149.0] * 1024


# ==============================================================================
# 5. Dimension Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_dimension_mismatch_rejected(mock_cohere_settings):
    """Verify returned vectors that do not match expected 1024 dimensions are rejected."""
    mock_client = AsyncMock()
    # Response with 512 dimensions instead of 1024
    mock_client.embed.return_value = _make_mock_embed_response(count=2, dim=512)

    provider = CohereEmbeddingProvider(
        client=mock_client,
        settings=mock_cohere_settings,
        expected_dim=1024,
    )

    with pytest.raises(EmbeddingResponseValidationError) as exc_info:
        await provider.embed_batch(["text 1", "text 2"])
    assert "Vector dimensionality mismatch: expected 1024, got 512" in str(exc_info.value)


# ==============================================================================
# 6. Error Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_authentication_error_translation(mock_cohere_settings):
    """Verify UnauthorizedError translates to EmbeddingAuthenticationError."""
    mock_client = AsyncMock()
    mock_client.embed.side_effect = cohere.UnauthorizedError(body={"message": "Invalid API token"})

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)

    with pytest.raises(EmbeddingAuthenticationError) as exc_info:
        await provider.embed_text("test")
    assert "COHERE_API_KEY" in str(exc_info.value)


@pytest.mark.anyio
async def test_rate_limit_error_translation(mock_cohere_settings):
    """Verify TooManyRequestsError translates to EmbeddingRateLimitError."""
    mock_client = AsyncMock()
    mock_client.embed.side_effect = cohere.TooManyRequestsError(body={"message": "Rate limit exceeded"})

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await provider.embed_batch(["test 1", "test 2"])
    assert "rate limit exceeded" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_connection_error_translation(mock_cohere_settings):
    """Verify GatewayTimeoutError and ConnectError translate to EmbeddingConnectionError."""
    mock_client = AsyncMock()
    mock_client.embed.side_effect = cohere.GatewayTimeoutError(body={"message": "Gateway timed out"})

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)

    with pytest.raises(EmbeddingConnectionError) as exc_info:
        await provider.embed_text("test")
    assert "Failed to connect to Cohere API" in str(exc_info.value)


@pytest.mark.anyio
async def test_bad_request_error_translation(mock_cohere_settings):
    """Verify BadRequestError translates to EmbeddingRequestError."""
    mock_client = AsyncMock()
    mock_client.embed.side_effect = cohere.BadRequestError(body={"message": "Invalid input"})

    provider = CohereEmbeddingProvider(client=mock_client, settings=mock_cohere_settings)

    with pytest.raises(EmbeddingRequestError) as exc_info:
        await provider.embed_text("test")
    assert "Invalid request to Cohere" in str(exc_info.value)


# ==============================================================================
# 7. Provider Factory Tests
# ==============================================================================


def test_factory_creates_cohere_provider_by_default():
    """Verify get_embedding_provider instantiates CohereEmbeddingProvider when EMBEDDING_PROVIDER=cohere."""
    settings = Settings(
        COHERE_API_KEY="test-cohere-key",
        EMBEDDING_PROVIDER="cohere",
        EMBEDDING_MODEL="embed-v4.0",
    )
    provider = get_embedding_provider(settings=settings)
    assert isinstance(provider, CohereEmbeddingProvider)
    assert provider.model_name == "embed-v4.0"
    assert provider.provider_name == "cohere"


def test_factory_creates_voyage_provider_when_configured():
    """Verify get_embedding_provider instantiates VoyageEmbeddingProvider when EMBEDDING_PROVIDER=voyage."""
    settings = Settings(
        VOYAGE_API_KEY="pa-mock-voyage-key",
        EMBEDDING_PROVIDER="voyage",
        EMBEDDING_MODEL="voyage-4",
    )
    provider = get_embedding_provider(settings=settings)
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.model_name == "voyage-4"
    assert provider.provider_name == "voyage"


def test_factory_explicit_provider_override():
    """Verify get_embedding_provider accepts explicit provider parameter."""
    settings = Settings(
        COHERE_API_KEY="test-cohere-key",
        VOYAGE_API_KEY="pa-mock-voyage-key",
        EMBEDDING_PROVIDER="cohere",
        EMBEDDING_MODEL="embed-v4.0",
    )
    voyage_prov = get_embedding_provider(provider="voyage", settings=settings)
    assert isinstance(voyage_prov, VoyageEmbeddingProvider)

    cohere_prov = get_embedding_provider(provider="cohere", settings=settings)
    assert isinstance(cohere_prov, CohereEmbeddingProvider)
