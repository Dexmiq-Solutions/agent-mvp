"""Unit tests for Voyage AI embedding provider, client, and models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voyageai.error as voyage_errors

from app.core.config import Settings
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.client import get_async_voyage_client, reset_async_voyage_client
from rag.embeddings.models import EmbeddingBatchResult, EmbeddingResult
from rag.embeddings.voyage import VoyageEmbeddingProvider
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
)
from rag.embeddings import get_embedding_provider, reset_embedding_provider


@pytest.fixture(autouse=True)
def cleanup_embedding_client():
    """Reset cached singleton client and provider before and after each test."""
    reset_async_voyage_client()
    reset_embedding_provider()
    yield
    reset_async_voyage_client()
    reset_embedding_provider()


@pytest.fixture
def mock_embedding_settings():
    """Sample settings with Voyage configuration."""
    return Settings(
        VOYAGE_API_KEY="pa-mock-voyage-key-12345",
        EMBEDDING_MODEL="voyage-3-large",
    )


@pytest.fixture
def mock_embeddings_response():
    """Mock EmbeddingsObject returned by voyageai.AsyncClient.embed."""
    response = MagicMock()
    response.embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    response.total_tokens = 15
    return response


@pytest.fixture
def mock_voyage_client(mock_embeddings_response):
    """Mock voyageai.AsyncClient instance."""
    client = AsyncMock()
    client.embed = AsyncMock(return_value=mock_embeddings_response)
    return client


# ==============================================================================
# 1. Configuration & Client Management Tests
# ==============================================================================


def test_get_async_voyage_client_missing_key():
    """Verify get_async_voyage_client raises EmbeddingConfigurationError when key is missing."""
    empty_settings = Settings(VOYAGE_API_KEY=None)
    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        get_async_voyage_client(settings=empty_settings)
    assert "VOYAGE_API_KEY" in str(exc_info.value)


def test_get_async_voyage_client_caching(mock_embedding_settings):
    """Verify get_async_voyage_client returns cached singleton client."""
    client1 = get_async_voyage_client(settings=mock_embedding_settings)
    client2 = get_async_voyage_client(settings=mock_embedding_settings)

    assert client1 is client2


def test_reset_async_voyage_client(mock_embedding_settings):
    """Verify reset_async_voyage_client clears cached client instance."""
    client1 = get_async_voyage_client(settings=mock_embedding_settings)
    reset_async_voyage_client()
    client2 = get_async_voyage_client(settings=mock_embedding_settings)

    assert client1 is not client2


# ==============================================================================
# 2. Input Validation Tests
# ==============================================================================


def test_validate_text_invalid_inputs(mock_embedding_settings):
    """Verify single text validation catches empty strings and non-string types."""
    provider = VoyageEmbeddingProvider(settings=mock_embedding_settings)

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_text("")

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_text("   \n\t  ")

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_text(None)

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_text(12345)


def test_validate_batch_invalid_inputs(mock_embedding_settings):
    """Verify batch validation catches empty lists, invalid types, and blank elements."""
    provider = VoyageEmbeddingProvider(settings=mock_embedding_settings)

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_batch([])

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_batch("not a list")

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_batch(["valid", ""])

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_batch(["valid", None])

    with pytest.raises(EmbeddingInputValidationError):
        provider._validate_batch(["valid", 123])


# ==============================================================================
# 3. Asynchronous Embedding Operations Tests
# ==============================================================================


@pytest.mark.anyio
async def test_embed_text_success(mock_embedding_settings, mock_voyage_client):
    """Verify single text embedding returns a single vector."""
    single_response = MagicMock()
    single_response.embeddings = [[0.11, 0.22, 0.33]]
    single_response.total_tokens = 5
    mock_voyage_client.embed.return_value = single_response

    provider = VoyageEmbeddingProvider(
        client=mock_voyage_client,
        settings=mock_embedding_settings,
    )

    vector = await provider.embed_text("Sample query for RAG", input_type="query")
    assert vector == [0.11, 0.22, 0.33]
    mock_voyage_client.embed.assert_awaited_once_with(
        texts=["Sample query for RAG"],
        model="voyage-3-large",
        input_type="query",
    )


@pytest.mark.anyio
async def test_embed_batch_success_and_order_preservation(
    mock_embedding_settings, mock_voyage_client, mock_embeddings_response
):
    """Verify batch embedding preserves 1-to-1 input-to-output order and returns EmbeddingBatchResult."""
    provider = VoyageEmbeddingProvider(
        client=mock_voyage_client,
        settings=mock_embedding_settings,
    )

    input_texts = ["First chunk of text", "Second chunk of text"]
    result = await provider.embed_batch(input_texts, input_type="document")

    assert isinstance(result, EmbeddingBatchResult)
    assert len(result) == 2
    assert result.model == "voyage-3-large"
    assert result.total_tokens == 15
    assert result.dimension == 4

    # Verify order preservation
    assert result[0] == [0.1, 0.2, 0.3, 0.4]
    assert result[1] == [0.5, 0.6, 0.7, 0.8]

    # Verify iteration
    vectors = list(result)
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3, 0.4]


# ==============================================================================
# 4. Error Handling & Exception Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_embed_authentication_error(mock_embedding_settings, mock_voyage_client):
    """Verify AuthenticationError is translated to EmbeddingAuthenticationError."""
    mock_voyage_client.embed.side_effect = voyage_errors.AuthenticationError(
        message="Invalid API Key provided"
    )
    provider = VoyageEmbeddingProvider(client=mock_voyage_client, settings=mock_embedding_settings)

    with pytest.raises(EmbeddingAuthenticationError) as exc_info:
        await provider.embed_text("test text")
    assert "Authentication failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_embed_rate_limit_error(mock_embedding_settings, mock_voyage_client):
    """Verify RateLimitError is translated to EmbeddingRateLimitError."""
    mock_voyage_client.embed.side_effect = voyage_errors.RateLimitError(
        message="Rate limit exceeded for TPM"
    )
    provider = VoyageEmbeddingProvider(client=mock_voyage_client, settings=mock_embedding_settings)

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await provider.embed_batch(["text1", "text2"])
    assert "rate limit exceeded" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_embed_connection_error(mock_embedding_settings, mock_voyage_client):
    """Verify connection errors and timeouts translate to EmbeddingConnectionError."""
    mock_voyage_client.embed.side_effect = voyage_errors.APIConnectionError(
        message="Connection refused to api.voyageai.com"
    )
    provider = VoyageEmbeddingProvider(client=mock_voyage_client, settings=mock_embedding_settings)

    with pytest.raises(EmbeddingConnectionError) as exc_info:
        await provider.embed_text("test text")
    assert "Failed to connect" in str(exc_info.value)


@pytest.mark.anyio
async def test_embed_invalid_request_error(mock_embedding_settings, mock_voyage_client):
    """Verify InvalidRequestError is translated to EmbeddingRequestError."""
    mock_voyage_client.embed.side_effect = voyage_errors.InvalidRequestError(
        message="Model context length exceeded"
    )
    provider = VoyageEmbeddingProvider(client=mock_voyage_client, settings=mock_embedding_settings)

    with pytest.raises(EmbeddingRequestError) as exc_info:
        await provider.embed_batch(["oversized text chunk"])

    assert "Invalid request" in str(exc_info.value)


# ==============================================================================
# 5. Model Classes & Factory Tests
# ==============================================================================


def test_embedding_result_models():
    """Verify EmbeddingResult dataclass attributes and dimension property."""
    res = EmbeddingResult(vector=[0.1, 0.2, 0.3], model="voyage-3-large", token_count=10)
    assert res.vector == [0.1, 0.2, 0.3]
    assert res.model == "voyage-3-large"
    assert res.token_count == 10
    assert res.dimension == 3


def test_get_embedding_provider_factory(mock_embedding_settings):
    """Verify get_embedding_provider returns BaseEmbeddingProvider with singleton caching."""
    provider1 = get_embedding_provider(settings=mock_embedding_settings)
    provider2 = get_embedding_provider(settings=mock_embedding_settings)

    assert isinstance(provider1, BaseEmbeddingProvider)
    assert provider1 is provider2
    assert provider1.model_name == "voyage-3-large"

    # Custom model returns a distinct instance
    custom_provider = get_embedding_provider(
        model="voyage-code-2", settings=mock_embedding_settings
    )
    assert custom_provider.model_name == "voyage-code-2"
    assert custom_provider is not provider1
