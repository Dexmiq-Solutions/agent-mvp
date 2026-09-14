"""Comprehensive unit and behavioral contract tests for Retrieval Query Embedding."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.config import Settings
from rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingAuthenticationError,
    EmbeddingBatchResult,
    EmbeddingConnectionError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    RedisEmbeddingCache,
    VoyageEmbeddingProvider,
    generate_embedding_cache_key,
    get_embedding_provider,
    reset_embedding_provider,
    reset_redis_embedding_cache,
)
from exceptions.retrieval import QueryEmbeddingError
from rag.retrieval import (
    EmbeddedQuery as RagEmbeddedQuery,
    EmbeddedQuerySet as RagEmbeddedQuerySet,
    QueryEmbeddingService as RagQueryEmbeddingService,
    embed_query_set as rag_embed_query_set,
    get_query_embedding_service as rag_get_query_embedding_service,
    reset_query_embedding_service as rag_reset_query_embedding_service,
)
from rag.retrieval import (
    EmbeddedQuery,
    EmbeddedQuerySet,
    QueryEmbeddingService,
    RetrievalQuerySet,
    embed_query_set,
    get_query_embedding_service,
    reset_query_embedding_service,
)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider for contract testing."""

    def __init__(
        self,
        model: str = "voyage-4",
        canned_vectors: list[list[float]] | None = None,
        should_fail_with: Exception | None = None,
        fail_attempts: int = 0,
    ) -> None:
        self._model = model
        self.canned_vectors = canned_vectors or [[0.1, 0.2, 0.3, 0.4]]
        self.should_fail_with = should_fail_with
        self.fail_attempts = fail_attempts
        self.call_count = 0
        self.batch_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_text(self, text: str, input_type: str | None = None) -> list[float]:
        res = await self.embed_batch([text], input_type=input_type)
        return res.embeddings[0]

    async def embed_batch(
        self, texts: list[str], input_type: str | None = None
    ) -> EmbeddingBatchResult:
        self.call_count += 1
        self.batch_calls.append(list(texts))

        if self.should_fail_with:
            if self.fail_attempts == 0 or self.call_count <= self.fail_attempts:
                raise self.should_fail_with

        # Return vectors matching the input size
        if len(self.canned_vectors) >= len(texts):
            vectors = self.canned_vectors[: len(texts)]
        else:
            vectors = [list(self.canned_vectors[0]) for _ in texts]

        return EmbeddingBatchResult(
            embeddings=vectors,
            model=self._model,
            total_tokens=len(texts) * 5,
        )


class MockRedisClient:
    """Mock Redis client for testing cache hits, misses, and failure modes."""

    def __init__(
        self,
        initial_data: dict[str, str] | None = None,
        fail_on_get: bool = False,
        fail_on_set: bool = False,
    ) -> None:
        self.storage: dict[str, str] = dict(initial_data or {})
        self.fail_on_get = fail_on_get
        self.fail_on_set = fail_on_set
        self.get_count = 0
        self.set_count = 0

    async def get(self, key: str) -> str | None:
        self.get_count += 1
        if self.fail_on_get:
            raise ConnectionError("Redis server unreachable on read")
        return self.storage.get(key)

    async def mget(self, keys: list[str]) -> list[str | None]:
        self.get_count += 1
        if self.fail_on_get:
            raise ConnectionError("Redis server unreachable on mget")
        return [self.storage.get(k) for k in keys]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.set_count += 1
        if self.fail_on_set:
            raise ConnectionError("Redis server unreachable on write")
        self.storage[key] = value


@pytest.fixture(autouse=True)
def cleanup_embedding_services():
    """Reset singletons before and after each test."""
    reset_embedding_provider()
    reset_redis_embedding_cache()
    reset_query_embedding_service()
    rag_reset_query_embedding_service()
    yield
    reset_embedding_provider()
    reset_redis_embedding_cache()
    reset_query_embedding_service()
    rag_reset_query_embedding_service()


# ==============================================================================
# 1. Single Query Embedding Tests
# ==============================================================================


class TestSingleQueryEmbedding:
    """Tests verifying behavior on a single retrieval query."""

    @pytest.mark.asyncio
    async def test_single_query_produces_single_vector(self):
        """One retrieval query produces one corresponding vector under original."""
        mock_provider = MockEmbeddingProvider(
            canned_vectors=[[0.11, 0.22, 0.33, 0.44]],
        )
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(original_query="What is the OAuth flow?")
        result = await service.embed_query_set(query_set)

        assert isinstance(result, EmbeddedQuerySet)
        assert isinstance(result.original, EmbeddedQuery)
        assert result.original.query == "What is the OAuth flow?"
        assert result.original.vector == [0.11, 0.22, 0.33, 0.44]
        assert result.original.query_type == "original"
        assert result.original.model == "voyage-4"
        assert result.transformed is None

        # Verify provider was called with batch of 1
        assert mock_provider.call_count == 1
        assert mock_provider.batch_calls == [["What is the OAuth flow?"]]

    @pytest.mark.asyncio
    async def test_convenience_function_embed_query_set(self):
        """Functional embed_query_set helper works seamlessly."""
        mock_provider = MockEmbeddingProvider(
            canned_vectors=[[0.5, 0.6, 0.7, 0.8]],
        )
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(original_query="Sample query")
        result = await embed_query_set(query_set, service=service)

        assert result.original.query == "Sample query"
        assert result.original.vector == [0.5, 0.6, 0.7, 0.8]


# ==============================================================================
# 2. Multiple Query Representations (Original + Transformed)
# ==============================================================================


class TestMultipleQueryEmbedding:
    """Tests verifying behavior when both original and transformed queries are present."""

    @pytest.mark.asyncio
    async def test_multiple_queries_batched_in_single_call(self):
        """Original + transformed queries are embedded via a single provider batch call."""
        mock_provider = MockEmbeddingProvider(
            canned_vectors=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(
            original_query="How to configure it?",
            transformed_query="How to configure Qdrant collection?",
            is_transformed=True,
            strategy_used="llm_rewrite",
        )

        result = await service.embed_query_set(query_set)

        assert isinstance(result, EmbeddedQuerySet)
        # Verify original mapping
        assert result.original.query == "How to configure it?"
        assert result.original.vector == [0.1, 0.2, 0.3]
        assert result.original.query_type == "original"
        assert result.original.model == "voyage-4"

        # Verify transformed mapping
        assert result.transformed is not None
        assert result.transformed.query == "How to configure Qdrant collection?"
        assert result.transformed.vector == [0.4, 0.5, 0.6]
        assert result.transformed.query_type == "transformed"
        assert result.transformed.model == "voyage-4"

        # Strictly 1 batch provider call made
        assert mock_provider.call_count == 1
        assert mock_provider.batch_calls == [
            ["How to configure it?", "How to configure Qdrant collection?"]
        ]

    @pytest.mark.asyncio
    async def test_untransformed_set_ignores_duplicate_transformed_query(self):
        """If is_transformed is False, transformed query is not embedded even if field populated."""
        mock_provider = MockEmbeddingProvider(canned_vectors=[[0.1, 0.2]])
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(
            original_query="What is the SLA?",
            transformed_query="What is the SLA?",
            is_transformed=False,
        )
        result = await service.embed_query_set(query_set)

        assert result.transformed is None
        assert mock_provider.call_count == 1
        assert mock_provider.batch_calls == [["What is the SLA?"]]


# ==============================================================================
# 3. Shared Redis Cache & Resilience Tests
# ==============================================================================


class TestRedisEmbeddingCache:
    """Tests verifying cache hit, cache miss, partial hit, and failure tolerance."""

    @pytest.mark.asyncio
    async def test_cache_hit_bypasses_provider(self):
        """A cached query embedding completely skips the provider call."""
        key = generate_embedding_cache_key(
            text="Cached user query",
            model="voyage-4",
            provider="voyage",
            input_type="query",
        )
        mock_redis = MockRedisClient(initial_data={key: "[0.99, 0.88, 0.77]"})
        cache = RedisEmbeddingCache(client=mock_redis)

        mock_provider = MockEmbeddingProvider()
        service = QueryEmbeddingService(provider=mock_provider, cache=cache)

        query_set = RetrievalQuerySet(original_query="Cached user query")
        result = await service.embed_query_set(query_set)

        assert result.original.vector == [0.99, 0.88, 0.77]
        # Provider never called on cache hit
        assert mock_provider.call_count == 0

    @pytest.mark.asyncio
    async def test_cache_miss_calls_provider_and_writes_cache(self):
        """Cache miss fetches from provider and populates cache."""
        mock_redis = MockRedisClient()
        cache = RedisEmbeddingCache(client=mock_redis)

        mock_provider = MockEmbeddingProvider(canned_vectors=[[0.12, 0.34, 0.56]])
        service = QueryEmbeddingService(provider=mock_provider, cache=cache)

        query_set = RetrievalQuerySet(original_query="Uncached query")
        result = await service.embed_query_set(query_set)

        assert result.original.vector == [0.12, 0.34, 0.56]
        assert mock_provider.call_count == 1
        assert mock_redis.set_count == 1

        # Second invocation is now a cache hit
        result2 = await service.embed_query_set(query_set)
        assert result2.original.vector == [0.12, 0.34, 0.56]
        assert mock_provider.call_count == 1  # Still 1, not 2!

    @pytest.mark.asyncio
    async def test_partial_cache_hit_batches_only_uncached_queries(self):
        """When 1 of 2 queries is cached, only the uncached query is sent to provider."""
        orig_key = generate_embedding_cache_key(
            text="Query 1",
            model="voyage-4",
            provider="voyage",
            input_type="query",
        )
        mock_redis = MockRedisClient(initial_data={orig_key: "[0.1, 0.2]"})
        cache = RedisEmbeddingCache(client=mock_redis)

        mock_provider = MockEmbeddingProvider(canned_vectors=[[0.8, 0.9]])
        service = QueryEmbeddingService(provider=mock_provider, cache=cache)

        query_set = RetrievalQuerySet(
            original_query="Query 1",
            transformed_query="Query 2",
            is_transformed=True,
        )
        result = await service.embed_query_set(query_set)

        assert result.original.vector == [0.1, 0.2]
        assert result.transformed is not None
        assert result.transformed.vector == [0.8, 0.9]

        # Only "Query 2" was sent to provider
        assert mock_provider.call_count == 1
        assert mock_provider.batch_calls == [["Query 2"]]

    @pytest.mark.asyncio
    async def test_redis_failure_degrades_gracefully(self):
        """Redis read/write connection errors do not prevent successful embedding."""
        mock_redis = MockRedisClient(fail_on_get=True, fail_on_set=True)
        cache = RedisEmbeddingCache(client=mock_redis)

        mock_provider = MockEmbeddingProvider(canned_vectors=[[0.42, 0.43]])
        service = QueryEmbeddingService(provider=mock_provider, cache=cache)

        query_set = RetrievalQuerySet(original_query="Query despite redis down")
        result = await service.embed_query_set(query_set)

        assert result.original.vector == [0.42, 0.43]
        assert mock_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_deterministic_cache_key_generation(self):
        """Cache keys safely differentiate text, model, input_type, and dimensions."""
        key1 = generate_embedding_cache_key("text A", model="voyage-4", input_type="query")
        key2 = generate_embedding_cache_key("text B", model="voyage-4", input_type="query")
        key_doc = generate_embedding_cache_key("text A", model="voyage-4", input_type="document")
        key_model = generate_embedding_cache_key("text A", model="other-model", input_type="query")

        assert key1 != key2
        assert key1 != key_doc
        assert key1 != key_model
        assert key1.startswith("embedding:voyage:voyage-4:query:")


# ==============================================================================
# 4. Error Handling & Bounded Retries
# ==============================================================================


class TestErrorHandlingAndRetries:
    """Tests verifying error handling, transient retries, and permanent failure fail-fast."""

    @pytest.mark.asyncio
    async def test_transient_rate_limit_retries_and_succeeds(self):
        """Transient rate limit error is retried and succeeds on second attempt."""
        mock_provider = MockEmbeddingProvider(
            canned_vectors=[[0.1, 0.2]],
            should_fail_with=EmbeddingRateLimitError("Rate limit exceeded"),
            fail_attempts=1,
        )
        service = QueryEmbeddingService(
            provider=mock_provider,
            max_retries=2,
            retry_delay=0.01,
        )

        query_set = RetrievalQuerySet(original_query="Query with transient rate limit")
        result = await service.embed_query_set(query_set)

        assert result.original.vector == [0.1, 0.2]
        assert mock_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_transient_connection_error_exceeds_max_retries(self):
        """Persistent connection errors fail with QueryEmbeddingError after bounded retries."""
        mock_provider = MockEmbeddingProvider(
            should_fail_with=EmbeddingConnectionError("Connection timeout"),
            fail_attempts=5,
        )
        service = QueryEmbeddingService(
            provider=mock_provider,
            max_retries=1,
            retry_delay=0.01,
        )

        query_set = RetrievalQuerySet(original_query="Connection failing query")
        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(query_set)

        assert "Query embedding failed after 2 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_permanent_authentication_error_fails_fast(self):
        """Permanent auth errors fail immediately on first attempt without retrying."""
        mock_provider = MockEmbeddingProvider(
            should_fail_with=EmbeddingAuthenticationError("Invalid API key"),
        )
        service = QueryEmbeddingService(
            provider=mock_provider,
            max_retries=3,
            retry_delay=0.01,
        )

        query_set = RetrievalQuerySet(original_query="Auth failing query")
        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(query_set)

        assert "Permanent embedding failure" in str(exc_info.value)
        # Never retried
        assert mock_provider.call_count == 1


# ==============================================================================
# 5. Response Validation Tests
# ==============================================================================


class TestResponseValidation:
    """Tests verifying strict validation on provider outputs."""

    @pytest.mark.asyncio
    async def test_mismatched_vector_count_rejected(self):
        """Provider returning fewer or more vectors than requested is rejected."""
        mock_provider = MockEmbeddingProvider(
            canned_vectors=[[0.1, 0.2]],  # Only 1 vector returned for 2 queries
        )
        # Force provider to return only 1 vector for 2 queries
        mock_provider.embed_batch = AsyncMock(
            return_value=EmbeddingBatchResult(
                embeddings=[[0.1, 0.2]],
                model="voyage-4",
            )
        )
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(
            original_query="Query 1",
            transformed_query="Query 2",
            is_transformed=True,
        )

        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(query_set)

        assert "Mismatched embedding count" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_vector_rejected(self):
        """Provider returning an empty vector is rejected."""
        mock_provider = MockEmbeddingProvider()
        mock_provider.embed_batch = AsyncMock(
            return_value=EmbeddingBatchResult(
                embeddings=[[]],
                model="voyage-4",
            )
        )
        service = QueryEmbeddingService(provider=mock_provider)

        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(RetrievalQuerySet(original_query="Q"))

        assert "vector cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_non_finite_vector_values_rejected(self):
        """Provider returning NaN or Inf in vectors is rejected."""
        mock_provider = MockEmbeddingProvider()
        mock_provider.embed_batch = AsyncMock(
            return_value=EmbeddingBatchResult(
                embeddings=[[0.1, float("nan"), 0.3]],
                model="voyage-4",
            )
        )
        service = QueryEmbeddingService(provider=mock_provider)

        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(RetrievalQuerySet(original_query="Q"))

        assert "non-finite" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_inconsistent_vector_dimensions_rejected(self):
        """Vectors of differing dimensions within the same batch are rejected."""
        mock_provider = MockEmbeddingProvider()
        mock_provider.embed_batch = AsyncMock(
            return_value=EmbeddingBatchResult(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5]],  # 3 dims vs 2 dims
                model="voyage-4",
            )
        )
        service = QueryEmbeddingService(provider=mock_provider)

        query_set = RetrievalQuerySet(
            original_query="Q1",
            transformed_query="Q2",
            is_transformed=True,
        )
        with pytest.raises(QueryEmbeddingError) as exc_info:
            await service.embed_query_set(query_set)

        assert "dimensionality" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_input_type_rejected(self):
        """Passing non-RetrievalQuerySet object raises QueryEmbeddingError."""
        service = QueryEmbeddingService()
        with pytest.raises(QueryEmbeddingError):
            await service.embed_query_set("raw string query")  # type: ignore


# ==============================================================================
# 6. Factory & Package Integration Tests
# ==============================================================================


class TestFactoryAndPackageIntegration:
    """Tests verifying singleton factories and package re-exports."""

    def test_get_query_embedding_service_singleton(self):
        """get_query_embedding_service returns cached singleton instance."""
        s1 = get_query_embedding_service()
        s2 = get_query_embedding_service()
        assert s1 is s2

        reset_query_embedding_service()
        s3 = get_query_embedding_service()
        assert s3 is not s1

    def test_rag_retrieval_alias_exports_match(self):
        """rag.retrieval re-exports identical classes and helpers."""
        assert RagEmbeddedQuery is EmbeddedQuery
        assert RagEmbeddedQuerySet is EmbeddedQuerySet
        assert RagQueryEmbeddingService is QueryEmbeddingService
        assert rag_get_query_embedding_service is get_query_embedding_service
        assert rag_embed_query_set is embed_query_set


# ==============================================================================
# 7. Centralized Model Configuration Tests
# ==============================================================================


class TestCentralizedModelConfigurationPath:
    """Tests proving Query Embedding dynamically consumes centralized Settings.EMBEDDING_MODEL.

    Confirms that changing the centralized model configuration propagates to Query Embedding
    without any query-specific model constants or divergence from document embedding.
    """

    @pytest.mark.asyncio
    async def test_query_embedding_dynamically_consumes_settings_model(self):
        """Query embedding resolves and uses custom Settings.EMBEDDING_MODEL dynamically."""
        custom_model = "voyage-custom-future-model"
        custom_settings = Settings(
            VOYAGE_API_KEY="mock-voyage-key",
            EMBEDDING_MODEL=custom_model,
        )

        # Mock voyage client that inspects the model passed to embed()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_voyage_client = AsyncMock()
        mock_voyage_client.embed = AsyncMock(return_value=mock_response)

        # Instantiate provider and service with custom settings
        provider = VoyageEmbeddingProvider(
            client=mock_voyage_client,
            settings=custom_settings,
        )
        service = QueryEmbeddingService(
            provider=provider,
            settings=custom_settings,
        )

        assert service.model_name == custom_model

        query_set = RetrievalQuerySet(original_query="Dynamic configuration query")
        result = await service.embed_query_set(query_set)

        # Vector result carries the centralized model name
        assert result.original.model == custom_model

        # Provider's client.embed was invoked with the exact centralized model
        mock_voyage_client.embed.assert_awaited_once_with(
            texts=["Dynamic configuration query"],
            model=custom_model,
            input_type="query",
        )

    def test_document_and_query_embedding_share_exact_same_model(self):
        """Document embedding provider and query embedding provider share the exact same model."""
        custom_model = "voyage-shared-unified-model"
        custom_settings = Settings(
            VOYAGE_API_KEY="mock-voyage-key",
            EMBEDDING_MODEL=custom_model,
        )

        doc_provider = get_embedding_provider(settings=custom_settings)
        query_service = QueryEmbeddingService(settings=custom_settings)

        assert doc_provider.model_name == custom_model
        assert query_service.model_name == custom_model
        assert query_service.provider.model_name == doc_provider.model_name

    def test_default_configured_model_is_voyage_4(self):
        """Authoritative default embedding model across the application is voyage-4."""
        default_settings = Settings()
        assert default_settings.EMBEDDING_MODEL == "voyage-4"
        provider = VoyageEmbeddingProvider(settings=default_settings, client=AsyncMock())
        assert provider.model_name == "voyage-4"
        service = QueryEmbeddingService(provider=provider, settings=default_settings)
        assert service.model_name == "voyage-4"
