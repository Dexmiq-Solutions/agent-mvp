"""Query Embedding service converting RetrievalQuerySet representations into dense vectors."""

import asyncio
import time
from typing import Optional

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseValidationError,
    RedisEmbeddingCache,
    generate_embedding_cache_key,
    get_embedding_provider,
    get_redis_embedding_cache,
    validate_embedding_batch,
)
from exceptions.retrieval import QueryEmbeddingError
from rag.retrieval.models import EmbeddedQuery, EmbeddedQuerySet, RetrievalQuerySet

logger = get_logger(__name__)

_query_embedding_service: Optional["QueryEmbeddingService"] = None


class QueryEmbeddingService:
    """Service responsible exclusively for converting RetrievalQuerySet representations into embedding vectors.

    Consumes the RetrievalQuerySet produced by Query Transformation, queries the shared
    Redis embedding cache, batches uncached representations into a single provider call
    via BaseEmbeddingProvider, validates response vectors, and preserves 1-to-1 query
    correspondence inside a minimal EmbeddedQuerySet for downstream Vector Search.
    """

    def __init__(
        self,
        provider: Optional[BaseEmbeddingProvider] = None,
        cache: Optional[RedisEmbeddingCache] = None,
        settings: Optional[Settings] = None,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize QueryEmbeddingService with shared infrastructure.

        Args:
            provider: Pre-configured BaseEmbeddingProvider. Defaults to application provider.
            cache: Pre-configured RedisEmbeddingCache. Defaults to shared Redis cache.
            settings: Application settings. Defaults to cached app settings.
            max_retries: Maximum bounded retries for transient provider failures.
            retry_delay: Initial retry backoff delay in seconds.
            retry_backoff: Exponential backoff multiplier.
        """
        self._settings = settings or get_settings()
        self._provider = provider or get_embedding_provider(settings=self._settings)
        self._cache = cache or get_redis_embedding_cache(settings=self._settings)
        self._cache_enabled = getattr(self._settings, "EMBEDDING_CACHE_ENABLED", True)
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._retry_backoff = max(1.0, retry_backoff)

    @property
    def provider(self) -> BaseEmbeddingProvider:
        """Return the active embedding provider."""
        return self._provider

    @property
    def model_name(self) -> str:
        """Return the active embedding model name from the provider."""
        return self._provider.model_name

    # --------------------------------------------------------------------------
    # Bounded Retries for Provider Calls
    # --------------------------------------------------------------------------

    async def _embed_batch_with_retries(self, queries: list[str]) -> list[list[float]]:
        """Execute batched query embedding with bounded retries on transient errors."""
        attempts = 0
        delay = self._retry_delay

        while True:
            attempts += 1
            try:
                logger.debug(
                    "Embedding batch of %d queries with model '%s' (attempt %d/%d)",
                    len(queries),
                    self._provider.model_name,
                    attempts,
                    self._max_retries + 1,
                )
                batch_result = await self._provider.embed_queries(queries)
                validated = validate_embedding_batch(
                    batch_result.embeddings,
                    expected_count=len(queries),
                )
                return validated
            except (
                EmbeddingAuthenticationError,
                EmbeddingConfigurationError,
                EmbeddingInputValidationError,
                EmbeddingRequestError,
                EmbeddingResponseValidationError,
            ) as exc:
                # Permanent failures: fail fast without retrying
                logger.error("Permanent embedding error on query batch: %s", exc)
                raise QueryEmbeddingError(
                    f"Permanent embedding failure: {exc}",
                    original_error=exc,
                ) from exc
            except (EmbeddingRateLimitError, EmbeddingConnectionError) as exc:
                if attempts > self._max_retries:
                    logger.error(
                        "Exceeded max retries (%d) for query embedding: %s",
                        self._max_retries,
                        exc,
                    )
                    raise QueryEmbeddingError(
                        f"Query embedding failed after {attempts} attempts: {exc}",
                        original_error=exc,
                    ) from exc

                logger.warning(
                    "Transient embedding error (attempt %d/%d): %s. Retrying in %.2fs...",
                    attempts,
                    self._max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= self._retry_backoff
            except EmbeddingError as exc:
                logger.error("Embedding provider failure: %s", exc)
                raise QueryEmbeddingError(
                    f"Embedding provider error: {exc}",
                    original_error=exc,
                ) from exc
            except Exception as exc:
                logger.error("Unexpected error during query embedding: %s", exc)
                raise QueryEmbeddingError(
                    f"Unexpected query embedding failure: {exc}",
                    original_error=exc,
                ) from exc

    # --------------------------------------------------------------------------
    # Core Operation
    # --------------------------------------------------------------------------

    async def embed_query_set(self, query_set: RetrievalQuerySet) -> EmbeddedQuerySet:
        """Convert a RetrievalQuerySet into an EmbeddedQuerySet.

        Args:
            query_set: Validated RetrievalQuerySet produced by Query Transformation.

        Returns:
            EmbeddedQuerySet: Minimal container holding original and optional transformed
                query embeddings in 1-to-1 correspondence.

        Raises:
            QueryEmbeddingError: If query embedding or response validation fails.
        """
        if not isinstance(query_set, RetrievalQuerySet):
            raise QueryEmbeddingError(
                f"Expected RetrievalQuerySet instance, got '{type(query_set).__name__}'."
            )

        start_time = time.perf_counter()
        original_query = query_set.original_query
        has_transformed = query_set.has_transformed_query
        transformed_query = query_set.transformed_query if has_transformed else None

        # Build list of distinct query texts to embed preserving order
        queries_to_embed: list[str] = [original_query]
        if has_transformed and transformed_query:
            queries_to_embed.append(transformed_query)

        model = self._provider.model_name
        provider_name = getattr(
            self._provider,
            "provider_name",
            getattr(self._settings, "EMBEDDING_PROVIDER", "cohere"),
        )
        cached_vectors: dict[str, list[float]] = {}
        uncached_queries: list[str] = []

        # 1. Check Shared Redis Cache
        if self._cache_enabled and self._cache is not None:
            cache_keys = [
                generate_embedding_cache_key(
                    text=q,
                    model=model,
                    provider=provider_name,
                    input_type="query",
                )
                for q in queries_to_embed
            ]
            lookups = await self._cache.get_many(cache_keys)
            for q, vec in zip(queries_to_embed, lookups):
                if vec is not None:
                    cached_vectors[q] = vec
                else:
                    uncached_queries.append(q)
        else:
            uncached_queries = list(queries_to_embed)

        # 2. Embed Uncached Queries in a Single Batch
        if uncached_queries:
            fresh_vectors = await self._embed_batch_with_retries(uncached_queries)

            # Store new vectors in cache (gracefully ignoring cache errors)
            if self._cache_enabled and self._cache is not None:
                cache_items = [
                    (
                        generate_embedding_cache_key(
                            text=q,
                            model=model,
                            provider=provider_name,
                            input_type="query",
                        ),
                        vec,
                    )
                    for q, vec in zip(uncached_queries, fresh_vectors)
                ]
                await self._cache.set_many(cache_items)

            for q, vec in zip(uncached_queries, fresh_vectors):
                cached_vectors[q] = vec

        # 3. Assemble Output Contract Preserving 1-to-1 Mapping
        original_vector = cached_vectors[original_query]
        original_embedded = EmbeddedQuery(
            query=original_query,
            vector=original_vector,
            query_type="original",
            model=model,
        )

        transformed_embedded: Optional[EmbeddedQuery] = None
        if has_transformed and transformed_query:
            transformed_vector = cached_vectors[transformed_query]
            transformed_embedded = EmbeddedQuery(
                query=transformed_query,
                vector=transformed_vector,
                query_type="transformed",
                model=model,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "Embedded query set in %.2fms (queries=%d, cached=%d, model='%s')",
            elapsed_ms,
            len(queries_to_embed),
            len(queries_to_embed) - len(uncached_queries),
            model,
        )

        return EmbeddedQuerySet(
            original=original_embedded,
            transformed=transformed_embedded,
        )


def get_query_embedding_service(
    provider: Optional[BaseEmbeddingProvider] = None,
    cache: Optional[RedisEmbeddingCache] = None,
    settings: Optional[Settings] = None,
) -> QueryEmbeddingService:
    """Get or create singleton QueryEmbeddingService instance."""
    global _query_embedding_service
    if provider is not None or cache is not None:
        return QueryEmbeddingService(provider=provider, cache=cache, settings=settings)

    if _query_embedding_service is None:
        _query_embedding_service = QueryEmbeddingService(settings=settings)
    return _query_embedding_service


def reset_query_embedding_service() -> None:
    """Reset cached singleton QueryEmbeddingService instance. Useful for tests."""
    global _query_embedding_service
    _query_embedding_service = None


async def embed_query_set(
    query_set: RetrievalQuerySet,
    service: Optional[QueryEmbeddingService] = None,
) -> EmbeddedQuerySet:
    """Convenience functional entrypoint to embed a RetrievalQuerySet."""
    active_service = service or get_query_embedding_service()
    return await active_service.embed_query_set(query_set)
