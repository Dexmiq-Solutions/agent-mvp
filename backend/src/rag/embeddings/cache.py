"""Shared Redis embedding-result cache infrastructure with graceful degradation."""

import hashlib
import json
from typing import Any, Optional

from core.config import Settings, get_settings
from observability.logging import get_logger

logger = get_logger(__name__)

_redis_embedding_cache: Optional["RedisEmbeddingCache"] = None


def generate_embedding_cache_key(
    text: str,
    model: str,
    provider: str = "voyage",
    input_type: str = "query",
    dimension: Optional[int] = None,
    version: Optional[str] = None,
) -> str:
    """Generate a canonical, deterministic cache key for an embedding text representation.

    Incorporates text content hash, provider, model name, input type (query vs document),
    dimensionality, and version to prevent cache poisoning across differing configurations.

    Args:
        text: Normalized text string.
        model: Active embedding model name (e.g., 'voyage-4').
        provider: Provider identifier (e.g., 'voyage').
        input_type: Input type hint ('query' or 'document').
        dimension: Optional expected vector dimension.
        version: Optional configuration version tag.

    Returns:
        str: Deterministic cache key string.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    dim_str = str(dimension) if dimension is not None else "default"
    ver_str = version or "v1"
    return f"embedding:{provider}:{model}:{input_type}:{dim_str}:{ver_str}:{text_hash}"


class RedisEmbeddingCache:
    """Shared Redis embedding-result cache supporting non-blocking get/set.

    Redis is treated strictly as a performance cache, never as the authoritative source
    of truth. If Redis is unconfigured, unreachable, or throws an error, operations
    gracefully degrade (returning None on reads, silently logging on writes) so the
    embedding pipeline continues uninterrupted.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        client: Any = None,
        default_ttl: int = 86400,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize RedisEmbeddingCache.

        Args:
            redis_url: Optional Redis connection string (e.g., 'redis://localhost:6379/0').
            client: Pre-configured Async Redis client (useful for dependency injection and tests).
            default_ttl: Default cache entry time-to-live in seconds (default: 86400 / 24h).
            settings: Optional application settings.
        """
        self._settings = settings or get_settings()
        self._redis_url = redis_url if redis_url is not None else getattr(self._settings, "REDIS_URL", None)
        self._client = client
        self._default_ttl = default_ttl
        self._initialized = False

    def _get_client(self) -> Any:
        """Resolve active async Redis client or return None if unconfigured/unavailable."""
        if self._client is not None:
            return self._client

        if not self._redis_url:
            return None

        if not self._initialized:
            try:
                import redis.asyncio as aioredis  # type: ignore

                self._client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                self._initialized = True
                logger.info("Connected to Redis embedding cache at %s", self._redis_url)
            except Exception as exc:
                logger.warning(
                    "Redis embedding cache client initialization failed (%s). Degrading gracefully.",
                    exc,
                )
                self._client = None
                self._initialized = True

        return self._client

    async def get(self, key: str) -> Optional[list[float]]:
        """Lookup cached embedding vector by key.

        Returns:
            list[float] if cache hit, None on cache miss or cache failure.
        """
        try:
            client = self._get_client()
            if client is None:
                return None

            raw_val = await client.get(key)
            if raw_val is None:
                return None

            data = json.loads(raw_val)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            return None
        except Exception as exc:
            logger.warning("Redis embedding cache lookup failed for key '%s': %s", key, exc)
            return None

    async def set(
        self,
        key: str,
        vector: list[float],
        ttl: Optional[int] = None,
    ) -> None:
        """Store embedding vector in cache.

        Silently suppresses errors if Redis is unavailable or write fails.
        """
        try:
            client = self._get_client()
            if client is None:
                return

            val = json.dumps(vector)
            effective_ttl = ttl if ttl is not None else self._default_ttl
            await client.set(key, val, ex=effective_ttl)
        except Exception as exc:
            logger.warning("Redis embedding cache write failed for key '%s': %s", key, exc)

    async def get_many(self, keys: list[str]) -> list[Optional[list[float]]]:
        """Lookup multiple keys preserving order. Returns None for misses or on error."""
        if not keys:
            return []

        try:
            client = self._get_client()
            if client is None:
                return [None] * len(keys)

            # Try mget if supported by client
            if hasattr(client, "mget"):
                raw_vals = await client.mget(keys)
                results: list[Optional[list[float]]] = []
                for val in raw_vals:
                    if val is None:
                        results.append(None)
                    else:
                        try:
                            data = json.loads(val)
                            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
                                results.append([float(x) for x in data])
                            else:
                                results.append(None)
                        except Exception:
                            results.append(None)
                return results

            # Fallback to sequential get
            return [await self.get(k) for k in keys]
        except Exception as exc:
            logger.warning("Redis embedding cache mget failed: %s", exc)
            return [None] * len(keys)

    async def set_many(
        self,
        items: list[tuple[str, list[float]]],
        ttl: Optional[int] = None,
    ) -> None:
        """Store multiple embedding vectors in cache."""
        for key, vector in items:
            await self.set(key, vector, ttl=ttl)


def get_redis_embedding_cache(settings: Optional[Settings] = None) -> RedisEmbeddingCache:
    """Get or create singleton RedisEmbeddingCache instance."""
    global _redis_embedding_cache
    if _redis_embedding_cache is None:
        _redis_embedding_cache = RedisEmbeddingCache(settings=settings)
    return _redis_embedding_cache


def reset_redis_embedding_cache() -> None:
    """Reset cached singleton RedisEmbeddingCache instance. Useful for tests."""
    global _redis_embedding_cache
    _redis_embedding_cache = None
