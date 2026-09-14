"""Centralized Qdrant asynchronous client provider."""

from typing import Optional

from qdrant_client import AsyncQdrantClient

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.vector import VectorStoreConfigurationError

logger = get_logger(__name__)

_qdrant_async_client: Optional[AsyncQdrantClient] = None


def get_async_qdrant_client(settings: Optional[Settings] = None) -> AsyncQdrantClient:
    """Get or create the singleton asynchronous Qdrant client.
    
    Reuses the client instance across the application lifecycle to avoid connection
    recreation overhead.
    
    Args:
        settings: Optional Settings override. Defaults to cached app settings.
        
    Returns:
        AsyncQdrantClient: Initialized asynchronous Qdrant client.
        
    Raises:
        VectorStoreConfigurationError: If QDRANT_URL is not configured.
    """
    global _qdrant_async_client

    if _qdrant_async_client is not None:
        return _qdrant_async_client

    app_settings = settings or get_settings()
    url = app_settings.QDRANT_URL

    if not url:
        raise VectorStoreConfigurationError(
            "Missing 'QDRANT_URL'. Please set QDRANT_URL in your environment or .env file."
        )

    logger.info("Initializing asynchronous Qdrant client (url: %s)", url)
    _qdrant_async_client = AsyncQdrantClient(
        url=url,
        api_key=app_settings.QDRANT_API_KEY,
        timeout=app_settings.QDRANT_TIMEOUT,
        check_compatibility=False,
    )
    return _qdrant_async_client



def reset_async_qdrant_client() -> None:
    """Reset the cached Qdrant client instance. Useful for tests."""
    global _qdrant_async_client
    _qdrant_async_client = None


async def close_async_qdrant_client() -> None:
    """Explicitly close the active Qdrant client connection if initialized."""
    global _qdrant_async_client
    if _qdrant_async_client is not None:
        try:
            await _qdrant_async_client.close()
        except Exception as exc:
            logger.warning("Error closing Qdrant client: %s", exc)
        finally:
            _qdrant_async_client = None
