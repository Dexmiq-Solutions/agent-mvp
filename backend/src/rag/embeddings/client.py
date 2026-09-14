"""Centralized Voyage AI asynchronous client provider."""

from typing import Optional

import voyageai

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.embedding import EmbeddingConfigurationError

logger = get_logger(__name__)

_voyage_async_client: Optional[voyageai.AsyncClient] = None


def get_async_voyage_client(settings: Optional[Settings] = None) -> voyageai.AsyncClient:
    """Get or create the singleton asynchronous Voyage AI client.
    
    Reuses the client instance across the application lifecycle to avoid repeated
    instantiation overhead.
    
    Args:
        settings: Optional Settings override. Defaults to cached app settings.
        
    Returns:
        voyageai.AsyncClient: Initialized asynchronous Voyage AI client.
        
    Raises:
        EmbeddingConfigurationError: If VOYAGE_API_KEY is not configured.
    """
    global _voyage_async_client

    if _voyage_async_client is not None:
        return _voyage_async_client

    app_settings = settings or get_settings()
    api_key = app_settings.VOYAGE_API_KEY

    if not api_key:
        raise EmbeddingConfigurationError(
            "Missing 'VOYAGE_API_KEY'. Please set VOYAGE_API_KEY in your environment or .env file."
        )

    logger.info("Initializing asynchronous Voyage AI client")
    _voyage_async_client = voyageai.AsyncClient(api_key=api_key)
    return _voyage_async_client


def reset_async_voyage_client() -> None:
    """Reset the cached Voyage AI client instance. Useful for tests."""
    global _voyage_async_client
    _voyage_async_client = None
