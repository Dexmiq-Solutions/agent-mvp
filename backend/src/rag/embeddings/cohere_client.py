"""Centralized Cohere asynchronous client provider."""

from typing import Optional

import cohere

from core.config import Settings, get_settings
from exceptions.embedding import EmbeddingConfigurationError
from observability.logging import get_logger

logger = get_logger(__name__)

_cohere_async_client: Optional[cohere.AsyncClient] = None


def get_async_cohere_client(settings: Optional[Settings] = None) -> cohere.AsyncClient:
    """Get or create the singleton asynchronous Cohere client.

    Reuses the client instance across the application lifecycle to avoid repeated
    instantiation overhead.

    Args:
        settings: Optional Settings override. Defaults to cached app settings.

    Returns:
        cohere.AsyncClient: Initialized asynchronous Cohere client.

    Raises:
        EmbeddingConfigurationError: If COHERE_API_KEY is not configured.
    """
    global _cohere_async_client

    if _cohere_async_client is not None:
        return _cohere_async_client

    app_settings = settings or get_settings()
    api_key = getattr(app_settings, "COHERE_API_KEY", None)

    if not api_key:
        raise EmbeddingConfigurationError(
            "Missing 'COHERE_API_KEY'. Please set COHERE_API_KEY in your environment or .env file."
        )

    logger.info("Initializing asynchronous Cohere client")
    _cohere_async_client = cohere.AsyncClient(api_key=api_key)
    return _cohere_async_client


def reset_async_cohere_client() -> None:
    """Reset the cached Cohere client instance. Useful for tests."""
    global _cohere_async_client
    _cohere_async_client = None
