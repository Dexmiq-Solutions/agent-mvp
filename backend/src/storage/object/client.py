"""Centralized asynchronous Supabase client provider."""

import asyncio
from typing import Optional

from supabase import AsyncClient, create_async_client

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.storage import StorageConfigurationError, StorageConnectionError

logger = get_logger(__name__)

_supabase_async_client: Optional[AsyncClient] = None
_client_lock = asyncio.Lock()


async def get_async_supabase_client(settings: Optional[Settings] = None) -> AsyncClient:
    """Get or create the singleton asynchronous Supabase client.
    
    Reuses the existing initialized client across the application lifecycle to avoid
    redundant connections and overhead.
    
    Args:
        settings: Optional Settings override. Defaults to cached app settings.
        
    Returns:
        AsyncClient: Official asynchronous Supabase client instance.
        
    Raises:
        StorageConfigurationError: If required Supabase credentials are missing.
        StorageConnectionError: If client initialization fails.
    """
    global _supabase_async_client

    if _supabase_async_client is not None:
        return _supabase_async_client

    async with _client_lock:
        if _supabase_async_client is not None:
            return _supabase_async_client

        app_settings = settings or get_settings()

        supabase_url = app_settings.SUPABASE_URL
        supabase_key = app_settings.supabase_storage_key

        if not supabase_url:
            raise StorageConfigurationError(
                "Missing 'SUPABASE_URL'. Please set SUPABASE_URL in your environment or .env file."
            )

        if not supabase_key:
            raise StorageConfigurationError(
                "Missing Supabase credentials. Please set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY."
            )

        try:
            logger.info("Initializing asynchronous Supabase client for URL: %s", supabase_url)
            _supabase_async_client = await create_async_client(supabase_url, supabase_key)
            return _supabase_async_client
        except Exception as exc:
            logger.error("Failed to initialize Supabase client: %s", exc)
            raise StorageConnectionError(
                f"Failed to initialize Supabase client: {exc}", original_error=exc
            ) from exc


def reset_async_supabase_client() -> None:
    """Reset the cached Supabase client instance. Useful for testing and lifecycle hooks."""
    global _supabase_async_client
    _supabase_async_client = None

