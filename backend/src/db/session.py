"""Database engine, session management, and connectivity verification."""

from collections.abc import AsyncIterator
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseSessionError,
)

logger = get_logger(__name__)

_async_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def get_async_engine(settings: Optional[Settings] = None) -> AsyncEngine:
    """Get or create the singleton SQLAlchemy asynchronous database engine.
    
    Configured for asynchronous PostgreSQL access with connection pooling suitable
    for Supabase and cloud environments.
    
    Args:
        settings: Optional application settings override.
        
    Returns:
        AsyncEngine: Configured SQLAlchemy asynchronous engine.
        
    Raises:
        DatabaseConfigurationError: If DATABASE_URL is not configured.
    """
    global _async_engine

    if _async_engine is not None:
        return _async_engine

    app_settings = settings or get_settings()
    db_url = app_settings.async_database_url

    if not db_url:
        raise DatabaseConfigurationError(
            "Missing 'DATABASE_URL'. Please set DATABASE_URL in your environment or .env file."
        )

    logger.info("Initializing SQLAlchemy async engine with pool_pre_ping enabled")
    _async_engine = create_async_engine(
        db_url,
        pool_size=app_settings.DB_POOL_SIZE,
        max_overflow=app_settings.DB_MAX_OVERFLOW,
        pool_timeout=app_settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=app_settings.DB_ECHO,
    )
    return _async_engine


def get_async_session_maker(
    engine: Optional[AsyncEngine] = None,
    settings: Optional[Settings] = None,
) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.
    
    Args:
        engine: Optional explicit AsyncEngine instance.
        settings: Optional application settings override.
        
    Returns:
        async_sessionmaker: Factory for producing AsyncSession instances.
    """
    global _async_session_maker

    if _async_session_maker is not None and engine is None:
        return _async_session_maker

    active_engine = engine or get_async_engine(settings=settings)
    _async_session_maker = async_sessionmaker(
        bind=active_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _async_session_maker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an asynchronous database session.
    
    Ensures that transactions are properly committed on success, rolled back on exceptions,
    and closed after request handling.
    
    Yields:
        AsyncSession: Active asynchronous database session.
    """
    session_maker = get_async_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error("Database session rolled back due to error: %s", exc)
            raise
        finally:
            await session.close()



async def dispose_engine() -> None:
    """Dispose of the database engine and connection pool on application shutdown."""
    global _async_engine, _async_session_maker

    if _async_engine is not None:
        logger.info("Disposing SQLAlchemy database engine connections")
        await _async_engine.dispose()
        _async_engine = None
        _async_session_maker = None


def reset_engine_sync() -> None:
    """Synchronously reset the cached engine and session maker references (useful for tests)."""
    global _async_engine, _async_session_maker
    _async_engine = None
    _async_session_maker = None



async def verify_database_connection(settings: Optional[Settings] = None) -> bool:
    """Verify connectivity to the configured database by executing a lightweight query.
    
    Args:
        settings: Optional application settings override.
        
    Returns:
        bool: True if connection is healthy.
        
    Raises:
        DatabaseConfigurationError: If DATABASE_URL is missing.
        DatabaseConnectionError: If connection fails.
    """
    engine = get_async_engine(settings=settings)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connectivity check passed successfully")
        return True
    except DatabaseConfigurationError:
        raise
    except Exception as exc:
        logger.error("Database connectivity check failed: %s", exc)
        raise DatabaseConnectionError(
            f"Failed to connect to database: {exc}", original_error=exc
        ) from exc
