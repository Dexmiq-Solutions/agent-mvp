"""Unit tests for PostgreSQL database foundation, engine, sessions, and configuration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings
from db.base import Base as DbBase
from db.session import (
    dispose_engine,
    get_async_engine,
    get_async_session_maker,
    get_db_session,
    reset_engine_sync,
    verify_database_connection,
)
from exceptions.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseSessionError,
)
from models.base import Base as ModelsBase


@pytest.fixture(autouse=True)
def cleanup_db_engine():
    """Reset database engine singleton before and after each test."""
    reset_engine_sync()
    yield
    reset_engine_sync()



@pytest.fixture
def mock_db_settings():
    """Settings instance with valid PostgreSQL configuration."""
    return Settings(
        DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/test_db",
        DB_POOL_SIZE=3,
        DB_MAX_OVERFLOW=5,
        DB_POOL_TIMEOUT=15,
        DB_ECHO=False,
    )


# ==============================================================================
# 1. Configuration & URL Normalization Tests
# ==============================================================================


def test_async_database_url_normalization():
    """Verify async_database_url correctly normalizes connection schemes."""
    # None when DATABASE_URL is not set
    settings_none = Settings(DATABASE_URL=None)
    assert settings_none.async_database_url is None

    # Normalizes postgresql:// to postgresql+asyncpg://
    settings_pg = Settings(DATABASE_URL="postgresql://user:pass@host:5432/db")
    assert settings_pg.async_database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    # Normalizes postgres:// to postgresql+asyncpg://
    settings_postgres = Settings(DATABASE_URL="postgres://user:pass@host:5432/db")
    assert settings_postgres.async_database_url == "postgresql+asyncpg://user:pass@host:5432/db"

    # Leaves postgresql+asyncpg:// intact
    settings_async = Settings(DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db")
    assert settings_async.async_database_url == "postgresql+asyncpg://user:pass@host:5432/db"


# ==============================================================================
# 2. Engine Lifecycle & Caching Tests
# ==============================================================================


def test_get_async_engine_missing_url():
    """Verify get_async_engine raises DatabaseConfigurationError when DATABASE_URL is unset."""
    empty_settings = Settings(DATABASE_URL=None)
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_async_engine(settings=empty_settings)
    assert "DATABASE_URL" in str(exc_info.value)


def test_get_async_engine_creation_and_caching(mock_db_settings):
    """Verify engine is created with correct settings and cached across calls."""
    engine1 = get_async_engine(settings=mock_db_settings)
    engine2 = get_async_engine(settings=mock_db_settings)

    assert isinstance(engine1, AsyncEngine)
    assert engine1 is engine2
    assert str(engine1.url) == "postgresql+asyncpg://test_user:***@localhost:5432/test_db"
    assert engine1.pool.size() == 3


@pytest.mark.anyio
async def test_dispose_engine(mock_db_settings):
    """Verify dispose_engine cleanly closes connection pool and resets singleton."""
    engine = get_async_engine(settings=mock_db_settings)
    assert engine is not None

    await dispose_engine()

    # Next call should create a fresh engine instance
    new_engine = get_async_engine(settings=mock_db_settings)
    assert new_engine is not None
    assert new_engine is not engine


# ==============================================================================
# 3. Session Factory & Dependency Injection Tests
# ==============================================================================


def test_get_async_session_maker(mock_db_settings):
    """Verify get_async_session_maker produces an async sessionmaker instance."""
    session_maker = get_async_session_maker(settings=mock_db_settings)
    assert isinstance(session_maker, async_sessionmaker)


@pytest.mark.anyio
async def test_get_db_session_success():
    """Verify get_db_session yields an AsyncSession, commits on success, and closes."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__.return_value = mock_session
    mock_maker.return_value.__aexit__.return_value = None

    with patch("db.session.get_async_session_maker", return_value=mock_maker):
        async for session in get_db_session():
            assert session is mock_session

        mock_session.commit.assert_awaited_once()
        mock_session.rollback.assert_not_awaited()
        mock_session.close.assert_awaited_once()


@pytest.mark.anyio
async def test_get_db_session_rollback_on_error():
    """Verify get_db_session rolls back transaction when an exception is thrown and propagates error."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__.return_value = mock_session
    mock_maker.return_value.__aexit__.return_value = None

    with patch("db.session.get_async_session_maker", return_value=mock_maker):
        gen = get_db_session()
        session = await gen.__anext__()
        assert session is mock_session

        with pytest.raises(ValueError) as exc_info:
            await gen.athrow(ValueError("Simulated query execution failure"))

        assert "Simulated query execution failure" in str(exc_info.value)
        mock_session.commit.assert_not_awaited()
        mock_session.rollback.assert_awaited_once()
        mock_session.close.assert_awaited_once()



# ==============================================================================
# 4. Connectivity Verification Tests
# ==============================================================================


@pytest.mark.anyio
async def test_verify_database_connection_success(mock_db_settings):
    """Verify verify_database_connection returns True on successful query execution."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    mock_engine.connect.return_value.__aexit__.return_value = None

    with patch("db.session.get_async_engine", return_value=mock_engine):
        result = await verify_database_connection(settings=mock_db_settings)
        assert result is True
        mock_conn.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_verify_database_connection_failure(mock_db_settings):
    """Verify verify_database_connection raises DatabaseConnectionError on connection failure."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.connect.side_effect = Exception("Connection refused on port 5432")

    with patch("db.session.get_async_engine", return_value=mock_engine):
        with pytest.raises(DatabaseConnectionError) as exc_info:
            await verify_database_connection(settings=mock_db_settings)
        assert "Connection refused" in str(exc_info.value)


@pytest.mark.anyio
async def test_verify_database_connection_missing_config():
    """Verify verify_database_connection raises DatabaseConfigurationError when unconfigured."""
    empty_settings = Settings(DATABASE_URL=None)
    with pytest.raises(DatabaseConfigurationError):
        await verify_database_connection(settings=empty_settings)


# ==============================================================================
# 5. Declarative Base Foundation Tests
# ==============================================================================


def test_declarative_base_foundation():
    """Verify Base is a valid DeclarativeBase class and exported consistently."""
    assert issubclass(DbBase, DeclarativeBase)
    assert issubclass(ModelsBase, DeclarativeBase)
    assert DbBase is ModelsBase
    assert hasattr(DbBase, "metadata")
    assert DbBase.metadata is not None


def test_all_models_registered_in_base_metadata():
    """Verify all application domain models are registered in Base.metadata."""
    import models  # noqa: F401
    table_names = set(DbBase.metadata.tables.keys())
    assert {"projects", "documents", "document_versions", "chunks", "conversations", "messages"}.issubset(table_names)


# ==============================================================================
# 6. Alembic Environment & Metadata Integration Tests
# ==============================================================================


def test_alembic_configuration_and_metadata():
    """Verify Alembic configuration file exists and points to Base.metadata."""
    import configparser
    from pathlib import Path

    # Verify alembic.ini
    ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    assert ini_path.exists(), "alembic.ini must exist in the backend root"

    ini_config = configparser.ConfigParser()
    ini_config.read(ini_path)
    assert ini_config.has_section("alembic")
    assert ini_config.get("alembic", "script_location") == "alembic"

    # Verify env.py exists and references Base.metadata
    env_path = Path(__file__).resolve().parents[1] / "alembic" / "env.py"
    assert env_path.exists(), "alembic/env.py must exist"
    env_content = env_path.read_text(encoding="utf-8")
    assert "target_metadata = Base.metadata" in env_content
    assert "from db.base import Base" in env_content

