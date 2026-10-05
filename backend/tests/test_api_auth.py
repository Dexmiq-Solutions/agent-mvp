"""API integration tests for authentication endpoints (/auth/*)."""

from collections.abc import AsyncIterator

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from core.rate_limiter import get_login_rate_limiter
from db.base import Base
from db.session import get_db_session


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def auth_client():
    """Configure in-memory database and AsyncClient for auth testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session
    limiter = get_login_rate_limiter()
    limiter.reset()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    limiter.reset()
    await engine.dispose()


@pytest.mark.anyio
async def test_api_signup_success(auth_client):
    """Verify POST /auth/signup registers a new user."""
    resp = await auth_client.post(
        "/auth/signup",
        json={"email": "newuser@example.com", "password": "Password123!"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.anyio
async def test_api_signup_duplicate_email(auth_client):
    """Verify POST /auth/signup returns 409 for duplicate email."""
    await auth_client.post(
        "/auth/signup",
        json={"email": "duplicate@example.com", "password": "Password123!"},
    )
    resp = await auth_client.post(
        "/auth/signup",
        json={"email": "duplicate@example.com", "password": "DifferentPassword123!"},
    )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_api_signup_short_password(auth_client):
    """Verify POST /auth/signup returns 422 for password shorter than 8 chars."""
    resp = await auth_client.post(
        "/auth/signup",
        json={"email": "short@example.com", "password": "short"},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_api_login_success_and_me(auth_client):
    """Verify POST /auth/login returns token and GET /auth/me returns profile."""
    # 1. Signup
    await auth_client.post(
        "/auth/signup",
        json={"email": "loginuser@example.com", "password": "ValidPassword123"},
    )

    # 2. Login
    login_resp = await auth_client.post(
        "/auth/login",
        json={"email": "loginuser@example.com", "password": "ValidPassword123"},
    )
    assert login_resp.status_code == 200
    tokens = login_resp.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"].lower() == "bearer"

    # 3. Access /auth/me with access token
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    me_resp = await auth_client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == "loginuser@example.com"


@pytest.mark.anyio
async def test_api_login_invalid_credentials(auth_client):
    """Verify POST /auth/login returns 401 for incorrect credentials."""
    await auth_client.post(
        "/auth/signup",
        json={"email": "user@example.com", "password": "ValidPassword123"},
    )

    resp = await auth_client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "WrongPassword"},
    )
    assert resp.status_code == 401
    assert "invalid email or password" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_api_login_rate_limiting(auth_client):
    """Verify rate limiter blocks login attempts after 5 failures with HTTP 429."""
    await auth_client.post(
        "/auth/signup",
        json={"email": "brute@example.com", "password": "RealPassword123"},
    )

    # Send 5 failed attempts
    for _ in range(5):
        resp = await auth_client.post(
            "/auth/login",
            json={"email": "brute@example.com", "password": "BadPassword"},
        )
        assert resp.status_code == 401

    # 6th attempt must be rate limited with 429
    limited_resp = await auth_client.post(
        "/auth/login",
        json={"email": "brute@example.com", "password": "BadPassword"},
    )
    assert limited_resp.status_code == 429
    assert "too many" in limited_resp.json()["detail"].lower()
    assert "retry-after" in limited_resp.headers


@pytest.mark.anyio
async def test_api_refresh_tokens_and_logout(auth_client):
    """Verify token refresh rotation and logout revocation."""
    # 1. Signup & Login
    await auth_client.post(
        "/auth/signup",
        json={"email": "refresh@example.com", "password": "Password123!"},
    )
    login_resp = await auth_client.post(
        "/auth/login",
        json={"email": "refresh@example.com", "password": "Password123!"},
    )
    tokens = login_resp.json()
    refresh_token = tokens["refresh_token"]

    # 2. Refresh token
    refresh_resp = await auth_client.post(
        "/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert new_tokens["refresh_token"] != refresh_token

    # 3. Old refresh token cannot be used again
    old_use_resp = await auth_client.post(
        "/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert old_use_resp.status_code == 401

    # 4. Logout with new refresh token
    logout_resp = await auth_client.post(
        "/auth/logout",
        json={"refresh_token": new_tokens["refresh_token"]},
    )
    assert logout_resp.status_code == 204

    # 5. Revoked token cannot refresh
    post_logout_resp = await auth_client.post(
        "/auth/refresh",
        json={"refresh_token": new_tokens["refresh_token"]},
    )
    assert post_logout_resp.status_code == 401
