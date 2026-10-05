"""Unit tests for AuthService, password hashing, JWT security, and rate limiting."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.rate_limiter import LoginRateLimiter
from core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from exceptions.auth import (
    InvalidCredentialsError,
    InvalidTokenError,
    RateLimitExceededError,
    UserAlreadyExistsError,
    UserInactiveError,
)
from models.base import Base
from models.user import UserModel
from services.auth_service import AuthService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_session() -> AsyncIterator[AsyncSession]:
    """Create an in-memory SQLite database and yield an active AsyncSession."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


# ==============================================================================
# Security Utilities Tests
# ==============================================================================


def test_password_hashing_and_verification():
    """Verify Argon2id password hashing produces valid hashes and verifies correctly."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert hashed.startswith("$argon2id$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False


def test_jwt_access_token_creation_and_decoding():
    """Verify JWT access tokens encode and decode claims properly."""
    user_id = "user-12345"
    email = "test@example.com"
    token = create_access_token(user_id=user_id, email=email)

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert payload["type"] == "access"
    assert "exp" in payload
    assert "jti" in payload


def test_jwt_access_token_expired():
    """Verify expired JWT tokens raise InvalidTokenError."""
    user_id = "user-12345"
    email = "test@example.com"
    token = create_access_token(user_id=user_id, email=email, expires_delta=timedelta(seconds=-10))

    with pytest.raises(InvalidTokenError) as exc_info:
        decode_access_token(token)
    assert "expired" in str(exc_info.value).lower()


# ==============================================================================
# Rate Limiter Tests
# ==============================================================================


def test_login_rate_limiter_sliding_window():
    """Verify rate limiter blocks after configured max attempts and resets on success."""
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    key = "192.168.1.1:user@example.com"

    assert limiter.is_rate_limited(key) is False
    limiter.record_failure(key)
    limiter.record_failure(key)
    assert limiter.is_rate_limited(key) is False

    limiter.record_failure(key)
    assert limiter.is_rate_limited(key) is True

    with pytest.raises(RateLimitExceededError):
        limiter.check_rate_limit(key)

    # Clearing on success
    limiter.record_success(key)
    assert limiter.is_rate_limited(key) is False


# ==============================================================================
# AuthService Tests
# ==============================================================================


@pytest.mark.anyio
async def test_auth_service_signup():
    """Verify successful user signup with normalized lowercase email."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        user = await service.signup(email="  User@Example.COM  ", password="StrongPassword123")
        assert user.id is not None
        assert user.email == "user@example.com"
        assert user.is_active is True
        assert verify_password("StrongPassword123", user.hashed_password) is True


@pytest.mark.anyio
async def test_auth_service_signup_duplicate_email():
    """Verify signup rejects duplicate email addresses."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        await service.signup(email="user@example.com", password="Password123!")
        with pytest.raises(UserAlreadyExistsError):
            await service.signup(email="USER@example.com", password="AnotherPassword456!")


@pytest.mark.anyio
async def test_auth_service_login_success():
    """Verify successful login returns valid tokens and resets rate limiter."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        await service.signup(email="login@example.com", password="CorrectPassword123")

        user, access_token, refresh_token, expires_in = await service.login(
            email="login@example.com",
            password="CorrectPassword123",
            client_ip="127.0.0.1",
        )

        assert user.email == "login@example.com"
        assert access_token is not None
        assert refresh_token is not None
        assert expires_in > 0

        # Verify access token is valid
        payload = decode_access_token(access_token)
        assert payload["sub"] == user.id


@pytest.mark.anyio
async def test_auth_service_login_invalid_password():
    """Verify failed login raises InvalidCredentialsError and counts failures."""
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        await service.signup(email="victim@example.com", password="RealPassword123")

        with pytest.raises(InvalidCredentialsError):
            await service.login(
                email="victim@example.com",
                password="WrongPassword",
                client_ip="10.0.0.1",
            )


@pytest.mark.anyio
async def test_auth_service_login_inactive_user():
    """Verify inactive account is blocked from login."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        user = await service.signup(email="inactive@example.com", password="Password123")
        user.is_active = False
        await session.flush()

        with pytest.raises(UserInactiveError):
            await service.login(
                email="inactive@example.com",
                password="Password123",
                client_ip="127.0.0.1",
            )


@pytest.mark.anyio
async def test_auth_service_refresh_token_rotation_and_revocation():
    """Verify refresh token rotation issues fresh tokens and revokes prior token."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        await service.signup(email="rotate@example.com", password="Password123")
        _, _, refresh_token, _ = await service.login(
            email="rotate@example.com",
            password="Password123",
            client_ip="127.0.0.1",
        )

        # 1. Rotate token
        user, new_access, new_refresh, _ = await service.refresh_tokens(refresh_token)
        assert user.email == "rotate@example.com"
        assert new_refresh != refresh_token

        # 2. Presenting original (now revoked) token must fail
        with pytest.raises(InvalidTokenError):
            await service.refresh_tokens(refresh_token)

        # 3. Presenting newly issued token succeeds
        _, newer_access, newer_refresh, _ = await service.refresh_tokens(new_refresh)
        assert newer_refresh != new_refresh


@pytest.mark.anyio
async def test_auth_service_logout():
    """Verify logout revokes active refresh token."""
    limiter = LoginRateLimiter(max_attempts=5, window_seconds=60)
    async with create_test_session() as session:
        service = AuthService(session=session, rate_limiter=limiter)

        await service.signup(email="logout@example.com", password="Password123")
        _, _, refresh_token, _ = await service.login(
            email="logout@example.com",
            password="Password123",
            client_ip="127.0.0.1",
        )

        await service.logout(refresh_token)

        # Token is now revoked
        with pytest.raises(InvalidTokenError):
            await service.refresh_tokens(refresh_token)
