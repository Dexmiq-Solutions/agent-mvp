"""Application service for user authentication, registration, and token lifecycle management."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.rate_limiter import LoginRateLimiter, get_login_rate_limiter
from core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from exceptions.auth import (
    InvalidCredentialsError,
    InvalidTokenError,
    UserAlreadyExistsError,
    UserInactiveError,
    UserNotFoundError,
)
from models.refresh_token import RefreshTokenModel
from models.user import UserModel
from observability.logging import get_logger

logger = get_logger(__name__)


class AuthService:
    """Service orchestrating user credentials, tokens, and authentication boundaries."""

    def __init__(
        self,
        session: AsyncSession,
        rate_limiter: Optional[LoginRateLimiter] = None,
    ) -> None:
        """Initialize AuthService with database session and optional rate limiter.

        Args:
            session: Active asynchronous SQLAlchemy session.
            rate_limiter: Optional LoginRateLimiter instance.
        """
        self._session = session
        self._rate_limiter = rate_limiter or get_login_rate_limiter()

    async def signup(self, email: str, password: str) -> UserModel:
        """Register a new user account with validated email and hashed password.

        Args:
            email: User's email address.
            password: Raw password string.

        Returns:
            Newly created and persisted UserModel instance.

        Raises:
            UserAlreadyExistsError: If a user with the normalized email already exists.
        """
        normalized_email = email.strip().lower()

        # Check for existing user
        stmt = select(UserModel).where(UserModel.email == normalized_email)
        result = await self._session.execute(stmt)
        if result.scalar_one_or_none() is not None:
            logger.warning("Signup rejected: user with email '%s' already exists", normalized_email)
            raise UserAlreadyExistsError(normalized_email)

        # Hash password using Argon2id
        hashed = hash_password(password)

        user = UserModel(
            email=normalized_email,
            hashed_password=hashed,
            is_active=True,
        )
        self._session.add(user)
        await self._session.flush()

        logger.info("Successfully registered user '%s' (id: %s)", user.email, user.id)
        return user

    async def login(
        self,
        email: str,
        password: str,
        client_ip: str,
    ) -> tuple[UserModel, str, str, int]:
        """Authenticate user credentials and issue access + refresh tokens.

        Args:
            email: User's email address.
            password: Raw password string.
            client_ip: Originating client IP address for rate limit tracking.

        Returns:
            tuple: (UserModel, access_token, refresh_token, expires_in_seconds).

        Raises:
            RateLimitExceededError: If failed attempts threshold is exceeded.
            InvalidCredentialsError: If email or password is incorrect.
            UserInactiveError: If user account is marked inactive.
        """
        normalized_email = email.strip().lower()
        rate_key = f"{client_ip}:{normalized_email}"

        # 1. Enforce rate limiting before checking credentials
        self._rate_limiter.check_rate_limit(rate_key)

        # 2. Query user by normalized email
        stmt = select(UserModel).where(UserModel.email == normalized_email)
        result = await self._session.execute(stmt)
        user = result.scalar_one_or_none()

        # 3. Verify password (or constant-time dummy verification if user not found)
        if user is None or not verify_password(password, user.hashed_password):
            self._rate_limiter.record_failure(rate_key)
            logger.warning("Failed login attempt for email '%s' from IP '%s'", normalized_email, client_ip)
            raise InvalidCredentialsError("Invalid email or password.")

        # 4. Check active account status
        if not user.is_active:
            self._rate_limiter.record_failure(rate_key)
            logger.warning("Inactive account login attempt for email '%s'", normalized_email)
            raise UserInactiveError("User account is inactive.")

        # 5. Clear rate limit failure counter upon successful authentication
        self._rate_limiter.record_success(rate_key)

        # 6. Issue access and refresh tokens
        settings = get_settings()
        access_token = create_access_token(user.id, user.email)
        refresh_token = generate_refresh_token()
        token_hash = hash_token(refresh_token)

        now = datetime.now(timezone.utc)
        refresh_record = RefreshTokenModel(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            created_at=now,
        )
        self._session.add(refresh_record)
        await self._session.flush()

        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        logger.info("User '%s' (id: %s) logged in successfully", user.email, user.id)
        return user, access_token, refresh_token, expires_in

    async def refresh_tokens(self, refresh_token: str) -> tuple[UserModel, str, str, int]:
        """Validate and rotate an existing refresh token, issuing a new token pair.

        Args:
            refresh_token: Raw opaque refresh token presented by client.

        Returns:
            tuple: (UserModel, new_access_token, new_refresh_token, expires_in_seconds).

        Raises:
            InvalidTokenError: If refresh token is expired, invalid, revoked, or user inactive.
        """
        token_hash = hash_token(refresh_token)

        # 1. Lookup stored token digest
        stmt = select(RefreshTokenModel).where(RefreshTokenModel.token_hash == token_hash)
        result = await self._session.execute(stmt)
        token_record = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if token_record is None or token_record.revoked_at is not None:
            logger.warning("Refresh token rejected: token not found or revoked")
            raise InvalidTokenError("Invalid or expired refresh token.")

        expires_at = token_record.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at <= now:
            logger.warning("Refresh token rejected: token expired")
            raise InvalidTokenError("Invalid or expired refresh token.")


        # 2. Lookup owning user
        stmt_user = select(UserModel).where(UserModel.id == token_record.user_id)
        res_user = await self._session.execute(stmt_user)
        user = res_user.scalar_one_or_none()

        if user is None or not user.is_active:
            logger.warning("Refresh token rejected: user '%s' not found or inactive", token_record.user_id)
            raise InvalidTokenError("User associated with token is not active.")

        # 3. Single-use rotation: Revoke current token
        token_record.revoked_at = now

        # 4. Issue new token pair
        settings = get_settings()
        new_access_token = create_access_token(user.id, user.email)
        new_refresh_token = generate_refresh_token()
        new_token_hash = hash_token(new_refresh_token)

        new_refresh_record = RefreshTokenModel(
            user_id=user.id,
            token_hash=new_token_hash,
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            created_at=now,
        )
        self._session.add(new_refresh_record)
        await self._session.flush()

        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        logger.info("Rotated refresh token for user id: %s", user.id)
        return user, new_access_token, new_refresh_token, expires_in

    async def logout(self, refresh_token: str) -> None:
        """Revoke a refresh token on user logout.

        Args:
            refresh_token: Raw opaque refresh token string.
        """
        token_hash = hash_token(refresh_token)
        stmt = select(RefreshTokenModel).where(RefreshTokenModel.token_hash == token_hash)
        result = await self._session.execute(stmt)
        token_record = result.scalar_one_or_none()

        if token_record is not None and token_record.revoked_at is None:
            token_record.revoked_at = datetime.now(timezone.utc)
            await self._session.flush()
            logger.info("Revoked refresh token id: %s for user id: %s", token_record.id, token_record.user_id)

    async def get_user_by_id(self, user_id: str) -> UserModel:
        """Retrieve user entity by primary key.

        Args:
            user_id: User identifier.

        Returns:
            Active UserModel instance.

        Raises:
            UserNotFoundError: If user not found.
            UserInactiveError: If user is inactive.
        """
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            raise UserNotFoundError(user_id)
        if not user.is_active:
            raise UserInactiveError("User account is inactive.")

        return user
