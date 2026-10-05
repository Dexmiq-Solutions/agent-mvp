"""Cryptographic security utilities for password hashing and token management."""

from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Any, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
import jwt

from core.config import get_settings
from exceptions.auth import InvalidTokenError

# Argon2id password hasher with RFC 9106 recommended parameters
_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Hash a password using Argon2id.

    Args:
        password: Raw plaintext password string.

    Returns:
        Argon2id encoded hash string.
    """
    return _password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against an Argon2id hash.

    Args:
        plain_password: Raw plaintext password to test.
        hashed_password: Stored Argon2id hash string.

    Returns:
        bool: True if password matches hash, False otherwise.
    """
    try:
        return _password_hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        user_id: Unique user identifier for token subject ('sub').
        email: User email address claim.
        expires_delta: Optional custom duration; defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        str: Encoded JWT string.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    duration = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = now + duration

    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": secrets.token_hex(16),
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token.

    Args:
        token: Raw JWT access token string.

    Returns:
        dict containing verified claims.

    Raises:
        InvalidTokenError: If token is expired, malformed, or has invalid claims.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub", "type"]},
        )
        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid token type. Access token required.")
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(f"Invalid access token: {exc}") from exc


def generate_refresh_token() -> str:
    """Generate a cryptographically secure random refresh token string.

    Returns:
        str: High-entropy URL-safe token.
    """
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """Compute deterministic SHA-256 digest of a token string for safe storage.

    Args:
        token: Plain token string.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
