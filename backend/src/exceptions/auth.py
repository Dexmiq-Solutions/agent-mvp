"""Domain exceptions for authentication and authorization."""

from typing import Optional


class AuthError(Exception):
    """Base exception for all authentication and authorization errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidCredentialsError(AuthError):
    """Raised when authentication credentials (email/password) are invalid."""

    def __init__(self, message: str = "Invalid email or password.") -> None:
        super().__init__(message)


class UserAlreadyExistsError(AuthError):
    """Raised when signup is attempted with an existing email address."""

    def __init__(self, email: str, message: Optional[str] = None) -> None:
        msg = message or f"A user with email '{email}' already exists."
        super().__init__(msg)
        self.email = email


class UserNotFoundError(AuthError):
    """Raised when a user is not found."""

    def __init__(self, user_id: str, message: Optional[str] = None) -> None:
        msg = message or f"User with ID '{user_id}' not found."
        super().__init__(msg)
        self.user_id = user_id


class UserInactiveError(AuthError):
    """Raised when an inactive user attempts to authenticate."""

    def __init__(self, message: str = "User account is inactive.") -> None:
        super().__init__(message)


class InvalidTokenError(AuthError):
    """Raised when a JWT access token or refresh token is malformed, expired, or invalid."""

    def __init__(self, message: str = "Invalid or expired token.") -> None:
        super().__init__(message)


class TokenRevokedError(InvalidTokenError):
    """Raised when a revoked refresh token is presented."""

    def __init__(self, message: str = "Token has been revoked.") -> None:
        super().__init__(message)


class RateLimitExceededError(AuthError):
    """Raised when rate limit is exceeded for login attempts."""

    def __init__(
        self,
        message: str = "Too many login attempts. Please try again later.",
        retry_after: int = 60,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
