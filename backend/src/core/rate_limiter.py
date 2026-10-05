"""Basic login rate limiter providing brute-force protection with sliding window tracking."""

from collections import defaultdict
from datetime import datetime, timezone
import threading
from typing import Optional

from core.config import Settings, get_settings
from exceptions.auth import RateLimitExceededError
from observability.logging import get_logger

logger = get_logger(__name__)


class LoginRateLimiter:
    """In-memory sliding window rate limiter for login attempts."""

    def __init__(
        self,
        max_attempts: Optional[int] = None,
        window_seconds: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            max_attempts: Maximum allowed failed attempts within the window.
            window_seconds: Sliding window duration in seconds.
            settings: Optional Settings override.
        """
        app_settings = settings or get_settings()
        self._max_attempts = (
            max_attempts
            if max_attempts is not None
            else app_settings.LOGIN_RATE_LIMIT_MAX_ATTEMPTS
        )
        self._window_seconds = (
            window_seconds
            if window_seconds is not None
            else app_settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS
        )
        self._lock = threading.Lock()
        self._attempts: dict[str, list[float]] = defaultdict(list)

    @property
    def window_seconds(self) -> int:
        """Return sliding window duration in seconds."""
        return self._window_seconds

    @property
    def max_attempts(self) -> int:
        """Return maximum attempts threshold."""
        return self._max_attempts

    def _purge_expired(self, key: str, now: float) -> None:
        """Remove timestamps outside the active sliding window."""
        cutoff = now - self._window_seconds
        self._attempts[key] = [ts for ts in self._attempts[key] if ts > cutoff]
        if not self._attempts[key]:
            self._attempts.pop(key, None)

    def is_rate_limited(self, key: str) -> bool:
        """Check whether the given key has exceeded the allowed attempt threshold.

        Args:
            key: Rate limit identity key (e.g., client IP or IP:email).

        Returns:
            bool: True if rate limit is exceeded.
        """
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            self._purge_expired(key, now)
            return len(self._attempts.get(key, [])) >= self._max_attempts

    def check_rate_limit(self, key: str) -> None:
        """Check rate limit and raise RateLimitExceededError if threshold is reached.

        Args:
            key: Rate limit identity key.

        Raises:
            RateLimitExceededError: If key is rate limited.
        """
        if self.is_rate_limited(key):
            logger.warning("Login rate limit exceeded for key '%s'", key)
            raise RateLimitExceededError(
                message=f"Too many failed login attempts. Please try again in {self._window_seconds} seconds.",
                retry_after=self._window_seconds,
            )

    def record_failure(self, key: str) -> None:
        """Record a failed login attempt for the key.

        Args:
            key: Rate limit identity key.
        """
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            self._purge_expired(key, now)
            self._attempts[key].append(now)

    def record_success(self, key: str) -> None:
        """Clear recorded failures upon a successful login.

        Args:
            key: Rate limit identity key.
        """
        with self._lock:
            self._attempts.pop(key, None)

    def reset(self) -> None:
        """Reset all rate limit tracking (useful for test isolation)."""
        with self._lock:
            self._attempts.clear()


_rate_limiter_instance: Optional[LoginRateLimiter] = None


def get_login_rate_limiter() -> LoginRateLimiter:
    """Get the singleton LoginRateLimiter instance."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = LoginRateLimiter()
    return _rate_limiter_instance
