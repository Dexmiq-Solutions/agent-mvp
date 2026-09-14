"""Configuration classes for LLM generation execution."""

from dataclasses import dataclass, field
import os
from typing import Optional, Sequence

from core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMConfig:
    """Configuration settings for LLM generation and provider execution.

    In accordance with security guidelines:
    - Sensitive values (API keys, custom endpoint URLs) are NEVER hard-coded.
    - Default values for api_key and base_url are None.
    - The LLMConfig fetches sensitive parameters strictly from environment variables
      or the application Settings single source of truth.
    - Model selection and operational generation parameters can be configured directly
      with sensible defaults and environment overrides.
    """

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 0.5
    retry_backoff: float = 2.0
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = field(default_factory=tuple)
    streaming_enabled: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "LLMConfig":
        """Construct LLMConfig from application Settings (Single Source of Truth).

        Fetches sensitive credentials (API key, base URL) and generation settings
        directly from the environment/Settings.
        """
        app_settings = settings or get_settings()

        # Sensitive parameters fetched strictly from environment settings
        api_key = getattr(app_settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        base_url = getattr(app_settings, "LLM_BASE_URL", None) or os.getenv("LLM_BASE_URL")
        if base_url:
            base_url = base_url.strip().rstrip("/")
            if not base_url:
                base_url = None

        # Model and generation parameters
        model = getattr(app_settings, "LLM_MODEL", "gpt-4o") or "gpt-4o"
        timeout = float(getattr(app_settings, "LLM_TIMEOUT", 30.0) or 30.0)
        max_retries = int(getattr(app_settings, "LLM_MAX_RETRIES", 2))
        retry_delay = float(getattr(app_settings, "LLM_RETRY_DELAY", 0.5) or 0.5)
        retry_backoff = float(getattr(app_settings, "LLM_RETRY_BACKOFF", 2.0) or 2.0)
        temperature = float(getattr(app_settings, "LLM_TEMPERATURE", 0.0) or 0.0)
        max_tokens = getattr(app_settings, "LLM_MAX_TOKENS", None)
        top_p = getattr(app_settings, "LLM_TOP_P", None)
        stop_seqs = getattr(app_settings, "LLM_STOP_SEQUENCES", None) or ()
        streaming_enabled = bool(getattr(app_settings, "LLM_STREAMING_ENABLED", False))

        return cls(
            model=model,
            api_key=api_key if api_key else None,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            stop_sequences=tuple(stop_seqs),
            streaming_enabled=streaming_enabled,
        )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Convenience factory constructing LLMConfig from environment settings."""
        return cls.from_settings()

    def __repr__(self) -> str:
        """Safe representation masking sensitive credentials like API keys."""
        masked_key = "***" if self.api_key else None
        return (
            f"LLMConfig(model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key!r}, "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries}, "
            f"temperature={self.temperature})"
        )
