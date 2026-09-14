"""Configuration settings for Groundedness and Safety evaluation."""

from dataclasses import dataclass
from typing import Optional

from core.config import Settings, get_settings


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration governing Groundedness and Safety quality gate evaluation.

    Follows project guidelines:
    - Reuses the application Settings as the Single Source of Truth.
    - Sensible defaults for bounded timeouts, retries, and regeneration limits.
    - Safe masked representation preventing leaking sensitive configuration.
    """

    enabled: bool = True
    model: str | None = None
    timeout: float = 15.0
    max_retries: int = 1
    temperature: float = 0.0
    max_regeneration_attempts: int = 2
    strict_mode: bool = True

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "EvaluationConfig":
        """Construct EvaluationConfig from application Settings (Single Source of Truth)."""
        app_settings = settings or get_settings()

        enabled = bool(getattr(app_settings, "EVALUATION_ENABLED", True))
        model = getattr(app_settings, "EVALUATION_MODEL", None) or getattr(app_settings, "LLM_MODEL", "gpt-4o")
        timeout = float(getattr(app_settings, "EVALUATION_TIMEOUT", 15.0) or 15.0)
        max_retries = int(getattr(app_settings, "EVALUATION_MAX_RETRIES", 1))
        temperature = float(getattr(app_settings, "EVALUATION_TEMPERATURE", 0.0) or 0.0)
        max_regeneration_attempts = int(getattr(app_settings, "EVALUATION_MAX_REGENERATION_ATTEMPTS", 2))

        return cls(
            enabled=enabled,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_regeneration_attempts=max_regeneration_attempts,
            strict_mode=True,
        )

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Convenience factory constructing EvaluationConfig from environment settings."""
        return cls.from_settings()

    def __repr__(self) -> str:
        """Safe representation of evaluation configuration parameters."""
        return (
            f"EvaluationConfig("
            f"enabled={self.enabled}, "
            f"model={self.model!r}, "
            f"timeout={self.timeout}, "
            f"temperature={self.temperature}, "
            f"max_retries={self.max_retries}, "
            f"max_regeneration_attempts={self.max_regeneration_attempts})"
        )
