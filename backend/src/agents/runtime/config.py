"""Configuration settings for the Agent runtime and model providers."""

from dataclasses import dataclass
import os
from typing import Optional

from core.config import Settings, get_settings
from exceptions.agent import AgentConfigurationError


@dataclass(frozen=True)
class AgentConfig:
    """Configuration settings for DeepAgents agent runtime and model provider execution.

    In accordance with security guidelines:
    - Sensitive credentials (API keys, authorization headers) are NEVER hard-coded.
    - Default values for api_key are None.
    - Values are loaded strictly from application Settings or environment variables.
    - String representations (__repr__) mask sensitive credentials.
    """

    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 2
    system_prompt: str = "You are a helpful AI assistant."

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "AgentConfig":
        """Construct AgentConfig from application Settings (Single Source of Truth).

        Evaluates OpenRouter settings first, with clean fallback to generic LLM settings.
        """
        app_settings = settings or get_settings()

        # Resolve credentials with fallback hierarchy
        api_key = (
            getattr(app_settings, "OPENROUTER_API_KEY", None)
            or os.getenv("OPENROUTER_API_KEY")
            or getattr(app_settings, "OPENAI_API_KEY", None)
            or os.getenv("OPENAI_API_KEY")
        )
        if api_key:
            api_key = api_key.strip()
            if not api_key:
                api_key = None

        # Resolve base URL with fallback hierarchy
        base_url = (
            getattr(app_settings, "OPENROUTER_BASE_URL", None)
            or os.getenv("OPENROUTER_BASE_URL")
            or getattr(app_settings, "LLM_BASE_URL", None)
            or os.getenv("LLM_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        if base_url:
            base_url = base_url.strip().rstrip("/")
            if not base_url:
                base_url = "https://openrouter.ai/api/v1"

        # Resolve model name with fallback hierarchy
        model = (
            getattr(app_settings, "AGENT_MODEL", None)
            or os.getenv("AGENT_MODEL")
            or getattr(app_settings, "LLM_MODEL", None)
            or os.getenv("LLM_MODEL")
            or "gpt-4o"
        )
        if model:
            model = model.strip()

        # Operational parameters
        temperature = float(getattr(app_settings, "AGENT_TEMPERATURE", 0.0) or 0.0)
        max_tokens = getattr(app_settings, "AGENT_MAX_TOKENS", None)
        timeout = float(getattr(app_settings, "AGENT_TIMEOUT", 60.0) or 60.0)
        max_retries = int(getattr(app_settings, "AGENT_MAX_RETRIES", 2))

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Convenience factory constructing AgentConfig directly from environment settings."""
        return cls.from_settings()

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            AgentConfigurationError: If required parameters or credentials are invalid.
        """
        if not self.model or not self.model.strip():
            raise AgentConfigurationError("Agent model name must not be empty.")
        if not self.api_key or not self.api_key.strip():
            raise AgentConfigurationError(
                "Missing API key for agent model provider. Configure OPENROUTER_API_KEY or OPENAI_API_KEY."
            )

    def __repr__(self) -> str:
        """Safe representation masking sensitive credentials like API keys."""
        masked_key = "***" if self.api_key else None
        return (
            f"AgentConfig(model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key!r}, "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries}, "
            f"temperature={self.temperature})"
        )
