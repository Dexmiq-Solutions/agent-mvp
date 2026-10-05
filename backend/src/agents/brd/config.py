"""Configuration and model provider helpers for the BRD Agent subsystem."""

from dataclasses import dataclass
import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import Settings, get_settings
from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    """Configuration settings for agent execution and model providers.

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

        model = (
            getattr(app_settings, "AGENT_MODEL", None)
            or os.getenv("AGENT_MODEL")
            or getattr(app_settings, "LLM_MODEL", None)
            or os.getenv("LLM_MODEL")
            or "gpt-4o"
        )
        if model:
            model = model.strip()

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
        """Validate configuration parameters."""
        if not self.model or not self.model.strip():
            raise ValueError("Agent model name must not be empty.")

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


def create_agent_model(config: Optional[AgentConfig] = None) -> BaseChatModel:
    """Create and configure a LangChain-compatible chat model for the agent.

    Connects to OpenRouter or any OpenAI-compatible provider using the specified configuration.
    """
    cfg = config or AgentConfig.from_settings()

    logger.debug(
        "Initializing Agent model provider (model: %s, base_url: %s)",
        cfg.model,
        cfg.base_url,
    )

    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key or "sk-dummy-key",
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        default_headers={
            "HTTP-Referer": "https://agent-mvp.local",
            "X-Title": "Agent MVP",
        },
    )
