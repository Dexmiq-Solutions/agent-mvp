"""Configuration and model provider helpers for the Recommendation Agent subsystem."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import Settings, get_settings
from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecommendationConfig:
    """Configuration settings for Recommendation Agent execution and model providers.

    Follows security guidelines:
    - Sensitive credentials are never hard-coded.
    - Default values for api_key are None.
    - Values are loaded strictly from application Settings or environment variables.
    - String representations (__repr__) mask sensitive credentials.
    """

    model: str = "gpt-4o"
    router_model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    router_temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> RecommendationConfig:
        """Construct RecommendationConfig from application Settings (Single Source of Truth)."""
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

        main_model = (
            getattr(app_settings, "AGENT_MODEL", None)
            or os.getenv("AGENT_MODEL")
            or getattr(app_settings, "LLM_MODEL", None)
            or os.getenv("LLM_MODEL")
            or "gpt-4o"
        )
        if main_model:
            main_model = main_model.strip()

        router_model = (
            os.getenv("RECOMMENDATION_ROUTER_MODEL")
            or os.getenv("ROUTER_MODEL")
            or "gpt-4o-mini"
        )
        if router_model:
            router_model = router_model.strip()

        temperature = float(getattr(app_settings, "AGENT_TEMPERATURE", 0.0) or 0.0)
        max_tokens = getattr(app_settings, "AGENT_MAX_TOKENS", None)
        timeout = float(getattr(app_settings, "AGENT_TIMEOUT", 60.0) or 60.0)
        max_retries = int(getattr(app_settings, "AGENT_MAX_RETRIES", 2))

        return cls(
            model=main_model,
            router_model=router_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            router_temperature=0.0,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            timeout=timeout,
            max_retries=max_retries,
        )

    def __repr__(self) -> str:
        """Safe representation masking sensitive credentials."""
        masked_key = "***" if self.api_key else None
        return (
            f"RecommendationConfig(model={self.model!r}, "
            f"router_model={self.router_model!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key!r}, "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )


def create_router_model(config: Optional[RecommendationConfig] = None) -> BaseChatModel:
    """Create and configure the lightweight chat model for the Intent Router Agent (D27)."""
    cfg = config or RecommendationConfig.from_settings()

    logger.debug(
        "Initializing Intent Router model provider (model: %s, base_url: %s)",
        cfg.router_model,
        cfg.base_url,
    )

    try:
        from observability.llm_telemetry import get_telemetry_callback_handler

        callbacks = [get_telemetry_callback_handler()]
    except Exception:
        callbacks = []

    return ChatOpenAI(
        model=cfg.router_model,
        api_key=cfg.api_key or "sk-dummy-key",
        base_url=cfg.base_url,
        temperature=cfg.router_temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        callbacks=callbacks,
        default_headers={
            "HTTP-Referer": "https://agent-mvp.local",
            "X-Title": "Agent MVP Recommendation Router",
        },
    )


def create_recommendation_model(config: Optional[RecommendationConfig] = None) -> BaseChatModel:
    """Create and configure the main chat model for the Recommendation Agent."""
    cfg = config or RecommendationConfig.from_settings()

    try:
        from observability.llm_telemetry import get_telemetry_callback_handler

        callbacks = [get_telemetry_callback_handler()]
    except Exception:
        callbacks = []

    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key or "sk-dummy-key",
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        callbacks=callbacks,
        default_headers={
            "HTTP-Referer": "https://agent-mvp.local",
            "X-Title": "Agent MVP Recommendation Agent",
        },
    )
