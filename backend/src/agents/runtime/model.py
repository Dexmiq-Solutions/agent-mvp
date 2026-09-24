"""Model factory and provider abstraction for the Agent runtime."""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agents.runtime.config import AgentConfig
from exceptions.agent import AgentConfigurationError, AgentModelError
from observability.logging import get_logger

logger = get_logger(__name__)


def create_agent_model(config: Optional[AgentConfig] = None) -> BaseChatModel:
    """Create and configure a LangChain-compatible chat model for the Agent runtime.

    Connects to OpenRouter or any OpenAI-compatible provider using the specified configuration.
    Maintains provider independence: callers receive a BaseChatModel interface without coupling
    to provider-specific SDK internals.

    Args:
        config: Optional AgentConfig instance. If omitted, loaded from application settings.

    Returns:
        BaseChatModel: Configured chat model instance ready for tool binding and agent execution.

    Raises:
        AgentConfigurationError: If required model credentials or configuration parameters are invalid.
        AgentModelError: If model instantiation fails.
    """
    cfg = config or AgentConfig.from_settings()
    cfg.validate()

    logger.debug(
        "Initializing Agent model provider (model: %s, base_url: %s)",
        cfg.model,
        cfg.base_url,
    )

    try:
        model = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
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
        return model
    except AgentConfigurationError:
        raise
    except Exception as exc:
        logger.error("Failed to construct Agent model: %s", exc)
        raise AgentModelError(f"Failed to initialize Agent model: {exc}", original_error=exc) from exc
