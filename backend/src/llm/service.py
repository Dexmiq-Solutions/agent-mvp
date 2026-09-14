"""LLM service providing application-level inference orchestration."""

from collections.abc import AsyncIterator
from typing import Any, Optional

from core.config import Settings, get_settings
from llm.adapters.openai import OpenAICompatibleLLMAdapter
from llm.config import LLMConfig
from llm.interface import BaseLLMInterface
from llm.models import LLMResult, LLMStreamEvent
from observability.logging import get_logger

logger = get_logger(__name__)

_llm_service: Optional["LLMService"] = None


class LLMService:
    """Application-level service orchestrating LLM execution.

    Positions as a root-level capability usable by RAG generation, autonomous Agents,
    Tools, and Workflows. Decoupled from concrete LLM providers via BaseLLMInterface
    dependency injection.
    """

    def __init__(
        self,
        adapter: Optional[BaseLLMInterface] = None,
        config: Optional[LLMConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize LLMService.

        Args:
            adapter: Optional provider adapter implementing BaseLLMInterface.
                     Defaults to OpenAICompatibleLLMAdapter.
            config: Optional LLMConfig instance.
            settings: Optional application Settings instance.
        """
        self._settings = settings or get_settings()
        self._config = config or LLMConfig.from_settings(self._settings)
        self._adapter = adapter or OpenAICompatibleLLMAdapter(config=self._config)

    @property
    def adapter(self) -> BaseLLMInterface:
        """Return the active provider adapter."""
        return self._adapter

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration."""
        return self._config

    async def generate(
        self,
        prompt: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute non-streaming LLM generation.

        Args:
            prompt: Prompt object with .to_messages(), sequence of message dicts, or raw string.
            temperature: Optional per-request temperature override.
            max_tokens: Optional per-request max tokens override.
            timeout: Optional per-request timeout in seconds.
            **kwargs: Additional generation parameters passed to adapter.

        Returns:
            LLMResult: Application-level normalized result.
        """
        return await self._adapter.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

    async def generate_stream(
        self,
        prompt: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming LLM generation, yielding token events.

        Args:
            prompt: Prompt object with .to_messages(), sequence of message dicts, or raw string.
            temperature: Optional per-request temperature override.
            max_tokens: Optional per-request max tokens override.
            timeout: Optional per-request timeout in seconds.
            **kwargs: Additional generation parameters passed to adapter.

        Yields:
            LLMStreamEvent: Stream event containing token delta, finish status, or usage.
        """
        async for event in self._adapter.generate_stream(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        ):
            yield event


def get_llm_service(
    adapter: Optional[BaseLLMInterface] = None,
    config: Optional[LLMConfig] = None,
    settings: Optional[Settings] = None,
) -> LLMService:
    """Retrieve or initialize the singleton LLMService instance.

    Args:
        adapter: Optional BaseLLMInterface adapter override.
        config: Optional LLMConfig override.
        settings: Optional Settings override.

    Returns:
        LLMService: Configured service instance.
    """
    global _llm_service
    if _llm_service is None or any(arg is not None for arg in (adapter, config, settings)):
        service = LLMService(
            adapter=adapter,
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (adapter, config, settings)):
            _llm_service = service
        return service
    return _llm_service


def reset_llm_service() -> None:
    """Reset the cached singleton LLMService instance (useful for test isolation)."""
    global _llm_service
    _llm_service = None


async def generate_async(
    prompt: Any,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    service: Optional[LLMService] = None,
    **kwargs: Any,
) -> LLMResult:
    """Module-level convenience function for non-streaming LLM generation."""
    active_service = service or get_llm_service()
    return await active_service.generate(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )


async def generate_stream_async(
    prompt: Any,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    service: Optional[LLMService] = None,
    **kwargs: Any,
) -> AsyncIterator[LLMStreamEvent]:
    """Module-level convenience generator for streaming LLM generation."""
    active_service = service or get_llm_service()
    async for event in active_service.generate_stream(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    ):
        yield event
