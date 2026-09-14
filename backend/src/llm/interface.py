"""Abstract base interface for LLM execution in the application."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Optional

from llm.models import LLMResult, LLMStreamEvent


class BaseLLMInterface(ABC):
    """Abstract base interface for LLM inference execution.

    Decouples callers (such as RAG generation, autonomous Agents, Tools, and Workflows)
    from any concrete LLM provider, SDK, or orchestration framework.

    The interface accepts a prompt (such as a prompt object with a .to_messages() method,
    a structured sequence of chat message dicts, or a raw string), invokes the underlying
    model execution asynchronously with bounded timeouts and retries, and returns an
    application-level normalized LLMResult without leaking provider-specific types into
    business logic.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g. 'openai_compatible', 'fake')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""

    @abstractmethod
    async def generate(
        self,
        prompt: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute non-streaming LLM generation asynchronously.

        Args:
            prompt: Prompt object with .to_messages(), sequence of message dicts, or raw string.
            temperature: Optional per-request sampling temperature override.
            max_tokens: Optional per-request maximum output tokens override.
            timeout: Optional per-request timeout in seconds.
            **kwargs: Additional provider-specific options.

        Returns:
            LLMResult: Application-level normalized result.

        Raises:
            LLMValidationError: If input prompt is invalid or empty.
            LLMTimeoutError: If execution exceeds timeout.
            LLMRateLimitError: If provider rate limit is encountered.
            LLMAuthenticationError: If provider credentials/auth fails.
            LLMUnavailableError: If provider connection fails.
            LLMProviderError: If provider execution fails.
        """

    @abstractmethod
    async def generate_stream(
        self,
        prompt: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming LLM generation asynchronously, yielding token events.

        Args:
            prompt: Prompt object with .to_messages(), sequence of message dicts, or raw string.
            temperature: Optional per-request sampling temperature override.
            max_tokens: Optional per-request maximum output tokens override.
            timeout: Optional per-request timeout in seconds.
            **kwargs: Additional provider-specific options.

        Yields:
            LLMStreamEvent: Normalized stream event containing token deltas and completion status.

        Raises:
            LLMValidationError: If input prompt is invalid or empty.
            LLMTimeoutError: If execution exceeds timeout.
            LLMRateLimitError: If provider rate limit is encountered.
            LLMAuthenticationError: If provider credentials/auth fails.
            LLMUnavailableError: If provider connection fails.
            LLMProviderError: If provider execution fails.
            LLMStreamError: If stream breaks unexpectedly.
        """
