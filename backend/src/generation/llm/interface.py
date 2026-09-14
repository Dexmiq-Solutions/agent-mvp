"""Abstract base interface for LLM execution in the RAG generation pipeline."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Optional

from generation.llm.models import LLMResult, LLMStreamEvent
from generation.prompt.models import ConstructedPrompt


class BaseLLMInterface(ABC):
    """Abstract base interface for LLM inference execution.

    Decouples the RAG generation pipeline from any concrete LLM provider,
    SDK, or orchestration framework (such as OpenAI, Anthropic, or future
    LangGraph / DeepAgent workflows).

    The interface accepts a ConstructedPrompt (or structured prompt/messages),
    invokes the underlying model execution asynchronously with bounded timeouts
    and retries, and returns an application-level normalized LLMResult without
    leaking provider-specific types into business logic.
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
        prompt: ConstructedPrompt | Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute non-streaming LLM generation asynchronously.

        Args:
            prompt: Upstream ConstructedPrompt or structured messages.
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
        prompt: ConstructedPrompt | Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming LLM generation asynchronously, yielding token events.

        Args:
            prompt: Upstream ConstructedPrompt or structured messages.
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
