"""Abstract base interface and exceptions for LLM providers."""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base exception for all LLM errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class LLMProviderError(LLMError):
    """Raised when an external LLM provider API call fails."""


class BaseLLMClient(ABC):
    """Abstract interface for LLM clients.

    Decouples contextual enrichment and other agent/RAG components from specific
    LLM providers (e.g. OpenAI, Anthropic, OpenRouter, local models).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the active model."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: User or task prompt.
            system_prompt: Optional system instruction prompt.

        Returns:
            str: Generated text response.

        Raises:
            LLMProviderError: If the external provider call fails.
        """
