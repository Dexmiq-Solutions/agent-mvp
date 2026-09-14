"""Minimal, asynchronous LLM provider abstraction for query transformation."""

from abc import ABC, abstractmethod
import asyncio
from typing import Any, Optional

import httpx

from observability.logging import get_logger
from exceptions.retrieval import (
    TransformationProviderError,
    TransformationTimeoutError,
    TransformationUnavailableError,
)

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract minimal interface for text completion providers.

    Focused strictly on asynchronous completion without framework overhead.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Execute text completion asynchronously.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system instruction prompt.
            model: Optional model identifier override.
            temperature: Sampling temperature (default: 0.0).
            max_tokens: Maximum tokens to generate.
            timeout: Optional per-request timeout in seconds.

        Returns:
            str: Generated completion text.

        Raises:
            TransformationTimeoutError: If request times out.
            TransformationUnavailableError: If provider service is unreachable.
            TransformationProviderError: If provider returns an error response.
        """


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Minimal HTTP client for OpenAI-compatible chat completion APIs.

    Supports OpenAI, OpenRouter, DeepSeek, vLLM, Ollama, and local mock endpoints
    via standard /chat/completions schema.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o",
        default_timeout: float = 5.0,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._default_model = default_model
        self._default_timeout = default_timeout
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._external_client = http_client

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Execute chat completion request with bounded retries on transient errors."""
        effective_model = model or self._default_model
        effective_timeout = timeout if timeout is not None else self._default_timeout

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        total_attempts = 1 + self._max_retries

        for attempt in range(1, total_attempts + 1):
            try:
                if self._external_client is not None:
                    response = await self._external_client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=effective_timeout,
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=effective_timeout,
                        )

                # Check HTTP status
                if response.status_code >= 400:
                    status = response.status_code
                    # Determine if transient and retryable
                    is_transient = status in {408, 429, 500, 502, 503, 504}
                    if is_transient and attempt < total_attempts:
                        delay = self._retry_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Transient provider HTTP %d on attempt %d/%d; retrying in %.2fs",
                            status,
                            attempt,
                            total_attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    error_text = response.text[:200]
                    raise TransformationProviderError(
                        f"Provider HTTP error {status}: {error_text}"
                    )

                data = response.json()
                choices = data.get("choices")
                if not choices or not isinstance(choices, list):
                    raise TransformationProviderError(
                        "Malformed provider response: missing or invalid 'choices'."
                    )

                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None:
                    raise TransformationProviderError(
                        "Malformed provider response: choice message content is None."
                    )

                return str(content)

            except httpx.TimeoutException as exc:
                if attempt < total_attempts:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Provider timeout on attempt %d/%d; retrying in %.2fs",
                        attempt,
                        total_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Provider call timed out after %d attempts", total_attempts)
                raise TransformationTimeoutError(
                    f"Provider request timed out after {effective_timeout}s: {exc}",
                    original_error=exc,
                ) from exc

            except httpx.ConnectError as exc:
                if attempt < total_attempts:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Provider connection error on attempt %d/%d; retrying in %.2fs",
                        attempt,
                        total_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Provider unreachable after %d attempts", total_attempts)
                raise TransformationUnavailableError(
                    f"Provider service unreachable at {self._base_url}: {exc}",
                    original_error=exc,
                ) from exc

            except (TransformationProviderError, TransformationTimeoutError, TransformationUnavailableError):
                raise

            except Exception as exc:
                logger.error("Unexpected error during provider completion: %s", exc, exc_info=True)
                raise TransformationProviderError(
                    f"Unexpected provider completion error: {exc}",
                    original_error=exc,
                ) from exc

        raise TransformationProviderError("Provider request failed after exhausting retries.")
