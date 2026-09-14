"""OpenAI-compatible LLM provider adapter using official OpenAI Python SDK."""

import asyncio
from collections.abc import AsyncIterator
import time
from typing import Any, Optional

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from app.core.logging import get_logger
from exceptions.generation import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMValidationError,
)
from rag.generation.llm.config import LLMConfig
from rag.generation.llm.interface import BaseLLMInterface
from rag.generation.llm.models import LLMResult, LLMStreamEvent, LLMUsage
from rag.generation.prompt.models import ConstructedPrompt

logger = get_logger(__name__)


class OpenAICompatibleLLMAdapter(BaseLLMInterface):
    """OpenAI-compatible LLM adapter powered by the official OpenAI Python SDK.

    Operates asynchronously without blocking the event loop. Reuses the AsyncOpenAI
    client instance safely across concurrent calls. Supports OpenAI, OpenRouter,
    vLLM, Ollama, LocalAI, and other OpenAI-compatible endpoints via configuration.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        """Initialize OpenAICompatibleLLMAdapter.

        Args:
            config: Optional LLMConfig instance. If omitted, loads from application Settings/env.
            client: Optional pre-configured AsyncOpenAI client instance (useful for testing/mocking).
        """
        self._config = config or LLMConfig.from_settings()

        if client is not None:
            self._client = client
        else:
            # Sensitive credentials loaded strictly from config/env
            api_key = self._config.api_key or "EMPTY"
            base_url = self._config.base_url

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._config.timeout,
                max_retries=0,  # We manage bounded retries explicitly with full observability
            )

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        """Return configured model identifier."""
        return self._config.model

    @property
    def config(self) -> LLMConfig:
        """Return active LLM configuration."""
        return self._config

    def _extract_messages(self, prompt: Any) -> list[dict[str, str]]:
        """Extract structured chat messages from input prompt representation.

        Preserves system instructions and user prompt without re-constructing
        or modifying retrieval context.

        Args:
            prompt: ConstructedPrompt, Sequence of message dicts, or raw text.

        Returns:
            list[dict[str, str]]: List of standard chat messages.

        Raises:
            LLMValidationError: If prompt is empty or structurally invalid.
        """
        if prompt is None:
            raise LLMValidationError("Prompt cannot be None.")

        if isinstance(prompt, ConstructedPrompt):
            msgs = prompt.to_messages()
            if not msgs:
                raise LLMValidationError("ConstructedPrompt contains no messages.")
            return msgs

        if isinstance(prompt, (list, tuple)):
            if not prompt:
                raise LLMValidationError("Prompt message list cannot be empty.")
            messages: list[dict[str, str]] = []
            for idx, msg in enumerate(prompt):
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise LLMValidationError(
                        f"Invalid message format at index {idx}. Expected dict with 'role' and 'content'."
                    )
                messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
            return messages

        if isinstance(prompt, str):
            stripped = prompt.strip()
            if not stripped:
                raise LLMValidationError("Input prompt string cannot be empty.")
            return [{"role": "user", "content": stripped}]

        # Fallback check for to_messages method
        if hasattr(prompt, "to_messages") and callable(prompt.to_messages):
            msgs = prompt.to_messages()
            if not msgs:
                raise LLMValidationError("Prompt to_messages() returned no messages.")
            return [dict(m) for m in msgs]

        raise LLMValidationError(
            f"Unsupported prompt type: {type(prompt).__name__}. Expected ConstructedPrompt, list of dicts, or str."
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determine if an exception represents a transient, retryable failure."""
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, asyncio.TimeoutError)):
            return True
        if isinstance(exc, InternalServerError):
            return True
        if isinstance(exc, APIStatusError):
            # HTTP 429, 408, or 5xx server errors are transient
            status = getattr(exc, "status_code", None)
            if status in {408, 429} or (status and status >= 500):
                return True
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return True
        return False

    def _translate_exception(self, exc: Exception, effective_timeout: float) -> LLMError:
        """Translate provider and SDK exceptions to application domain exceptions."""
        if isinstance(exc, AuthenticationError):
            return LLMAuthenticationError(
                "Provider authentication failed. Verify configured API credentials.",
                status_code=401,
                original_error=exc,
            )

        if isinstance(exc, RateLimitError):
            return LLMRateLimitError(
                "Provider rate limit exceeded. Please retry after backoff.",
                status_code=429,
                original_error=exc,
            )

        if isinstance(exc, (APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return LLMTimeoutError(
                f"LLM request timed out after {effective_timeout:.2f}s.",
                status_code=408,
                original_error=exc,
            )

        if isinstance(exc, (APIConnectionError, httpx.ConnectError)):
            return LLMUnavailableError(
                "LLM provider service unreachable or connection failed.",
                status_code=503,
                original_error=exc,
            )

        if isinstance(exc, BadRequestError):
            return LLMValidationError(
                f"Provider rejected request structure or parameters: {exc.message}",
                original_error=exc,
            )

        if isinstance(exc, NotFoundError):
            return LLMConfigurationError(
                f"Model or endpoint not found on provider: {exc.message}",
                original_error=exc,
            )

        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            return LLMProviderError(
                f"Provider returned HTTP error {status}: {exc.message}",
                status_code=status,
                original_error=exc,
            )

        if isinstance(exc, APIError):
            return LLMProviderError(
                f"Provider API error: {exc.message}",
                original_error=exc,
            )

        if isinstance(exc, LLMError):
            return exc

        return LLMProviderError(
            f"Unexpected LLM inference error: {exc}",
            original_error=exc,
        )

    def _normalize_usage(self, raw_usage: Any) -> Optional[LLMUsage]:
        """Safely extract normalized token usage without fabricating missing values."""
        if raw_usage is None:
            return None

        prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
        completion_tokens = getattr(raw_usage, "completion_tokens", None)
        total_tokens = getattr(raw_usage, "total_tokens", None)

        if isinstance(raw_usage, dict):
            prompt_tokens = raw_usage.get("prompt_tokens")
            completion_tokens = raw_usage.get("completion_tokens")
            total_tokens = raw_usage.get("total_tokens")

        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None

        return LLMUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def generate(
        self,
        prompt: ConstructedPrompt | Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute non-streaming completion with bounded retries and timeouts."""
        messages = self._extract_messages(prompt)

        effective_model = self._config.model
        effective_temp = temperature if temperature is not None else self._config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        effective_timeout = timeout if timeout is not None else self._config.timeout
        effective_top_p = kwargs.get("top_p", self._config.top_p)
        effective_stop = kwargs.get("stop", self._config.stop_sequences or None)

        payload_params: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": effective_temp,
        }
        if effective_max_tokens is not None:
            payload_params["max_tokens"] = effective_max_tokens
        if effective_top_p is not None:
            payload_params["top_p"] = effective_top_p
        if effective_stop:
            payload_params["stop"] = list(effective_stop)

        # Forward any caller extra options
        for key, val in kwargs.items():
            if key not in ("top_p", "stop"):
                payload_params[key] = val

        total_attempts = 1 + max(0, self._config.max_retries)
        start_time = time.perf_counter()

        for attempt in range(1, total_attempts + 1):
            try:
                attempt_start = time.perf_counter()

                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**payload_params),
                    timeout=effective_timeout,
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                choices = getattr(response, "choices", None)
                if not choices or len(choices) == 0:
                    raise LLMProviderError("Provider response contains no choices.")

                first_choice = choices[0]
                message = getattr(first_choice, "message", None)
                content = getattr(message, "content", None) or ""
                finish_reason = getattr(first_choice, "finish_reason", None)
                usage = self._normalize_usage(getattr(response, "usage", None))

                request_id = getattr(response, "_request_id", None) or getattr(response, "id", None)
                model_used = getattr(response, "model", effective_model)

                logger.info(
                    "LLM generation successful: model='%s', attempt=%d/%d, duration=%.2fms, finish_reason='%s', tokens=%s",
                    model_used,
                    attempt,
                    total_attempts,
                    elapsed_ms,
                    finish_reason,
                    usage.total_tokens if usage else "N/A",
                )

                return LLMResult(
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    metadata={
                        "model": model_used,
                        "provider": self.provider_name,
                        "latency_ms": round(elapsed_ms, 2),
                        "attempts": attempt,
                        "request_id": request_id,
                    },
                )

            except Exception as exc:
                is_transient = self._is_transient_error(exc)
                if is_transient and attempt < total_attempts:
                    delay = self._config.retry_delay * (self._config.retry_backoff ** (attempt - 1))
                    logger.warning(
                        "Transient error during LLM generation (attempt %d/%d): %s; retrying in %.2fs",
                        attempt,
                        total_attempts,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                translated = self._translate_exception(exc, effective_timeout)
                logger.error(
                    "LLM generation failed after attempt %d/%d: %s",
                    attempt,
                    total_attempts,
                    translated.message,
                )
                raise translated from exc

        raise LLMProviderError("LLM generation failed after exhausting all retry attempts.")

    async def generate_stream(
        self,
        prompt: ConstructedPrompt | Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming completion asynchronously, yielding token events."""
        messages = self._extract_messages(prompt)

        effective_model = self._config.model
        effective_temp = temperature if temperature is not None else self._config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        effective_timeout = timeout if timeout is not None else self._config.timeout
        effective_top_p = kwargs.get("top_p", self._config.top_p)
        effective_stop = kwargs.get("stop", self._config.stop_sequences or None)

        payload_params: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": effective_temp,
            "stream": True,
        }
        if effective_max_tokens is not None:
            payload_params["max_tokens"] = effective_max_tokens
        if effective_top_p is not None:
            payload_params["top_p"] = effective_top_p
        if effective_stop:
            payload_params["stop"] = list(effective_stop)

        # Stream options if requested
        payload_params["stream_options"] = {"include_usage": True}

        start_time = time.perf_counter()

        try:
            stream_resp = await asyncio.wait_for(
                self._client.chat.completions.create(**payload_params),
                timeout=effective_timeout,
            )
        except Exception as exc:
            translated = self._translate_exception(exc, effective_timeout)
            logger.error("Failed to initiate LLM stream: %s", translated.message)
            raise translated from exc

        try:
            async for chunk in stream_resp:
                choices = getattr(chunk, "choices", None) or []
                raw_usage = getattr(chunk, "usage", None)
                usage = self._normalize_usage(raw_usage)

                delta_text = ""
                finish_reason = None

                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    delta = getattr(first_choice, "delta", None)
                    if delta:
                        delta_text = getattr(delta, "content", None) or ""
                    finish_reason = getattr(first_choice, "finish_reason", None)

                # Yield event if there is delta content, finish reason, or usage
                if delta_text or finish_reason or usage:
                    yield LLMStreamEvent(
                        delta=delta_text,
                        finish_reason=finish_reason,
                        usage=usage,
                        metadata={
                            "model": getattr(chunk, "model", effective_model),
                            "provider": self.provider_name,
                        },
                    )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info("LLM stream completed in %.2fms", elapsed_ms)

        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            translated = self._translate_exception(exc, effective_timeout)
            logger.error("Error encountered during active stream: %s", translated.message)
            raise LLMStreamError(
                f"Stream interrupted unexpectedly: {translated.message}",
                original_error=exc,
            ) from exc
