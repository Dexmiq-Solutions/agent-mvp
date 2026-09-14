"""Comprehensive test suite for the LLM Interface generation-stage component."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.core.config import Settings
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
from rag.generation.llm.adapters.openai import OpenAICompatibleLLMAdapter
from rag.generation.llm.config import LLMConfig
from rag.generation.llm.interface import BaseLLMInterface
from rag.generation.llm.models import LLMResult, LLMStreamEvent, LLMUsage
from rag.generation.llm.service import (
    LLMService,
    generate_async,
    generate_stream_async,
    get_llm_service,
    reset_llm_service,
)
from rag.generation.prompt.models import ConstructedPrompt


# ==============================================================================
# Helpers and Fixtures
# ==============================================================================

def make_sample_constructed_prompt(
    system_instruction: str = "You are a test assistant.",
    user_query: str = "What is the capital of France?",
    context_text: str = "Paris is the capital of France.",
) -> ConstructedPrompt:
    """Create a sample ConstructedPrompt instance matching Prompt Construction output."""
    user_prompt = f"## Retrieved Context\n{context_text}\n\n## User Query\n{user_query}"
    messages = (
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    )
    return ConstructedPrompt(
        system_instruction=system_instruction,
        user_query=user_query,
        context_text=context_text,
        user_prompt=user_prompt,
        messages=messages,
        project_id="test-project",
        context_items_count=1,
    )


def make_mock_choice(content: str = "Paris", finish_reason: str = "stop") -> MagicMock:
    """Create a mock ChatCompletionChoice object."""
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    return choice


def make_mock_usage(
    prompt_tokens: int = 15,
    completion_tokens: int = 5,
    total_tokens: int = 20,
) -> MagicMock:
    """Create a mock CompletionUsage object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    return usage


def make_mock_chat_completion(
    content: str = "Paris",
    finish_reason: str = "stop",
    usage: Optional[MagicMock] = None,
    model: str = "gpt-4o",
    completion_id: str = "chatcmpl-test-123",
) -> MagicMock:
    """Create a mock ChatCompletion response."""
    resp = MagicMock()
    resp.id = completion_id
    resp.model = model
    resp.choices = [make_mock_choice(content=content, finish_reason=finish_reason)]
    resp.usage = usage if usage is not None else make_mock_usage()
    return resp


class FakeLLMAdapter(BaseLLMInterface):
    """Fake LLM Adapter for testing dependency replacement without OpenAI SDK."""

    def __init__(self, response_text: str = "Fake answer", model: str = "fake-model") -> None:
        self._response_text = response_text
        self._model = model
        self.invocations: list[Any] = []

    @property
    def provider_name(self) -> str:
        return "fake_provider"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: Any,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.invocations.append({"prompt": prompt, "temp": temperature, "kwargs": kwargs})
        return LLMResult(
            content=self._response_text,
            finish_reason="stop",
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            metadata={"model": self._model, "provider": self.provider_name},
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
        self.invocations.append({"prompt": prompt, "stream": True})
        for word in self._response_text.split():
            yield LLMStreamEvent(delta=word + " ")
        yield LLMStreamEvent(delta="", finish_reason="stop", usage=LLMUsage(10, 5, 15))


@pytest.fixture(autouse=True)
def cleanup_service():
    """Reset the singleton service before and after each test."""
    reset_llm_service()
    yield
    reset_llm_service()


# ==============================================================================
# Model and Domain Contract Tests
# ==============================================================================

def test_llm_usage_normalization():
    """Verify LLMUsage model behavior and serialization."""
    usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150

    d = usage.to_dict()
    assert d == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


def test_llm_usage_missing_fields():
    """Verify LLMUsage handles missing fields cleanly without fabricating numbers."""
    usage = LLMUsage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_llm_result_contract():
    """Verify LLMResult normalizes content and preserves raw generation without manipulation."""
    res = LLMResult(
        content="Paris is the capital.",
        finish_reason="stop",
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        metadata={"model": "gpt-4o", "latency_ms": 120.5},
    )
    assert res.content == "Paris is the capital."
    assert res.generated_content == "Paris is the capital."
    assert res.finish_reason == "stop"
    assert res.usage.total_tokens == 15
    assert res.metadata["latency_ms"] == 120.5

    d = res.to_dict()
    assert d["content"] == "Paris is the capital."
    assert d["finish_reason"] == "stop"
    assert d["usage"]["total_tokens"] == 15


def test_llm_stream_event_contract():
    """Verify LLMStreamEvent marks finality and serialization."""
    event1 = LLMStreamEvent(delta="Hello ")
    assert not event1.is_final
    assert event1.delta == "Hello "

    event2 = LLMStreamEvent(delta="", finish_reason="stop", usage=LLMUsage(10, 2, 12))
    assert event2.is_final
    assert event2.finish_reason == "stop"
    assert event2.usage.total_tokens == 12


# ==============================================================================
# Configuration Tests
# ==============================================================================

def test_llm_config_defaults():
    """Verify LLMConfig default values and safe representation."""
    config = LLMConfig()
    assert config.model == "gpt-4o"
    assert config.api_key is None
    assert config.base_url is None
    assert config.timeout == 30.0
    assert config.max_retries == 2
    assert config.temperature == 0.0

    # Verify API key is masked in repr
    config_with_key = LLMConfig(api_key="sk-secret-key-12345")
    repr_str = repr(config_with_key)
    assert "sk-secret-key-12345" not in repr_str
    assert "***" in repr_str


def test_llm_config_from_settings():
    """Verify LLMConfig correctly reads from application Settings."""
    settings = Settings(
        OPENAI_API_KEY="sk-custom-env-key",
        LLM_BASE_URL="https://custom.endpoint.com/v1",
        LLM_MODEL="gpt-4o-mini",
        LLM_TIMEOUT=15.0,
        LLM_MAX_RETRIES=3,
        LLM_TEMPERATURE=0.7,
        LLM_MAX_TOKENS=512,
    )
    config = LLMConfig.from_settings(settings)
    assert config.api_key == "sk-custom-env-key"
    assert config.base_url == "https://custom.endpoint.com/v1"
    assert config.model == "gpt-4o-mini"
    assert config.timeout == 15.0
    assert config.max_retries == 3
    assert config.temperature == 0.7
    assert config.max_tokens == 512


# ==============================================================================
# Non-Streaming Inference & Provider Invocation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_successful_non_streaming_inference():
    """Verify constructed prompt produces normalized LLMResult using mocked AsyncOpenAI."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_resp = make_mock_chat_completion(
        content="Paris is indeed the capital.",
        finish_reason="stop",
        usage=make_mock_usage(10, 6, 16),
        model="gpt-4o",
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(model="gpt-4o", timeout=10.0),
        client=mock_client,
    )

    prompt = make_sample_constructed_prompt()
    result = await adapter.generate(prompt)

    assert isinstance(result, LLMResult)
    assert result.content == "Paris is indeed the capital."
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 16
    assert result.metadata["model"] == "gpt-4o"
    assert result.metadata["provider"] == "openai_compatible"
    assert result.metadata["attempts"] == 1

    # Verify provider received the expected structured messages
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_provider_invocation_parameters_passed():
    """Verify temperature, max_tokens, stop sequences, and top_p are forwarded correctly."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=make_mock_chat_completion())

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(model="gpt-4o", temperature=0.1, max_tokens=100),
        client=mock_client,
    )

    prompt = make_sample_constructed_prompt()
    await adapter.generate(
        prompt,
        temperature=0.5,
        max_tokens=250,
        top_p=0.9,
        stop=["END", "STOP"],
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 250
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["stop"] == ["END", "STOP"]


@pytest.mark.asyncio
async def test_missing_usage_handled_gracefully():
    """Verify that when provider omits usage data, usage is None (not fabricated)."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_resp = make_mock_chat_completion(content="Result without usage")
    mock_resp.usage = None  # Provider did not supply usage
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = OpenAICompatibleLLMAdapter(client=mock_client)
    result = await adapter.generate("Simple query")

    assert result.content == "Result without usage"
    assert result.usage is None


@pytest.mark.asyncio
async def test_finish_reason_preserved():
    """Verify finish reason ('length', 'stop') is preserved in LLMResult."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_resp = make_mock_chat_completion(content="Truncated output...", finish_reason="length")
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = OpenAICompatibleLLMAdapter(client=mock_client)
    result = await adapter.generate("Give me long text")

    assert result.finish_reason == "length"


# ==============================================================================
# Prompt Input Validation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_rejects_none_prompt():
    """Verify adapter raises LLMValidationError if prompt is None."""
    adapter = OpenAICompatibleLLMAdapter()
    with pytest.raises(LLMValidationError, match="Prompt cannot be None"):
        await adapter.generate(None)


@pytest.mark.asyncio
async def test_rejects_empty_string_prompt():
    """Verify adapter raises LLMValidationError if prompt is whitespace or empty string."""
    adapter = OpenAICompatibleLLMAdapter()
    with pytest.raises(LLMValidationError, match="cannot be empty"):
        await adapter.generate("   ")


@pytest.mark.asyncio
async def test_rejects_empty_messages_list():
    """Verify adapter raises LLMValidationError if message list is empty."""
    adapter = OpenAICompatibleLLMAdapter()
    with pytest.raises(LLMValidationError, match="cannot be empty"):
        await adapter.generate([])


@pytest.mark.asyncio
async def test_rejects_malformed_message_dict():
    """Verify adapter raises LLMValidationError if message dict lacks role or content."""
    adapter = OpenAICompatibleLLMAdapter()
    with pytest.raises(LLMValidationError, match="Expected dict with 'role' and 'content'"):
        await adapter.generate([{"role": "user"}])


# ==============================================================================
# Timeout and Retry Behavior Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_timeout_error_translation():
    """Verify APITimeoutError or asyncio.TimeoutError translates to LLMTimeoutError."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    # APITimeoutError takes request argument
    mock_client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=0, timeout=2.0),
        client=mock_client,
    )

    prompt = make_sample_constructed_prompt()
    with pytest.raises(LLMTimeoutError) as exc_info:
        await adapter.generate(prompt)

    assert "timed out" in exc_info.value.message.lower()
    assert exc_info.value.status_code == 408


@pytest.mark.asyncio
async def test_retryable_transient_failure_succeeds_on_retry():
    """Verify transient error (RateLimitError 429) retries and succeeds."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    # RateLimitError requires message, response, body
    mock_response = MagicMock(status_code=429, headers={})
    rate_limit_err = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

    success_resp = make_mock_chat_completion(content="Recovered after retry!")

    mock_client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_err, success_resp]
    )

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=2, retry_delay=0.01, retry_backoff=1.0),
        client=mock_client,
    )

    result = await adapter.generate(make_sample_constructed_prompt())
    assert result.content == "Recovered after retry!"
    assert result.metadata["attempts"] == 2
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_retryable_failure_exhausts_retries():
    """Verify exhausted retries on transient errors raises application exception."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_response = MagicMock(status_code=429, headers={})
    rate_limit_err = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

    mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=2, retry_delay=0.01, retry_backoff=1.0),
        client=mock_client,
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        await adapter.generate(make_sample_constructed_prompt())

    assert exc_info.value.status_code == 429
    assert mock_client.chat.completions.create.call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_non_retryable_failure_fails_immediately():
    """Verify permanent failure (AuthenticationError 401) is not retried."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_response = MagicMock(status_code=401, headers={})
    auth_err = AuthenticationError("Invalid API key provided", response=mock_response, body=None)

    mock_client.chat.completions.create = AsyncMock(side_effect=auth_err)

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=3, retry_delay=0.01),
        client=mock_client,
    )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await adapter.generate(make_sample_constructed_prompt())

    assert exc_info.value.status_code == 401
    # Crucial: Must only be called once, NOT retried!
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_bad_request_error_fails_immediately():
    """Verify BadRequestError (400) translates to LLMValidationError without retries."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    mock_response = MagicMock(status_code=400, headers={})
    bad_req_err = BadRequestError("Invalid model parameter", response=mock_response, body=None)

    mock_client.chat.completions.create = AsyncMock(side_effect=bad_req_err)

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=2, retry_delay=0.01),
        client=mock_client,
    )

    with pytest.raises(LLMValidationError):
        await adapter.generate(make_sample_constructed_prompt())

    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_connection_error_translation():
    """Verify APIConnectionError translates to LLMUnavailableError."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    conn_err = APIConnectionError(request=MagicMock())
    mock_client.chat.completions.create = AsyncMock(side_effect=conn_err)

    adapter = OpenAICompatibleLLMAdapter(
        config=LLMConfig(max_retries=0),
        client=mock_client,
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        await adapter.generate(make_sample_constructed_prompt())

    assert exc_info.value.status_code == 503


# ==============================================================================
# Streaming Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_streaming_event_delivery_and_completion():
    """Verify streaming token delivery, completion status, and usage."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    # Create mock stream chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Paris"), finish_reason=None)]
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content=" is"), finish_reason=None)]
    chunk2.usage = None

    chunk3 = MagicMock()
    chunk3.choices = [MagicMock(delta=MagicMock(content=" capital."), finish_reason="stop")]
    chunk3.usage = make_mock_usage(10, 3, 13)

    async def mock_stream_gen():
        for chunk in [chunk1, chunk2, chunk3]:
            yield chunk

    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream_gen())

    adapter = OpenAICompatibleLLMAdapter(client=mock_client)
    events: list[LLMStreamEvent] = []

    async for event in adapter.generate_stream(make_sample_constructed_prompt()):
        events.append(event)

    assert len(events) == 3
    assert events[0].delta == "Paris"
    assert not events[0].is_final

    assert events[1].delta == " is"
    assert not events[1].is_final

    assert events[2].delta == " capital."
    assert events[2].is_final
    assert events[2].finish_reason == "stop"
    assert events[2].usage is not None
    assert events[2].usage.total_tokens == 13


@pytest.mark.asyncio
async def test_streaming_error_translation():
    """Verify provider error during streaming translates to LLMStreamError."""
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()

    async def failing_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Start"), finish_reason=None)]
        chunk.usage = None
        yield chunk
        raise APIConnectionError(request=MagicMock())

    mock_client.chat.completions.create = AsyncMock(return_value=failing_stream())

    adapter = OpenAICompatibleLLMAdapter(client=mock_client)

    events: list[LLMStreamEvent] = []
    with pytest.raises(LLMStreamError) as exc_info:
        async for event in adapter.generate_stream("Test query"):
            events.append(event)

    assert len(events) == 1
    assert events[0].delta == "Start"
    assert "interrupted" in exc_info.value.message.lower()


# ==============================================================================
# Service & Dependency Replacement Tests (Future Deep Agent Compatibility)
# ==============================================================================

@pytest.mark.asyncio
async def test_llm_service_with_fake_adapter_dependency_replacement():
    """Verify LLMService operates seamlessly against a FakeLLMAdapter with 0 OpenAI dependency."""
    fake_adapter = FakeLLMAdapter(response_text="Custom Agent Answer")
    service = LLMService(adapter=fake_adapter)

    prompt = make_sample_constructed_prompt()
    result = await service.generate(prompt)

    assert result.content == "Custom Agent Answer"
    assert result.metadata["provider"] == "fake_provider"
    assert len(fake_adapter.invocations) == 1


@pytest.mark.asyncio
async def test_llm_service_streaming_with_fake_adapter():
    """Verify streaming works with fake adapter dependency injection."""
    fake_adapter = FakeLLMAdapter(response_text="Word1 Word2")
    service = LLMService(adapter=fake_adapter)

    chunks: list[str] = []
    async for event in service.generate_stream(make_sample_constructed_prompt()):
        if event.delta:
            chunks.append(event.delta)

    assert "".join(chunks).strip() == "Word1 Word2"


@pytest.mark.asyncio
async def test_singleton_get_and_reset_service():
    """Verify singleton get_llm_service and reset_llm_service lifecycle."""
    s1 = get_llm_service()
    s2 = get_llm_service()
    assert s1 is s2

    reset_llm_service()
    s3 = get_llm_service()
    assert s3 is not s1


@pytest.mark.asyncio
async def test_module_level_convenience_functions():
    """Verify generate_async and generate_stream_async module functions."""
    fake = FakeLLMAdapter(response_text="Module level answer")
    service = LLMService(adapter=fake)

    prompt = make_sample_constructed_prompt()
    res = await generate_async(prompt, service=service)
    assert res.content == "Module level answer"

    streamed: list[str] = []
    async for event in generate_stream_async(prompt, service=service):
        if event.delta:
            streamed.append(event.delta)
    assert "".join(streamed).strip() == "Module level answer"
