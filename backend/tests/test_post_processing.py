"""Comprehensive test suite for the Post-processing generation-stage component."""

import asyncio
from dataclasses import dataclass
from typing import Any
import pytest

from app.core.config import Settings
from exceptions.generation import (
    GenerationError,
    PostProcessingError,
    PostProcessingValidationError,
    StructuredOutputError,
)
from rag.generation.llm.models import LLMResult, LLMUsage
from rag.generation.postprocessing.base import BasePostProcessor
from rag.generation.postprocessing.config import PostProcessingConfig
from rag.generation.postprocessing.models import PostProcessedResponse, ProcessedResponse
from rag.generation.postprocessing.service import (
    PostProcessingService,
    get_post_processing_service,
    post_process,
    post_process_async,
    reset_post_processing_service,
)


# ==============================================================================
# Helpers and Fixtures
# ==============================================================================

def make_sample_llm_result(
    content: str = "Paris is the capital of France.",
    finish_reason: str = "stop",
    input_tokens: int = 15,
    output_tokens: int = 7,
    total_tokens: int = 22,
    model: str = "gpt-4o",
    extra_metadata: dict[str, Any] | None = None,
) -> LLMResult:
    """Create a sample LLMResult matching LLM Interface output."""
    metadata = {"model": model, "provider": "openai_compatible"}
    if extra_metadata:
        metadata.update(extra_metadata)
    usage = LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    return LLMResult(
        content=content,
        finish_reason=finish_reason,
        usage=usage,
        metadata=metadata,
    )


@pytest.fixture(autouse=True)
def reset_service() -> None:
    """Ensure clean service singleton state for each test."""
    reset_post_processing_service()
    yield
    reset_post_processing_service()


# ==============================================================================
# Domain Models Tests
# ==============================================================================

class TestPostProcessingModels:
    """Verify domain response models behavior, serialization, and safety."""

    def test_processed_response_creation_and_properties(self) -> None:
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        response = ProcessedResponse(
            content="Normalized answer",
            finish_reason="stop",
            usage=usage,
            model="gpt-4o",
            structured_output=None,
            raw_content="  Normalized answer\r\n",
            metadata={"source": "test"},
        )

        assert response.content == "Normalized answer"
        assert response.text == "Normalized answer"
        assert response.finish_reason == "stop"
        assert response.usage == usage
        assert response.model == "gpt-4o"
        assert response.structured_output is None
        assert response.raw_content == "  Normalized answer\r\n"
        assert response.metadata == {"source": "test"}

    def test_post_processed_response_alias(self) -> None:
        assert PostProcessedResponse is ProcessedResponse

    def test_to_dict_serialization(self) -> None:
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        response = ProcessedResponse(
            content="Answer text",
            finish_reason="stop",
            usage=usage,
            model="gpt-4o",
            structured_output={"key": "val"},
            raw_content="Answer text",
            metadata={"model": "gpt-4o"},
        )
        d = response.to_dict()

        assert d["content"] == "Answer text"
        assert d["finish_reason"] == "stop"
        assert d["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        assert d["model"] == "gpt-4o"
        assert d["structured_output"] == {"key": "val"}
        assert d["raw_content"] == "Answer text"
        assert d["metadata"] == {"model": "gpt-4o"}

    def test_to_dict_with_none_usage(self) -> None:
        response = ProcessedResponse(
            content="Answer text",
            finish_reason="stop",
            usage=None,
        )
        assert response.to_dict()["usage"] is None

    def test_safe_repr_masking(self) -> None:
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        long_content = "A" * 200
        response = ProcessedResponse(
            content=long_content,
            finish_reason="stop",
            usage=usage,
            model="gpt-4o",
        )
        rep = repr(response)

        assert "ProcessedResponse" in rep
        assert "gpt-4o" in rep
        assert "tokens=15" in rep
        assert len(rep) < 150  # Must not dump entire 200 character string
        assert "..." in rep


# ==============================================================================
# Configuration Tests
# ==============================================================================

class TestPostProcessingConfig:
    """Verify configuration loading and defaults."""

    def test_default_config(self) -> None:
        config = PostProcessingConfig()
        assert config.strip_whitespace is True
        assert config.normalize_line_endings is True
        assert config.normalize_finish_reason is True
        assert config.allow_empty_content is False
        assert config.parse_json is False
        assert config.require_json is False
        assert config.strip_code_fences_for_json is True
        assert config.max_content_length is None

    def test_from_settings(self) -> None:
        settings = Settings(
            POST_PROCESSING_STRIP_WHITESPACE=False,
            POST_PROCESSING_NORMALIZE_LINE_ENDINGS=False,
            POST_PROCESSING_ALLOW_EMPTY_CONTENT=True,
            POST_PROCESSING_PARSE_JSON=True,
        )
        config = PostProcessingConfig.from_settings(settings)
        assert config.strip_whitespace is False
        assert config.normalize_line_endings is False
        assert config.allow_empty_content is True
        assert config.parse_json is True

    def test_from_env(self) -> None:
        config = PostProcessingConfig.from_env()
        assert isinstance(config, PostProcessingConfig)

    def test_config_repr(self) -> None:
        config = PostProcessingConfig()
        rep = repr(config)
        assert "PostProcessingConfig" in rep
        assert "strip_whitespace=True" in rep


# ==============================================================================
# Core Post-Processing Service Tests
# ==============================================================================

class TestPostProcessingServiceCore:
    """Verify core extraction, normalization, and preservation behavior."""

    def test_process_canonical_llm_result(self) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(
            content="  Paris is the capital of France.\r\n",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=8,
            total_tokens=28,
            model="gpt-4o",
        )

        response = service.process(result)

        assert isinstance(response, ProcessedResponse)
        assert response.content == "Paris is the capital of France."
        assert response.raw_content == "  Paris is the capital of France.\r\n"
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.input_tokens == 20
        assert response.usage.output_tokens == 8
        assert response.usage.total_tokens == 28
        assert response.model == "gpt-4o"
        assert response.metadata["postprocessing_success"] is True
        assert "postprocessing_duration_ms" in response.metadata
        assert response.metadata["raw_content_length"] == len("  Paris is the capital of France.\r\n")
        assert response.metadata["processed_content_length"] == len("Paris is the capital of France.")

    def test_deterministic_line_ending_normalization(self) -> None:
        service = PostProcessingService()
        crlf_content = "Line 1\r\nLine 2\r\nLine 3\r\n"
        result = make_sample_llm_result(content=crlf_content)

        response = service.process(result)

        assert response.content == "Line 1\nLine 2\nLine 3"
        assert "\r" not in response.content

    def test_deterministic_line_ending_normalization_disabled(self) -> None:
        config = PostProcessingConfig(normalize_line_endings=False, strip_whitespace=False)
        service = PostProcessingService(config=config)
        crlf_content = "Line 1\r\nLine 2\r\n"
        result = make_sample_llm_result(content=crlf_content)

        response = service.process(result)

        assert response.content == crlf_content

    def test_deterministic_whitespace_stripping(self) -> None:
        service = PostProcessingService()
        padded_content = "\n\n  Indented internal text:\n    - item 1\n    - item 2\n\n  "
        result = make_sample_llm_result(content=padded_content)

        response = service.process(result)

        # Leading and trailing outer whitespace stripped, internal indentation preserved
        assert response.content == "Indented internal text:\n    - item 1\n    - item 2"

    def test_deterministic_whitespace_stripping_disabled(self) -> None:
        config = PostProcessingConfig(strip_whitespace=False)
        service = PostProcessingService(config=config)
        padded_content = "  Preserve outer spaces  "
        result = make_sample_llm_result(content=padded_content)

        response = service.process(result)

        assert response.content == "  Preserve outer spaces  "

    def test_no_semantic_rewriting(self) -> None:
        """Verify that Post-processing never rewrites words or adds semantic changes."""
        service = PostProcessingService()
        factual_typo = "The eiffel tower is located in Madrid."
        result = make_sample_llm_result(content=factual_typo)

        response = service.process(result)

        # Factual error must NOT be altered by post-processing
        assert response.content == factual_typo

    def test_max_content_length_guard(self) -> None:
        config = PostProcessingConfig(max_content_length=10)
        service = PostProcessingService(config=config)
        result = make_sample_llm_result(content="1234567890EXTRA")

        response = service.process(result)

        assert response.content == "1234567890"


# ==============================================================================
# Finish Reason Normalization Tests
# ==============================================================================

class TestFinishReasonNormalization:
    """Verify provider finish reasons are mapped to standard application tokens."""

    @pytest.mark.parametrize(
        "provider_reason,expected_normalized",
        [
            ("stop", "stop"),
            ("STOP", "stop"),
            ("end_turn", "stop"),
            ("complete", "stop"),
            ("COMPLETED", "stop"),
            ("length", "length"),
            ("max_tokens", "length"),
            ("MAX_TOKENS", "length"),
            ("content_filter", "content_filter"),
            ("safety", "content_filter"),
            ("tool_calls", "tool_calls"),
            ("function_call", "tool_calls"),
            ("custom_unknown_reason", "custom_unknown_reason"),
        ],
    )
    def test_finish_reason_mappings(
        self,
        provider_reason: str,
        expected_normalized: str,
    ) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(finish_reason=provider_reason)

        response = service.process(result)

        assert response.finish_reason == expected_normalized
        if provider_reason != expected_normalized:
            assert response.metadata["raw_finish_reason"] == provider_reason

    def test_finish_reason_normalization_disabled(self) -> None:
        config = PostProcessingConfig(normalize_finish_reason=False)
        service = PostProcessingService(config=config)
        result = make_sample_llm_result(finish_reason="MAX_TOKENS")

        response = service.process(result)

        assert response.finish_reason == "MAX_TOKENS"


# ==============================================================================
# Structured Output Tests
# ==============================================================================

class TestStructuredOutputProcessing:
    """Verify structured output extraction, code block stripping, and validation."""

    def test_parse_valid_json_string(self) -> None:
        service = PostProcessingService()
        json_content = '{"answer": "Paris", "confidence": 0.99}'
        result = make_sample_llm_result(content=json_content)

        response = service.process(result, parse_json=True)

        assert response.structured_output == {"answer": "Paris", "confidence": 0.99}
        assert response.metadata["structured_output_parsed"] is True

    def test_parse_json_wrapped_in_markdown_code_fence(self) -> None:
        service = PostProcessingService()
        wrapped_json = "```json\n{\n  \"city\": \"Paris\",\n  \"country\": \"France\"\n}\n```"
        result = make_sample_llm_result(content=wrapped_json)

        response = service.process(result, parse_json=True)

        assert response.structured_output == {"city": "Paris", "country": "France"}
        assert response.metadata["structured_output_parsed"] is True

    def test_parse_json_wrapped_in_generic_code_fence(self) -> None:
        service = PostProcessingService()
        wrapped_json = "```\n{\"result\": [1, 2, 3]}\n```"
        result = make_sample_llm_result(content=wrapped_json)

        response = service.process(result, parse_json=True)

        assert response.structured_output == {"result": [1, 2, 3]}

    def test_invalid_json_with_require_json_raises_error(self) -> None:
        service = PostProcessingService()
        invalid_json = "This is not valid JSON at all."
        result = make_sample_llm_result(content=invalid_json)

        with pytest.raises(StructuredOutputError) as exc_info:
            service.process(result, parse_json=True, require_json=True)

        assert "Failed to parse structured JSON output" in str(exc_info.value)
        assert isinstance(exc_info.value, PostProcessingError)
        assert isinstance(exc_info.value, GenerationError)

    def test_invalid_json_without_require_json_falls_back_gracefully(self) -> None:
        service = PostProcessingService()
        invalid_json = "This is plain text."
        result = make_sample_llm_result(content=invalid_json)

        response = service.process(result, parse_json=True, require_json=False)

        assert response.content == "This is plain text."
        assert response.structured_output is None
        assert response.metadata["structured_output_parsed"] is False
        assert "json_parse_error" in response.metadata

    def test_parse_json_disabled_by_default(self) -> None:
        service = PostProcessingService()
        json_content = '{"city": "Paris"}'
        result = make_sample_llm_result(content=json_content)

        response = service.process(result)

        assert response.content == '{"city": "Paris"}'
        assert response.structured_output is None
        assert "structured_output_parsed" not in response.metadata


# ==============================================================================
# Duck-Typing and Provider Independence Tests
# ==============================================================================

class TestDuckTypingAndProviderIndependence:
    """Verify Post-processing operates on multiple input representations without vendor SDK coupling."""

    def test_process_dictionary_input(self) -> None:
        service = PostProcessingService()
        dict_input = {
            "content": "  Dictionary based response.  ",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
            },
            "metadata": {"model": "claude-3-5-sonnet", "custom_id": "abc"},
        }

        response = service.process(dict_input)

        assert response.content == "Dictionary based response."
        assert response.finish_reason == "stop"
        assert response.model == "claude-3-5-sonnet"
        assert response.usage == LLMUsage(input_tokens=12, output_tokens=4, total_tokens=16)
        assert response.metadata["custom_id"] == "abc"

    def test_process_raw_string_input(self) -> None:
        service = PostProcessingService()
        raw_str = "  Direct string output from agent.  \n"

        response = service.process(raw_str)

        assert response.content == "Direct string output from agent."
        assert response.finish_reason == "stop"
        assert response.usage is None

    def test_process_duck_typed_custom_object(self) -> None:
        @dataclass
        class CustomAgentResult:
            content: str
            finish_reason: str = "end_turn"
            model: str = "deep-agent-v1"

        custom_result = CustomAgentResult(content="  Result from deep agent.  ")
        service = PostProcessingService()

        response = service.process(custom_result)

        assert response.content == "Result from deep agent."
        assert response.finish_reason == "stop"
        assert response.model == "deep-agent-v1"

    def test_process_duck_typed_generated_content_attr(self) -> None:
        class DuckObj:
            generated_content = "Output from generated_content"
            finish_reason = "stop"

        service = PostProcessingService()
        response = service.process(DuckObj())
        assert response.content == "Output from generated_content"

    def test_runtime_model_override(self) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(content="Hello", model="gpt-4o")

        response = service.process(result, model="custom-override-model")

        assert response.model == "custom-override-model"

    def test_runtime_extra_metadata_merge(self) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(content="Hello")

        response = service.process(result, trace_id="trace-12345", session_id="sess-99")

        assert response.metadata["trace_id"] == "trace-12345"
        assert response.metadata["session_id"] == "sess-99"


# ==============================================================================
# Error Handling and Invariant Validation Tests
# ==============================================================================

class TestPostProcessingErrorHandling:
    """Verify validation and contract invariant enforcement."""

    def test_none_input_raises_validation_error(self) -> None:
        service = PostProcessingService()
        with pytest.raises(PostProcessingValidationError) as exc_info:
            service.process(None)
        assert "Generation result cannot be None" in str(exc_info.value)

    def test_unsupported_type_raises_validation_error(self) -> None:
        service = PostProcessingService()
        with pytest.raises(PostProcessingValidationError) as exc_info:
            service.process(12345)
        assert "Unsupported generation result type 'int'" in str(exc_info.value)

    def test_missing_content_raises_validation_error(self) -> None:
        service = PostProcessingService()
        result = {"content": None, "finish_reason": "stop"}
        with pytest.raises(PostProcessingValidationError) as exc_info:
            service.process(result)
        assert "Generated response content is missing or None" in str(exc_info.value)

    def test_empty_content_disallowed_by_default(self) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(content="   \n\t   ")
        with pytest.raises(PostProcessingValidationError) as exc_info:
            service.process(result)
        assert "Generated response content is empty or contains only whitespace" in str(exc_info.value)

    def test_empty_content_allowed_when_configured(self) -> None:
        config = PostProcessingConfig(allow_empty_content=True)
        service = PostProcessingService(config=config)
        result = make_sample_llm_result(content="")

        response = service.process(result)
        assert response.content == ""

    def test_none_content_allowed_when_configured(self) -> None:
        config = PostProcessingConfig(allow_empty_content=True)
        service = PostProcessingService(config=config)
        result = {"content": None}

        response = service.process(result)
        assert response.content == ""

    def test_non_string_content_raises_validation_error(self) -> None:
        service = PostProcessingService()
        result = {"content": [1, 2, 3]}
        with pytest.raises(PostProcessingValidationError) as exc_info:
            service.process(result)
        assert "Generated response content must be a string" in str(exc_info.value)


# ==============================================================================
# Service Lifecycle and Async Entrypoints Tests
# ==============================================================================

class TestPostProcessingServiceLifecycle:
    """Verify singleton retrieval, overrides, and async entrypoints."""

    def test_get_and_reset_post_processing_service(self) -> None:
        s1 = get_post_processing_service()
        s2 = get_post_processing_service()
        assert s1 is s2

        reset_post_processing_service()
        s3 = get_post_processing_service()
        assert s3 is not s1

    def test_get_service_with_custom_config(self) -> None:
        custom_config = PostProcessingConfig(strip_whitespace=False)
        s = get_post_processing_service(config=custom_config)
        assert s.config.strip_whitespace is False

    def test_functional_post_process(self) -> None:
        result = make_sample_llm_result(content="  Direct functional process  ")
        response = post_process(result)
        assert response.content == "Direct functional process"

    @pytest.mark.asyncio
    async def test_post_process_async_entrypoint(self) -> None:
        result = make_sample_llm_result(content="  Async functional process  ")
        response = await post_process_async(result)
        assert response.content == "Async functional process"

    @pytest.mark.asyncio
    async def test_service_process_async_method(self) -> None:
        service = PostProcessingService()
        result = make_sample_llm_result(content="  Async service process  ")
        response = await service.process_async(result)
        assert response.content == "Async service process"

    def test_base_post_processor_subclassing(self) -> None:
        class CustomProcessor(BasePostProcessor):
            def process(self, result: Any, **kwargs: Any) -> ProcessedResponse:
                return ProcessedResponse(content="Custom processed")

        proc = CustomProcessor()
        res = proc.process("raw")
        assert res.content == "Custom processed"
