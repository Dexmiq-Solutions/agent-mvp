"""Comprehensive test suite for the Groundedness and Safety evaluation quality gate stage."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional
import pytest

from app.core.config import Settings
from exceptions.generation import (
    EvaluationError,
    EvaluationOutputError,
    EvaluationProviderError,
    EvaluationTimeoutError,
    EvaluationValidationError,
    GenerationError,
    LLMProviderError,
    LLMTimeoutError,
    RegenerationExhaustedError,
)
from generation.evaluation.base import BaseEvaluator
from generation.evaluation.config import EvaluationConfig
from generation.evaluation.models import EvaluationRequest, EvaluationResult
from generation.evaluation.service import (
    DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION,
    EvaluationService,
    evaluate,
    evaluate_async,
    generate_with_evaluation_async,
    get_evaluation_service,
    reset_evaluation_service,
)
from generation.formatting.models import FormattedContext, FormattedContextItem
from generation.llm.interface import BaseLLMInterface
from generation.llm.models import LLMResult, LLMStreamEvent, LLMUsage
from generation.postprocessing.models import ProcessedResponse
from generation.postprocessing.service import PostProcessingService
from generation.prompt.service import PromptConstructionService


# ==============================================================================
# Mock LLM Interface for Evaluator & Generation
# ==============================================================================

class MockLLMInterface(BaseLLMInterface):
    """Mock LLM adapter simulating provider responses, errors, and structured output."""

    def __init__(
        self,
        canned_response: Optional[str] = None,
        responses: Optional[list[str]] = None,
        exception_to_raise: Optional[Exception] = None,
        model: str = "mock-evaluator",
    ) -> None:
        self._canned_response = canned_response
        self._responses = list(responses) if responses is not None else None
        self._response_idx = 0
        self._exception_to_raise = exception_to_raise
        self._model = model
        self.call_count = 0
        self.recorded_prompts: list[Any] = []
        self.recorded_kwargs: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

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
        self.call_count += 1
        self.recorded_prompts.append(prompt)
        self.recorded_kwargs.append(kwargs)

        if self._exception_to_raise is not None:
            raise self._exception_to_raise

        if self._responses is not None:
            if self._response_idx < len(self._responses):
                content = self._responses[self._response_idx]
                self._response_idx += 1
            else:
                content = self._responses[-1]
        elif self._canned_response is not None:
            content = self._canned_response
        else:
            content = json.dumps({
                "grounded": True,
                "safe": True,
                "reason": "Default mock evaluation passed.",
            })

        return LLMResult(
            content=content,
            finish_reason="stop",
            usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130),
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
        res = await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout, **kwargs)
        yield LLMStreamEvent(delta=res.content, finish_reason="stop", usage=res.usage)


# ==============================================================================
# Fixtures & Helpers
# ==============================================================================

@pytest.fixture(autouse=True)
def reset_service() -> None:
    """Reset evaluation singleton state between tests."""
    reset_evaluation_service()
    yield
    reset_evaluation_service()


def make_sample_context(
    text: str = "Our platform supports Credit Cards and Net Banking.",
    project_id: str = "proj-123",
) -> FormattedContext:
    """Create a sample FormattedContext matching upstream pipeline output."""
    item = FormattedContextItem(
        index=1,
        text=text,
        content=text,
        chunk_id="chk-001",
        document_id="doc-001",
        metadata={"project_id": project_id},
    )
    return FormattedContext(
        text=text,
        items=(item,),
        item_count=1,
        project_id=project_id,
        strategy="text",
    )


def make_sample_response(
    content: str = "We support Credit Cards and Net Banking.",
    model: str = "gpt-4o",
) -> ProcessedResponse:
    """Create a sample ProcessedResponse matching post-processing output."""
    return ProcessedResponse(
        content=content,
        finish_reason="stop",
        usage=LLMUsage(input_tokens=50, output_tokens=15, total_tokens=65),
        model=model,
        metadata={"project_id": "proj-123"},
    )


# ==============================================================================
# Domain Models Tests
# ==============================================================================

class TestEvaluationModels:
    """Verify evaluation domain models, serialization, and quality gate decision."""

    def test_evaluation_result_passed_when_grounded_and_safe(self) -> None:
        res = EvaluationResult(
            grounded=True,
            safe=True,
            reason="Claims directly verified in context.",
            score=1.0,
            metadata={"latency_ms": 12.5},
        )
        assert res.passed is True
        assert res.is_acceptable is True
        assert res.grounded is True
        assert res.safe is True

        d = res.to_dict()
        assert d["passed"] is True
        assert d["grounded"] is True
        assert d["safe"] is True
        assert d["score"] == 1.0

    def test_evaluation_result_fails_when_ungrounded(self) -> None:
        res = EvaluationResult(
            grounded=False,
            safe=True,
            reason="Claims UPI support without context evidence.",
        )
        assert res.passed is False
        assert res.is_acceptable is False

    def test_evaluation_result_fails_when_unsafe(self) -> None:
        res = EvaluationResult(
            grounded=True,
            safe=False,
            reason="Exposes sensitive server instructions.",
        )
        assert res.passed is False
        assert res.is_acceptable is False

    def test_evaluation_result_fails_when_ungrounded_and_unsafe(self) -> None:
        res = EvaluationResult(
            grounded=False,
            safe=False,
            reason="Hallucinated harmful instructions.",
        )
        assert res.passed is False

    def test_evaluation_result_safe_repr(self) -> None:
        long_reason = "X" * 100
        res = EvaluationResult(grounded=True, safe=True, reason=long_reason)
        repr_str = repr(res)
        assert "EvaluationResult" in repr_str
        assert "grounded=True" in repr_str
        assert "safe=True" in repr_str
        assert "..." in repr_str

    def test_evaluation_request_model(self) -> None:
        req = EvaluationRequest(
            query="What are the supported payment methods?",
            context="Credit cards only.",
            response="Credit cards are supported.",
            project_id="proj-1",
        )
        d = req.to_dict()
        assert d["query"] == "What are the supported payment methods?"
        assert d["project_id"] == "proj-1"


# ==============================================================================
# Configuration Tests
# ==============================================================================

class TestEvaluationConfig:
    """Verify configuration resolution from settings and environment."""

    def test_default_config_values(self) -> None:
        config = EvaluationConfig()
        assert config.enabled is True
        assert config.timeout == 15.0
        assert config.temperature == 0.0
        assert config.max_retries == 1
        assert config.max_regeneration_attempts == 2
        assert config.strict_mode is True

    def test_from_settings(self) -> None:
        settings = Settings(
            EVALUATION_ENABLED=False,
            EVALUATION_MODEL="custom-eval-model",
            EVALUATION_TIMEOUT=20.0,
            EVALUATION_MAX_RETRIES=2,
            EVALUATION_TEMPERATURE=0.1,
            EVALUATION_MAX_REGENERATION_ATTEMPTS=3,
        )
        config = EvaluationConfig.from_settings(settings)
        assert config.enabled is False
        assert config.model == "custom-eval-model"
        assert config.timeout == 20.0
        assert config.max_retries == 2
        assert config.temperature == 0.1
        assert config.max_regeneration_attempts == 3

    def test_config_repr(self) -> None:
        config = EvaluationConfig(model="gpt-4o")
        repr_str = repr(config)
        assert "EvaluationConfig" in repr_str
        assert "model='gpt-4o'" in repr_str


# ==============================================================================
# Input Validation & Tenant Isolation Tests
# ==============================================================================

class TestEvaluationInputValidation:
    """Verify input validation, query preservation, and tenant isolation."""

    @pytest.mark.asyncio
    async def test_none_query_raises_validation_error(self) -> None:
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)
        with pytest.raises(EvaluationValidationError, match="Original user query cannot be None"):
            await service.evaluate_async(query=None, context="Context", response="Response")

    @pytest.mark.asyncio
    async def test_empty_query_raises_validation_error(self) -> None:
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)
        with pytest.raises(EvaluationValidationError, match="Original user query cannot be empty"):
            await service.evaluate_async(query="   ", context="Context", response="Response")

    @pytest.mark.asyncio
    async def test_none_context_raises_validation_error(self) -> None:
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)
        with pytest.raises(EvaluationValidationError, match="Context cannot be None"):
            await service.evaluate_async(query="Query", context=None, response="Response")

    @pytest.mark.asyncio
    async def test_none_response_raises_validation_error(self) -> None:
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)
        with pytest.raises(EvaluationValidationError, match="Generated response cannot be None"):
            await service.evaluate_async(query="Query", context="Context", response=None)

    @pytest.mark.asyncio
    async def test_duck_typed_query_extracted(self) -> None:
        class CustomQuery:
            original_query = "What payment methods are supported?"

        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)
        res = await service.evaluate_async(
            query=CustomQuery(),
            context="Credit cards",
            response="Credit cards",
        )
        assert res.passed is True
        assert "What payment methods are supported?" in mock_llm.recorded_prompts[0][1]["content"]

    @pytest.mark.asyncio
    async def test_project_boundary_match_succeeds(self) -> None:
        context = make_sample_context(project_id="proj-alpha")
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="Query",
            context=context,
            response="Response",
            project_id="proj-alpha",
        )
        assert res.passed is True
        assert res.metadata["project_id"] == "proj-alpha"

    @pytest.mark.asyncio
    async def test_project_boundary_mismatch_raises_validation_error(self) -> None:
        context = make_sample_context(project_id="proj-alpha")
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationValidationError, match="violates requested project boundary"):
            await service.evaluate_async(
                query="Query",
                context=context,
                response="Response",
                project_id="proj-beta",
            )


# ==============================================================================
# Semantic Evaluation Scenarios
# ==============================================================================

class TestSemanticEvaluationScenarios:
    """Verify semantic evaluation across groundedness, relevance, safety, and evidence dimensions."""

    @pytest.mark.asyncio
    async def test_grounded_and_safe_response(self) -> None:
        mock_output = json.dumps({
            "grounded": True,
            "safe": True,
            "reason": "All payment methods mentioned are documented in the project context.",
            "score": 1.0,
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="Which payment methods do you support?",
            context="We accept Credit Cards, Debit Cards, and Net Banking.",
            response="We accept Credit Cards, Debit Cards, and Net Banking.",
        )

        assert res.grounded is True
        assert res.safe is True
        assert res.passed is True
        assert "payment methods" in res.reason
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_ungrounded_response_with_unsupported_claims(self) -> None:
        mock_output = json.dumps({
            "grounded": False,
            "safe": True,
            "reason": "The response claims UPI support, but the supplied context does not provide evidence for UPI.",
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="Can I pay using UPI?",
            context="We accept Credit Cards and Net Banking.",
            response="Yes, UPI is fully supported across all transactions.",
        )

        assert res.grounded is False
        assert res.safe is True
        assert res.passed is False
        assert "UPI" in res.reason

    @pytest.mark.asyncio
    async def test_grounded_but_unsafe_response(self) -> None:
        mock_output = json.dumps({
            "grounded": True,
            "safe": False,
            "reason": "The response leaks sensitive internal credentials and administrative endpoints.",
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="How do I access the admin console?",
            context="Internal admin console credentials: user=admin pass=secret123.",
            response="Log in with user=admin pass=secret123.",
        )

        assert res.grounded is True
        assert res.safe is False
        assert res.passed is False
        assert "credentials" in res.reason

    @pytest.mark.asyncio
    async def test_ungrounded_and_unsafe_response(self) -> None:
        mock_output = json.dumps({
            "grounded": False,
            "safe": False,
            "reason": "Hallucinated instructions that suggest executing malicious commands.",
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="How to run tests?",
            context="Run pytest backend/tests.",
            response="Run rm -rf / and format hard drive.",
        )

        assert res.grounded is False
        assert res.safe is False
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_empty_response_fast_path(self) -> None:
        """Verify empty responses immediately fail groundedness without calling the LLM."""
        mock_llm = MockLLMInterface()
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="What is the refund policy?",
            context="Refunds within 30 days.",
            response="   ",
        )

        assert res.grounded is False
        assert res.safe is True
        assert res.passed is False
        assert "empty" in res.reason
        assert mock_llm.call_count == 0  # Fast path avoided unnecessary LLM call

    @pytest.mark.asyncio
    async def test_evaluation_disabled_by_config(self) -> None:
        """Verify evaluation is cleanly bypassed when disabled by configuration."""
        mock_llm = MockLLMInterface()
        config = EvaluationConfig(enabled=False)
        service = EvaluationService(config=config, llm=mock_llm)

        res = await service.evaluate_async(
            query="Query",
            context="Context",
            response="Response",
        )

        assert res.passed is True
        assert "bypassed" in res.reason.lower()
        assert mock_llm.call_count == 0

    @pytest.mark.asyncio
    async def test_evaluator_never_modifies_retrieval_or_generates_replacement(self) -> None:
        """Verify evaluator only evaluates and returns structured output without replacement."""
        mock_output = json.dumps({
            "grounded": True,
            "safe": True,
            "reason": "Accurate.",
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(
            query="Original user query",
            context="Retrieved context",
            response="Generated response",
        )

        assert isinstance(res, EvaluationResult)
        assert not hasattr(res, "replacement_response")
        assert not hasattr(res, "rewritten_response")


# ==============================================================================
# Structured Output Parsing & Code Fences
# ==============================================================================

class TestStructuredOutputParsing:
    """Verify robust extraction of JSON from markdown blocks, string booleans, and malformed inputs."""

    @pytest.mark.asyncio
    async def test_parses_json_in_markdown_code_fences(self) -> None:
        fenced_output = (
            "```json\n"
            "{\n"
            '  "grounded": true,\n'
            '  "safe": true,\n'
            '  "reason": "Response accurately reflects supplied documentation."\n'
            "}\n"
            "```"
        )
        mock_llm = MockLLMInterface(canned_response=fenced_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(query="Query", context="Context", response="Response")
        assert res.passed is True
        assert res.reason == "Response accurately reflects supplied documentation."

    @pytest.mark.asyncio
    async def test_parses_case_insensitive_string_booleans(self) -> None:
        string_bool_output = json.dumps({
            "grounded": "YES",
            "safe": "true",
            "reason": "Supported.",
        })
        mock_llm = MockLLMInterface(canned_response=string_bool_output)
        service = EvaluationService(llm=mock_llm)

        res = await service.evaluate_async(query="Query", context="Context", response="Response")
        assert res.passed is True
        assert res.grounded is True
        assert res.safe is True

    @pytest.mark.asyncio
    async def test_missing_grounded_field_raises_output_error(self) -> None:
        bad_json = json.dumps({"safe": True, "reason": "Missing grounded field."})
        mock_llm = MockLLMInterface(canned_response=bad_json)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationOutputError, match="missing required 'grounded' field"):
            await service.evaluate_async(query="Query", context="Context", response="Response")

    @pytest.mark.asyncio
    async def test_missing_safe_field_raises_output_error(self) -> None:
        bad_json = json.dumps({"grounded": True, "reason": "Missing safe field."})
        mock_llm = MockLLMInterface(canned_response=bad_json)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationOutputError, match="missing required 'safe' field"):
            await service.evaluate_async(query="Query", context="Context", response="Response")

    @pytest.mark.asyncio
    async def test_malformed_json_raises_output_error(self) -> None:
        raw_text = "This is not valid JSON at all."
        mock_llm = MockLLMInterface(canned_response=raw_text)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationOutputError, match="not valid JSON"):
            await service.evaluate_async(query="Query", context="Context", response="Response")

    @pytest.mark.asyncio
    async def test_empty_evaluator_response_raises_output_error(self) -> None:
        mock_llm = MockLLMInterface(canned_response="")
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationOutputError, match="empty response"):
            await service.evaluate_async(query="Query", context="Context", response="Response")


# ==============================================================================
# Provider & Timeout Error Handling
# ==============================================================================

class TestProviderErrorHandling:
    """Verify provider timeouts and errors are caught and translated without quality gate bypass."""

    @pytest.mark.asyncio
    async def test_timeout_translates_to_evaluation_timeout_error(self) -> None:
        timeout_exc = LLMTimeoutError("Request timed out", status_code=408)
        mock_llm = MockLLMInterface(exception_to_raise=timeout_exc)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationTimeoutError) as exc_info:
            await service.evaluate_async(query="Query", context="Context", response="Response")
        assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_provider_error_translates_to_evaluation_provider_error(self) -> None:
        provider_exc = LLMProviderError("Internal Server Error", status_code=500)
        mock_llm = MockLLMInterface(exception_to_raise=provider_exc)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationProviderError) as exc_info:
            await service.evaluate_async(query="Query", context="Context", response="Response")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_failure_never_produces_passing_gate(self) -> None:
        """Critical security test: provider failure must NEVER bypass the quality gate."""
        generic_exc = RuntimeError("Catastrophic connection crash")
        mock_llm = MockLLMInterface(exception_to_raise=generic_exc)
        service = EvaluationService(llm=mock_llm)

        with pytest.raises(EvaluationError):
            await service.evaluate_async(query="Query", context="Context", response="Response")


# ==============================================================================
# Prompt Construction Feedback Integration
# ==============================================================================

class TestPromptConstructionFeedbackIntegration:
    """Verify that feedback is propagated into prompt construction on regeneration."""

    def test_prompt_incorporates_evaluation_feedback(self) -> None:
        prompt_service = PromptConstructionService()
        feedback_text = "The response falsely claimed PayPal support."
        context = make_sample_context("Credit Cards and Debit Cards only.")

        prompt = prompt_service.construct(
            query="What payment methods are supported?",
            context=context,
            evaluation_feedback=feedback_text,
        )

        assert "PREVIOUS ATTEMPT FEEDBACK" in prompt.user_prompt
        assert feedback_text in prompt.user_prompt
        assert prompt.metadata["has_evaluation_feedback"] is True

    def test_prompt_without_feedback_remains_clean(self) -> None:
        prompt_service = PromptConstructionService()
        context = make_sample_context("Credit Cards only.")
        prompt = prompt_service.construct(
            query="What payment methods are supported?",
            context=context,
            evaluation_feedback=None,
        )

        assert "PREVIOUS ATTEMPT FEEDBACK" not in prompt.user_prompt
        assert prompt.metadata["has_evaluation_feedback"] is False


# ==============================================================================
# Bounded Regeneration & Pipeline Orchestration Tests
# ==============================================================================

class TestBoundedRegenerationOrchestration:
    """Verify bounded generation with quality gate evaluation and regeneration feedback."""

    @pytest.mark.asyncio
    async def test_generation_passes_on_first_attempt(self) -> None:
        # LLM generation returns good response; Evaluator returns passed
        gen_llm = MockLLMInterface(canned_response="We support Credit Cards.")
        eval_llm = MockLLMInterface(
            canned_response=json.dumps({
                "grounded": True,
                "safe": True,
                "reason": "Supported by context.",
            })
        )
        eval_service = EvaluationService(llm=eval_llm)
        context = make_sample_context("We support Credit Cards.")

        response = await generate_with_evaluation_async(
            query="What payment methods are supported?",
            context=context,
            llm_service=gen_llm,
            evaluation_service=eval_service,
        )

        assert isinstance(response, ProcessedResponse)
        assert response.content == "We support Credit Cards."
        assert response.metadata["evaluation_passed"] is True
        assert response.metadata["evaluation_attempts"] == 1
        assert gen_llm.call_count == 1
        assert eval_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_generation_passes_on_second_attempt_with_feedback(self) -> None:
        # Attempt 1: gen returns hallucinated answer, eval rejects
        # Attempt 2: gen returns corrected answer, eval passes
        gen_llm = MockLLMInterface(
            responses=[
                "We support Credit Cards and UPI.",
                "We support Credit Cards only.",
            ]
        )
        eval_llm = MockLLMInterface(
            responses=[
                json.dumps({
                    "grounded": False,
                    "safe": True,
                    "reason": "UPI is not mentioned in context.",
                }),
                json.dumps({
                    "grounded": True,
                    "safe": True,
                    "reason": "Claims match context.",
                }),
            ]
        )
        eval_service = EvaluationService(llm=eval_llm)
        context = make_sample_context("We support Credit Cards only.")

        response = await generate_with_evaluation_async(
            query="What payment methods are supported?",
            context=context,
            llm_service=gen_llm,
            evaluation_service=eval_service,
            max_attempts=3,
        )

        assert response.content == "We support Credit Cards only."
        assert response.metadata["evaluation_passed"] is True
        assert response.metadata["evaluation_attempts"] == 2
        assert gen_llm.call_count == 2
        assert eval_llm.call_count == 2

        # Verify feedback was included in the second generation prompt
        second_prompt_content = gen_llm.recorded_prompts[1].user_prompt
        assert "UPI is not mentioned in context" in second_prompt_content

    @pytest.mark.asyncio
    async def test_regeneration_exhaustion_raises_exception(self) -> None:
        """Verify bounded attempts fail gracefully on exhaustion without fabricating answers."""
        gen_llm = MockLLMInterface(canned_response="Persistent hallucination.")
        eval_llm = MockLLMInterface(
            canned_response=json.dumps({
                "grounded": False,
                "safe": True,
                "reason": "Persistent hallucination.",
            })
        )
        eval_service = EvaluationService(llm=eval_llm)
        context = make_sample_context("Official documentation context.")

        with pytest.raises(RegenerationExhaustedError) as exc_info:
            await generate_with_evaluation_async(
                query="Query",
                context=context,
                llm_service=gen_llm,
                evaluation_service=eval_service,
                max_attempts=3,
            )

        assert exc_info.value.attempts == 3
        assert exc_info.value.last_evaluation is not None
        assert exc_info.value.last_evaluation.passed is False
        assert gen_llm.call_count == 3
        assert eval_llm.call_count == 3


# ==============================================================================
# Service Singletons & Sync Helpers
# ==============================================================================

class TestServiceSingletonsAndSync:
    """Verify singleton management and synchronous evaluation helpers."""

    def test_get_and_reset_evaluation_service(self) -> None:
        s1 = get_evaluation_service()
        s2 = get_evaluation_service()
        assert s1 is s2

        reset_evaluation_service()
        s3 = get_evaluation_service()
        assert s3 is not s1

    def test_sync_evaluate_helper(self) -> None:
        mock_output = json.dumps({
            "grounded": True,
            "safe": True,
            "reason": "Sync evaluation passed.",
        })
        mock_llm = MockLLMInterface(canned_response=mock_output)
        service = EvaluationService(llm=mock_llm)

        res = evaluate(
            query="Query",
            context="Context",
            response="Response",
            service=service,
        )

        assert res.passed is True
        assert "Sync evaluation passed" in res.reason
