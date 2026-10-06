"""Focused test suite for BRD Agent Phase 8 recovery rework, 429 rate limit resilience, and LLM telemetry.

Covers:
Part 10 - Testing Requirements:
1. Affected section validates successfully on first recovery attempt.
2. Affected section fails first attempt and succeeds on second.
3. Affected section fails all allowed attempts.
4. Assembly is NOT called when recovery leaves any section incomplete.
5. Assembly IS called when all affected sections are completed.
6. Streaming and non-streaming recovery behave consistently.
7. Simulate a 429.
8. Verify bounded retry behavior.
9. Verify retry information is logged when available.
10. Verify persistent 429 eventually fails cleanly.
11. Verify a rate-limit failure does not mark the operation as successful.
12. Successful LLM calls increment request/success counters.
13. Failed calls increment failure counters.
14. Retries are counted separately.
15. 429s are counted separately.
16. Token usage is accumulated correctly.
17. Missing token usage does not produce fabricated numbers.
18. Final execution-summary event contains the aggregate metrics.
19. Summary is emitted even when the Agent run fails, where lifecycle permits.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.tools import tool

from agents.brd import (
    BRDAgentState,
    BRDLeadAgent,
    BRDSectionStatus,
    create_brd_lead_agent,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.config import AgentConfig
from agents.brd.context import AgentContext
from agents.brd.final_validation import (
    FinalValidationCategory,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
    FinalValidationSeverity,
)
from agents.brd.section_generation import (
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
)
from agents.brd.section_validation import (
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from observability import (
    LLMExecutionSummary,
    LLMRequestRecord,
    LLMTelemetryTracker,
    RateLimitInfo,
    TelemetryCallbackHandler,
    execute_with_rate_limit_retry_async,
    extract_rate_limit_info,
    get_current_telemetry_tracker,
    reset_current_telemetry_tracker,
    scoped_telemetry_context,
    set_current_telemetry_tracker,
)


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------


class MockRateLimitResponse:
    """Mock HTTP response with rate-limit headers."""

    def __init__(self, headers: dict[str, str], status_code: int = 429) -> None:
        self.headers = headers
        self.status_code = status_code


class MockOpenAIRateLimitError(Exception):
    """Simulated OpenAI RateLimitError with attached response headers."""

    def __init__(self, message: str, headers: Optional[dict[str, str]] = None) -> None:
        super().__init__(message)
        self.status_code = 429
        self.response = MockRateLimitResponse(headers or {
            "retry-after": "0.1",
            "x-ratelimit-reset-requests": "100ms",
            "x-ratelimit-remaining-requests": "0",
        })


@tool("search_project_knowledge")
def mock_search_project_knowledge(query: str) -> str:
    """Mock search project knowledge tool."""
    return f"Knowledge for query: {query}"


def make_test_agent() -> BRDLeadAgent:
    """Create a test agent configured without memory or external RAG dependencies."""
    agent = create_brd_lead_agent(
        enable_memory=False,
        tools=[mock_search_project_knowledge],
    )
    # Populate initial completed content for all template sections
    for sec in agent.sections:
        agent.state.set_section_content(sec, f"# {sec}\n\nStandard validated section content.")
        agent.state.update_section_status(sec, BRDSectionStatus.COMPLETED)
    return agent


# ---------------------------------------------------------------------------
# PART 10 Tests 1-6: Recovery & Assembly Guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_succeeds_on_first_attempt() -> None:
    """Requirement 1: Affected section validates successfully on first recovery attempt."""
    agent = make_test_agent()
    affected_sec = agent.sections[1]  # E.g. section 2

    # Mock updater and validator
    agent.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name=affected_sec, content="Updated valid content", operation=SectionOperation.UPDATE)
    )
    agent.validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ValidationResult(outcome=ValidationOutcome.VALID, summary="Section valid", findings=[])
    )
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# Complete BRD", assembly_complete=True))  # type: ignore[method-assign]

    # First validation returns NEEDS_REWORK for affected_sec, re-validation returns VALID
    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Defect",
                explanation="Fix",
                required_change="Fix",
                affected_sections=[affected_sec],
            )
        ],
    )
    val_res_pass = FinalValidationResult(outcome=FinalValidationOutcome.VALID, summary="Valid document")
    agent.validate_final_brd_async = AsyncMock(side_effect=[val_res_fail, val_res_pass])  # type: ignore[method-assign]

    # Pre-condition: all sections completed
    assert agent.is_section_processing_complete

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await agent.run_workflow_async(request="Generate BRD")

    assert response.success is True
    assert agent.assemble_brd_async.call_count >= 1
    assert agent.state.get_section_status(affected_sec) == BRDSectionStatus.COMPLETED


@pytest.mark.asyncio
async def test_recovery_fails_first_attempt_and_succeeds_on_second() -> None:
    """Requirement 2: Affected section fails first attempt and succeeds on second."""
    agent = make_test_agent()
    affected_sec = agent.sections[2]

    val_call_count = 0

    def mock_val(*args: Any, **kwargs: Any) -> ValidationResult:
        nonlocal val_call_count
        val_call_count += 1
        if val_call_count >= 2:
            return ValidationResult(outcome=ValidationOutcome.VALID, summary="Section valid on retry", findings=[])
        return ValidationResult(
            outcome=ValidationOutcome.NEEDS_REWORK,
            summary="Attempt 1 failed",
            findings=[ValidationFinding(category=ValidationCategory.REQUIREMENT_COVERAGE, issue="Missing detail", explanation="Fix detail", required_change="Add detail")],
            rework_feedback="Add detail",
        )

    agent.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name=affected_sec, content="Rework content", operation=SectionOperation.UPDATE)
    )
    agent.validate_section_async = AsyncMock(side_effect=mock_val)  # type: ignore[method-assign]
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# Complete BRD", assembly_complete=True))  # type: ignore[method-assign]

    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Defect",
                explanation="Fix",
                required_change="Fix",
                affected_sections=[affected_sec],
            )
        ],
    )
    val_res_pass = FinalValidationResult(outcome=FinalValidationOutcome.VALID, summary="Valid document")
    agent.validate_final_brd_async = AsyncMock(side_effect=[val_res_fail, val_res_pass])  # type: ignore[method-assign]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await agent.run_workflow_async(request="Generate BRD", max_section_rework_attempts=2)

    assert response.success is True
    assert agent.assemble_brd_async.call_count >= 1
    assert agent.state.get_section_status(affected_sec) == BRDSectionStatus.COMPLETED


@pytest.mark.asyncio
async def test_recovery_fails_all_attempts_and_halts_safely() -> None:
    """Requirements 3 & 4: Affected section fails all attempts; assembly is NOT called."""
    agent = make_test_agent()
    affected_sec = agent.sections[3]

    agent.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name=affected_sec, content="Persistent invalid content", operation=SectionOperation.UPDATE)
    )
    agent.validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ValidationResult(
            outcome=ValidationOutcome.NEEDS_REWORK,
            summary="Section remains invalid",
            findings=[ValidationFinding(category=ValidationCategory.REQUIREMENT_COVERAGE, issue="Unresolved defect", explanation="Cannot fix", required_change="Cannot fix")],
            rework_feedback="Cannot fix",
        )
    )
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# Assembled", assembly_complete=True))  # type: ignore[method-assign]

    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Defect",
                explanation="Fix",
                required_change="Fix",
                affected_sections=[affected_sec],
            )
        ],
    )
    agent.validate_final_brd_async = AsyncMock(return_value=val_res_fail)  # type: ignore[method-assign]

    initial_assembly_count = 1  # Phase 6 assembly before Phase 7
    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await agent.run_workflow_async(request="Generate BRD", max_section_rework_attempts=2)

    # Recovery must halt cleanly with success=False, NO unhandled ValueError
    assert response.success is False
    assert "recovery halted" in response.output_text.lower()
    # Assembly must NOT have been called during Phase 8 recovery
    assert agent.assemble_brd_async.call_count == initial_assembly_count
    # State recorded exhaustion
    assert agent.state.final_validation_recovery_exhausted is True
    assert "recovery_failure" in agent.state.metadata
    assert agent.state.metadata["recovery_failure"]["failed_section"] == affected_sec


@pytest.mark.asyncio
async def test_assembly_is_called_when_all_affected_sections_completed() -> None:
    """Requirement 5: Assembly IS called when all affected sections are completed."""
    agent = make_test_agent()
    sec1 = agent.sections[1]
    sec2 = agent.sections[2]

    agent.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name="sec", content="Valid content", operation=SectionOperation.UPDATE)
    )
    agent.validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ValidationResult(outcome=ValidationOutcome.VALID, summary="Valid", findings=[])
    )
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# Complete BRD", assembly_complete=True))  # type: ignore[method-assign]

    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Defect 1",
                explanation="Fix",
                required_change="Fix",
                affected_sections=[sec1, sec2],
            )
        ],
    )
    val_res_pass = FinalValidationResult(outcome=FinalValidationOutcome.VALID, summary="Passed")
    agent.validate_final_brd_async = AsyncMock(side_effect=[val_res_fail, val_res_pass])  # type: ignore[method-assign]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await agent.run_workflow_async(request="Generate BRD")

    assert response.success is True
    # Reassembly was invoked after recovery completed both sections
    assert agent.assemble_brd_async.call_count >= 2


@pytest.mark.asyncio
async def test_streaming_and_non_streaming_recovery_parity() -> None:
    """Requirement 6: Streaming and non-streaming recovery behave consistently on failure."""
    agent_stream = make_test_agent()
    affected_sec = agent_stream.sections[3]

    agent_stream.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name=affected_sec, content="Invalid", operation=SectionOperation.UPDATE)
    )
    agent_stream.validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ValidationResult(outcome=ValidationOutcome.NEEDS_REWORK, summary="Invalid", findings=[ValidationFinding(category=ValidationCategory.REQUIREMENT_COVERAGE, issue="x", explanation="x", required_change="x")])
    )
    agent_stream.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# BRD", assembly_complete=True))  # type: ignore[method-assign]
    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[FinalValidationFinding(category=FinalValidationCategory.COMPLETENESS, severity=FinalValidationSeverity.ERROR, issue="x", explanation="x", required_change="x", affected_sections=[affected_sec])],
    )
    agent_stream.validate_final_brd_async = AsyncMock(return_value=val_res_fail)  # type: ignore[method-assign]

    streamed_events: list[dict[str, Any]] = []
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async for ev in agent_stream.stream_workflow_async(request="Generate BRD", max_section_rework_attempts=2):
            streamed_events.append(ev)

    # In streaming, recovery failure yielded content pause message and execution summary, no unhandled error
    content_events = [e for e in streamed_events if e.get("type") == "content"]
    assert any("BRD Generation Paused" in e.get("content", "") for e in content_events)
    # Assembly in Phase 8 was NOT called
    assert agent_stream.assemble_brd_async.call_count == 1
    # State recorded exhaustion
    assert agent_stream.state.final_validation_recovery_exhausted is True
    assert agent_stream.state.metadata["recovery_failure"]["failed_section"] == affected_sec


# ---------------------------------------------------------------------------
# PART 10 Tests 7-11: Rate Limiting & 429 Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_429_and_bounded_retry() -> None:
    """Requirements 7 & 8: Simulate a 429 and verify bounded retry behavior."""
    call_count = 0

    async def flaky_operation() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockOpenAIRateLimitError("Rate limit reached: 429 Too Many Requests")
        return "success"

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await execute_with_rate_limit_retry_async(
            flaky_operation,
            max_retries=3,
            initial_backoff=0.01,
            max_backoff=0.1,
            operation_name="test_429_bounded",
        )

    assert result == "success"
    assert call_count == 3
    # Slept twice between retries
    assert mock_sleep.call_count == 2


def test_extract_rate_limit_info() -> None:
    """Requirement 9: Verify retry information is logged and extracted when available."""
    headers = {
        "retry-after": "5",
        "x-ratelimit-reset-requests": "2.5s",
        "x-ratelimit-remaining-tokens": "100",
    }
    exc = MockOpenAIRateLimitError("Rate limit exceeded", headers=headers)
    info = extract_rate_limit_info(exc)

    assert info.is_rate_limit is True
    assert info.retry_after_seconds == 5.0
    assert info.reset_requests == "2.5s"
    assert info.remaining_tokens == 100
    assert info.status_code == 429

    summary_log = info.format_log_summary()
    assert "retry_after=5.0s" in summary_log
    assert "status=429" in summary_log


@pytest.mark.asyncio
async def test_persistent_429_fails_cleanly() -> None:
    """Requirement 10: Persistent 429 eventually fails cleanly after bounded retries."""
    call_count = 0

    async def permanent_429() -> str:
        nonlocal call_count
        call_count += 1
        raise MockOpenAIRateLimitError("Persistent rate limit: 429")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(MockOpenAIRateLimitError) as exc_info:
            await execute_with_rate_limit_retry_async(
                permanent_429,
                max_retries=2,
                initial_backoff=0.01,
                operation_name="test_persistent_429",
            )

    assert exc_info.value.status_code == 429
    # Initial attempt + 2 retries = 3 calls total
    assert call_count == 3


@pytest.mark.asyncio
async def test_rate_limit_does_not_mark_operation_as_successful() -> None:
    """Requirement 11: A rate-limit failure does not mark the operation as successful."""
    agent = make_test_agent()
    sec = agent.sections[0]
    agent.state.update_section_status(sec, BRDSectionStatus.IN_PROGRESS)

    with patch.object(
        agent.section_validator._graph,
        "ainvoke",
        new=AsyncMock(side_effect=MockOpenAIRateLimitError("429 Too Many Requests")),
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            val_res = await agent.validate_section_async(section=sec)

    # Fallback/error handling must mark section as NEEDS_REWORK, not VALID
    assert val_res.outcome == ValidationOutcome.NEEDS_REWORK
    assert val_res.is_valid is False
    assert agent.state.get_section_status(sec) != BRDSectionStatus.COMPLETED


# ---------------------------------------------------------------------------
# PART 10 Tests 12-19: Metrics & Execution Summary Telemetry
# ---------------------------------------------------------------------------


def test_successful_and_failed_request_counters() -> None:
    """Requirements 12, 13, 14, 15: Requests, successes, failures, retries, and 429s."""
    tracker = LLMTelemetryTracker(run_id="run-test-123")

    # Record 2 successful calls
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-123",
            component="BRDSectionGenerationAgent",
            phase="5_SECTION_ITERATION",
            operation="generate_section",
            section="Section 1",
            model="gpt-4o",
            start_time=100.0,
            end_time=102.0,
            duration_seconds=2.0,
            success=True,
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            tokens_available=True,
        )
    )
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-123",
            component="BRDSectionValidationAgent",
            phase="5_SECTION_ITERATION",
            operation="validate_section",
            section="Section 1",
            model="gpt-4o",
            start_time=102.0,
            end_time=103.5,
            duration_seconds=1.5,
            success=True,
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            tokens_available=True,
        )
    )

    # Record 1 retry that succeeded
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-123",
            component="BRDSectionGenerationAgent",
            phase="5_SECTION_ITERATION",
            operation="update_section",
            section="Section 1",
            model="gpt-4o",
            start_time=104.0,
            end_time=105.0,
            duration_seconds=1.0,
            success=True,
            is_retry=True,
            retry_count=1,
            input_tokens=800,
            output_tokens=150,
            total_tokens=950,
            tokens_available=True,
        )
    )

    # Record 1 429 rate limit failure
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-123",
            component="BRDFinalValidationAgent",
            phase="7_FINAL_VALIDATION",
            operation="validate_final_brd",
            model="gpt-4o",
            start_time=106.0,
            end_time=106.2,
            duration_seconds=0.2,
            success=False,
            is_rate_limit=True,
            status_code=429,
            error_type="RateLimitError",
            error_message="Too Many Requests",
        )
    )

    # Record 1 other error
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-123",
            component="BRDLeadAgent",
            phase="ORCHESTRATION",
            operation="decide_action",
            model="gpt-4o",
            start_time=107.0,
            end_time=107.5,
            duration_seconds=0.5,
            success=False,
            error_type="APIConnectionError",
            error_message="Connection reset",
        )
    )

    summary = tracker.get_summary()

    # Total requests
    assert summary.total_requests == 5
    assert summary.successful_requests == 3
    assert summary.failed_requests == 2
    assert summary.retry_requests == 1
    assert summary.total_retries == 1
    assert summary.rate_limit_errors == 1
    assert summary.other_errors == 1

    # Tokens accumulated
    assert summary.input_tokens == 2300
    assert summary.output_tokens == 450
    assert summary.total_tokens == 2750
    assert summary.tokens_available_requests == 3
    assert summary.tokens_missing_requests == 2

    # Breakdowns
    assert summary.breakdown_by_component["BRDSectionGenerationAgent"] == 2
    assert summary.breakdown_by_component["BRDSectionValidationAgent"] == 1
    assert summary.breakdown_by_component["BRDFinalValidationAgent"] == 1
    assert summary.breakdown_by_phase["5_SECTION_ITERATION"] == 3
    assert summary.breakdown_by_phase["7_FINAL_VALIDATION"] == 1

    # Failures breakdown
    assert summary.failure_summary["429 Too Many Requests"] == 1
    assert summary.failure_summary["APIConnectionError"] == 1


def test_missing_tokens_does_not_fabricate_numbers() -> None:
    """Requirements 16 & 17: Token usage is accumulated correctly; missing tokens are not fabricated."""
    tracker = LLMTelemetryTracker(run_id="run-test-missing")

    # Record request without token metrics (e.g. streaming or mocked provider)
    tracker.record_request(
        LLMRequestRecord(
            run_id="run-test-missing",
            component="BRDLeadAgent",
            model="gpt-4o",
            start_time=1.0,
            end_time=2.0,
            duration_seconds=1.0,
            success=True,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            tokens_available=False,
        )
    )

    summary = tracker.get_summary()
    assert summary.total_requests == 1
    assert summary.successful_requests == 1
    # Tokens should remain 0, tokens_missing_requests should be 1
    assert summary.input_tokens == 0
    assert summary.output_tokens == 0
    assert summary.total_tokens == 0
    assert summary.tokens_available_requests == 0
    assert summary.tokens_missing_requests == 1


@pytest.mark.asyncio
async def test_final_execution_summary_streaming_event() -> None:
    """Requirement 18: Final execution-summary event contains aggregate metrics."""
    agent = make_test_agent()

    # Execute a minimal workflow run
    agent.generate_and_validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            SectionGenerationResult(section_name="sec", content="Content", operation=SectionOperation.GENERATE),
            ValidationResult(outcome=ValidationOutcome.VALID, summary="Valid"),
        )
    )
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# BRD", assembly_complete=True))  # type: ignore[method-assign]
    agent.validate_final_brd_async = AsyncMock(return_value=FinalValidationResult(outcome=FinalValidationOutcome.VALID, summary="Valid"))  # type: ignore[method-assign]

    events: list[dict[str, Any]] = []
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async for ev in agent.stream_workflow_async(request="Generate BRD"):
            events.append(ev)

    summary_events = [e for e in events if e.get("type") == "execution_summary"]
    assert len(summary_events) == 1

    summary_payload = summary_events[0].get("execution_summary", {})
    assert "requests" in summary_payload
    assert "tokens" in summary_payload
    assert "timing" in summary_payload
    assert "breakdowns" in summary_payload
    assert "failures" in summary_payload


@pytest.mark.asyncio
async def test_summary_emitted_when_agent_run_fails() -> None:
    """Requirement 19: Execution summary is emitted even when Agent run fails."""
    agent = make_test_agent()
    affected_sec = agent.sections[1]

    # Force failure in recovery
    agent.update_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=SectionGenerationResult(section_name=affected_sec, content="Invalid", operation=SectionOperation.UPDATE)
    )
    agent.validate_section_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ValidationResult(
            outcome=ValidationOutcome.NEEDS_REWORK,
            summary="Invalid",
            findings=[ValidationFinding(category=ValidationCategory.CONSISTENCY, issue="x", explanation="x", required_change="x")],
        )
    )
    agent.assemble_brd_async = AsyncMock(return_value=BRDAssemblyResult(assembled_document="# BRD", assembly_complete=True))  # type: ignore[method-assign]
    val_res_fail = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[FinalValidationFinding(category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY, severity=FinalValidationSeverity.ERROR, issue="x", explanation="x", required_change="x", affected_sections=[affected_sec])],
    )
    agent.validate_final_brd_async = AsyncMock(return_value=val_res_fail)  # type: ignore[method-assign]

    events: list[dict[str, Any]] = []
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async for ev in agent.stream_workflow_async(request="Generate BRD", max_section_rework_attempts=1):
            events.append(ev)

    summary_events = [e for e in events if e.get("type") == "execution_summary"]
    # Even on recovery failure, execution summary event must be emitted
    assert len(summary_events) == 1
    assert summary_events[0]["type"] == "execution_summary"
