"""LLM Execution Telemetry, Rate-Limit Handling, and Run-Level Metrics.

Provides:
- Lightweight request-level LLM execution tracking.
- Run-level aggregate metrics (requests, tokens, duration, 429s, retries, breakdowns).
- Extraction and structured logging of provider rate-limit (HTTP 429) response headers.
- Safe bounded exponential backoff retry wrapper for OpenAI 429 handling.
- LangChain BaseCallbackHandler integration for automatic token usage capture.
- ContextVar isolation per agent run to eliminate cross-request state leakage.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Callable, Coroutine, Optional, Sequence
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limit Information Extraction
# ---------------------------------------------------------------------------

@dataclass
class RateLimitInfo:
    """Structured rate limit details extracted from an HTTP 429 response."""

    is_rate_limit: bool = False
    status_code: Optional[int] = None
    error_type: str = ""
    error_message: str = ""
    retry_after: Optional[float] = None
    reset_requests: Optional[str] = None
    reset_tokens: Optional[str] = None
    remaining_requests: Optional[str] = None
    remaining_tokens: Optional[str] = None
    limit_requests: Optional[str] = None
    limit_tokens: Optional[str] = None
    provider_body: Optional[Any] = None

    @property
    def retry_after_seconds(self) -> Optional[float]:
        """Convenience property alias for retry_after."""
        return self.retry_after

    def to_dict(self) -> dict[str, Any]:
        """Convert rate limit info to a dictionary."""
        return {
            "is_rate_limit": self.is_rate_limit,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_after": self.retry_after,
            "reset_requests": self.reset_requests,
            "reset_tokens": self.reset_tokens,
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": self.remaining_tokens,
            "limit_requests": self.limit_requests,
            "limit_tokens": self.limit_tokens,
            "provider_body": self.provider_body,
        }

    def format_log_summary(self) -> str:
        """Format a concise single-line summary of rate limit details."""
        parts = [f"is_rate_limit={self.is_rate_limit}", f"status={self.status_code}"]
        if self.retry_after is not None:
            parts.append(f"retry_after={self.retry_after}s")
        if self.reset_requests is not None:
            parts.append(f"reset_requests={self.reset_requests}")
        if self.reset_tokens is not None:
            parts.append(f"reset_tokens={self.reset_tokens}")
        if self.remaining_tokens is not None:
            parts.append(f"remaining_tokens={self.remaining_tokens}")
        return ", ".join(parts)


def _parse_int_or_str(v: Any) -> Union[int, str]:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return str(v)


def extract_rate_limit_info(exc: Exception) -> RateLimitInfo:
    """Extract all available rate limit information from an OpenAI or HTTP exception.

    Supports:
    - openai.RateLimitError / openai.APIStatusError
    - httpx.HTTPStatusError
    - Generic exceptions with status_code or rate-limit message markers
    """
    status_code = getattr(exc, "status_code", None)
    err_str = str(exc)
    err_type = exc.__class__.__name__

    is_429 = (
        status_code == 429
        or "429" in err_str
        or "ratelimit" in err_type.lower()
        or "rate limit" in err_str.lower()
    )

    if not is_429:
        return RateLimitInfo(
            is_rate_limit=False,
            status_code=status_code,
            error_type=err_type,
            error_message=err_str,
        )

    info = RateLimitInfo(
        is_rate_limit=True,
        status_code=status_code or 429,
        error_type=err_type,
        error_message=err_str,
    )

    # Check for response object
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        # Parse headers case-insensitively
        for k, v in headers.items():
            k_lower = str(k).lower()
            if k_lower == "retry-after":
                try:
                    info.retry_after = float(v)
                except (ValueError, TypeError):
                    pass
            elif k_lower == "retry-after-ms":
                try:
                    info.retry_after = float(v) / 1000.0
                except (ValueError, TypeError):
                    pass
            elif k_lower == "x-ratelimit-reset-requests":
                info.reset_requests = str(v)
            elif k_lower == "x-ratelimit-reset-tokens":
                info.reset_tokens = str(v)
            elif k_lower == "x-ratelimit-remaining-requests":
                info.remaining_requests = _parse_int_or_str(v)
            elif k_lower == "x-ratelimit-remaining-tokens":
                info.remaining_tokens = _parse_int_or_str(v)
            elif k_lower == "x-ratelimit-limit-requests":
                info.limit_requests = _parse_int_or_str(v)
            elif k_lower == "x-ratelimit-limit-tokens":
                info.limit_tokens = _parse_int_or_str(v)

    body = getattr(exc, "body", None)
    if body is not None:
        info.provider_body = body

    return info


# ---------------------------------------------------------------------------
# Request Records & Aggregated Metrics
# ---------------------------------------------------------------------------

@dataclass
class LLMRequestRecord:
    """Telemetry record for an individual LLM request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    component: str = "Unknown"
    workflow_phase: str = "Unknown"
    phase: str = ""
    operation: str = "llm_call"
    section: Optional[str] = None
    model: str = "unknown"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    status_code: Optional[int] = None
    is_rate_limit: bool = False
    is_retry: bool = False
    retry_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_available: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    rate_limit_info: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.phase and self.workflow_phase == "Unknown":
            self.workflow_phase = self.phase
        elif self.workflow_phase != "Unknown" and not self.phase:
            self.phase = self.workflow_phase

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "component": self.component,
            "workflow_phase": self.workflow_phase,
            "operation": self.operation,
            "section": self.section,
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 3),
            "success": self.success,
            "status_code": self.status_code,
            "is_rate_limit": self.is_rate_limit,
            "is_retry": self.is_retry,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_available": self.tokens_available,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class LLMExecutionSummary:
    """Run-level aggregated metrics for all LLM calls in a BRD run."""

    run_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retry_requests: int = 0
    total_retries: int = 0
    rate_limit_errors: int = 0
    other_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_available_requests: int = 0
    tokens_missing_requests: int = 0
    total_llm_time_seconds: float = 0.0
    average_duration_seconds: float = 0.0
    breakdown_by_component: dict[str, int] = field(default_factory=dict)
    breakdown_by_phase: dict[str, int] = field(default_factory=dict)
    breakdown_by_operation: dict[str, int] = field(default_factory=dict)
    breakdown_by_section: dict[str, int] = field(default_factory=dict)
    failure_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Format clean structured event payload for frontend consumption."""
        return {
            "run_id": self.run_id,
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "retries": self.retry_requests,
                "total_retries": self.total_retries,
                "rate_limited": self.rate_limit_errors,
                "other_errors": self.other_errors,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "tokens_available_requests": self.tokens_available_requests,
                "tokens_missing_requests": self.tokens_missing_requests,
            },
            "timing": {
                "total_llm_time_seconds": round(self.total_llm_time_seconds, 2),
                "average_duration_seconds": round(self.average_duration_seconds, 2),
            },
            "breakdown": {
                "by_component": dict(self.breakdown_by_component),
                "by_phase": dict(self.breakdown_by_phase),
                "by_operation": dict(self.breakdown_by_operation),
                "by_section": dict(self.breakdown_by_section),
            },
            "breakdowns": {
                "by_component": dict(self.breakdown_by_component),
                "by_phase": dict(self.breakdown_by_phase),
                "by_operation": dict(self.breakdown_by_operation),
                "by_section": dict(self.breakdown_by_section),
            },
            "failures": dict(self.failure_summary),
        }

    def format_log_summary(self) -> str:
        """Format structured plain text summary for logging."""
        lines = [
            "==================================================",
            "BRD LLM EXECUTION SUMMARY",
            f"run_id={self.run_id}",
            f"total_requests={self.total_requests}",
            f"successful_requests={self.successful_requests}",
            f"failed_requests={self.failed_requests}",
            f"retry_requests={self.retry_requests}",
            f"total_retries={self.total_retries}",
            f"rate_limit_errors={self.rate_limit_errors}",
            f"input_tokens={self.input_tokens}",
            f"output_tokens={self.output_tokens}",
            f"total_tokens={self.total_tokens}",
            f"total_llm_time_seconds={round(self.total_llm_time_seconds, 2)}",
            "==================================================",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telemetry Tracker & Context Management
# ---------------------------------------------------------------------------

_current_telemetry_tracker: contextvars.ContextVar[Optional[LLMTelemetryTracker]] = (
    contextvars.ContextVar("current_telemetry_tracker", default=None)
)

_current_telemetry_context: contextvars.ContextVar[dict[str, Any]] = (
    contextvars.ContextVar("current_telemetry_context", default={})
)


def get_current_telemetry_tracker() -> Optional[LLMTelemetryTracker]:
    """Retrieve the telemetry tracker active for the current async execution context."""
    return _current_telemetry_tracker.get()


def set_current_telemetry_tracker(tracker: Optional[LLMTelemetryTracker]) -> contextvars.Token:
    """Set the telemetry tracker active for the current async execution context."""
    return _current_telemetry_tracker.set(tracker)


def reset_current_telemetry_tracker(token: contextvars.Token) -> None:
    """Reset the telemetry tracker active for the current async execution context."""
    _current_telemetry_tracker.reset(token)


def get_current_telemetry_context() -> dict[str, Any]:
    """Retrieve the current telemetry contextual tags."""
    return dict(_current_telemetry_context.get())


class scoped_telemetry_context:
    """Context manager (both sync and async) to scope component, phase, section, and operation tags to LLM calls."""

    def __init__(
        self,
        component: Optional[str] = None,
        phase: Optional[str] = None,
        section: Optional[str] = None,
        operation: Optional[str] = None,
        is_retry: bool = False,
        retry_count: int = 0,
    ) -> None:
        self.component = component
        self.phase = phase
        self.section = section
        self.operation = operation
        self.is_retry = is_retry
        self.retry_count = retry_count
        self._token: Optional[Any] = None

    def __enter__(self) -> dict[str, Any]:
        prior = dict(_current_telemetry_context.get())
        updated = dict(prior)
        if self.component is not None:
            updated["component"] = self.component
        if self.phase is not None:
            updated["phase"] = self.phase
        if self.section is not None:
            updated["section"] = self.section
        if self.operation is not None:
            updated["operation"] = self.operation
        if self.is_retry:
            updated["is_retry"] = True
        if self.retry_count > 0:
            updated["retry_count"] = self.retry_count

        self._token = _current_telemetry_context.set(updated)
        return updated

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _current_telemetry_context.reset(self._token)
            self._token = None

    async def __aenter__(self) -> dict[str, Any]:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class LLMTelemetryTracker:
    """Collector and aggregator for all LLM calls within a single BRD agent run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._records: list[LLMRequestRecord] = []
        self._active_calls: dict[str, dict[str, Any]] = {}

    @property
    def records(self) -> list[LLMRequestRecord]:
        """Return shallow copy of recorded requests."""
        return list(self._records)

    def record_request(self, record: LLMRequestRecord) -> None:
        """Record an LLM request execution."""
        self._records.append(record)

    def get_summary(self) -> LLMExecutionSummary:
        """Compute aggregate metrics across all recorded LLM calls."""
        summary = LLMExecutionSummary(run_id=self.run_id)
        summary.total_requests = len(self._records)

        total_duration = 0.0
        for r in self._records:
            total_duration += r.duration_seconds
            if r.success:
                summary.successful_requests += 1
            else:
                summary.failed_requests += 1
                if r.is_rate_limit:
                    summary.rate_limit_errors += 1
                else:
                    summary.other_errors += 1

                err_label = r.error_type or "Error"
                if r.is_rate_limit:
                    err_label = "429 Too Many Requests"
                summary.failure_summary[err_label] = summary.failure_summary.get(err_label, 0) + 1

            if r.is_retry:
                summary.retry_requests += 1
            if r.retry_count > 0:
                summary.total_retries += r.retry_count

            # Token accounting: ONLY accumulate when explicitly available
            if r.tokens_available and r.total_tokens is not None:
                summary.tokens_available_requests += 1
                summary.input_tokens += r.input_tokens or 0
                summary.output_tokens += r.output_tokens or 0
                summary.total_tokens += r.total_tokens
            else:
                summary.tokens_missing_requests += 1

            # Breakdowns
            comp = r.component or "unknown"
            summary.breakdown_by_component[comp] = summary.breakdown_by_component.get(comp, 0) + 1

            ph = r.workflow_phase or "unknown"
            summary.breakdown_by_phase[ph] = summary.breakdown_by_phase.get(ph, 0) + 1

            op = r.operation or "unknown"
            summary.breakdown_by_operation[op] = summary.breakdown_by_operation.get(op, 0) + 1

            if r.section:
                summary.breakdown_by_section[r.section] = (
                    summary.breakdown_by_section.get(r.section, 0) + 1
                )

        summary.total_llm_time_seconds = total_duration
        if summary.total_requests > 0:
            summary.average_duration_seconds = total_duration / summary.total_requests

        return summary

    def format_log_summary(self) -> str:
        """Format structured plain text summary for logging."""
        return self.get_summary().format_log_summary()


# ---------------------------------------------------------------------------
# LangChain Callback Handler
# ---------------------------------------------------------------------------

class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler automatically capturing LLM calls, durations, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._runs: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record start of LLM invocation."""
        rid_str = str(run_id)
        current_ctx = get_current_telemetry_context()

        # Extract model name
        model_name = "unknown"
        if serialized:
            kwargs_dict = serialized.get("kwargs", {})
            model_name = kwargs_dict.get("model_name") or kwargs_dict.get("model") or model_name
        if metadata and "model" in metadata:
            model_name = metadata["model"]

        self._runs[rid_str] = {
            "start_time": time.perf_counter(),
            "model": model_name,
            "context": current_ctx,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Record completion of LLM invocation and extract provider token usage."""
        rid_str = str(run_id)
        run_info = self._runs.pop(rid_str, None)
        start_time = run_info["start_time"] if run_info else time.perf_counter()
        model_name = run_info["model"] if run_info else "unknown"
        ctx = run_info["context"] if run_info else get_current_telemetry_context()

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Extract token usage from provider response
        in_tokens: Optional[int] = None
        out_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        tokens_avail = False

        if response and response.llm_output:
            usage = response.llm_output.get("token_usage")
            if isinstance(usage, dict):
                in_tokens = usage.get("prompt_tokens")
                out_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                if total_tokens is not None:
                    tokens_avail = True
            if not model_name or model_name == "unknown":
                model_name = response.llm_output.get("model_name", model_name)

        if not tokens_avail and response and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        usage_meta = getattr(msg, "usage_metadata", None)
                        if isinstance(usage_meta, dict):
                            in_tokens = usage_meta.get("input_tokens")
                            out_tokens = usage_meta.get("output_tokens")
                            total_tokens = usage_meta.get("total_tokens")
                            if total_tokens is not None:
                                tokens_avail = True
                                break
                        resp_meta = getattr(msg, "response_metadata", {})
                        if isinstance(resp_meta, dict):
                            token_usage = resp_meta.get("token_usage")
                            if isinstance(token_usage, dict):
                                in_tokens = token_usage.get("prompt_tokens")
                                out_tokens = token_usage.get("completion_tokens")
                                total_tokens = token_usage.get("total_tokens")
                                if total_tokens is not None:
                                    tokens_avail = True
                                    break
                if tokens_avail:
                    break

        record = LLMRequestRecord(
            request_id=rid_str,
            component=ctx.get("component", "unknown"),
            workflow_phase=ctx.get("phase", "unknown"),
            operation=ctx.get("operation", "unknown"),
            section=ctx.get("section"),
            model=model_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=True,
            status_code=200,
            is_rate_limit=False,
            is_retry=ctx.get("is_retry", False),
            retry_count=ctx.get("retry_count", 0),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            tokens_available=tokens_avail,
        )

        tracker = get_current_telemetry_tracker()
        if tracker:
            tracker.record_request(record)

        logger.info(
            "LLM request completed [component=%s, phase=%s, section=%s, op=%s, model=%s]: duration=%.2fs, tokens=%s (in=%s, out=%s)",
            record.component,
            record.workflow_phase,
            record.section or "none",
            record.operation,
            record.model,
            duration,
            record.total_tokens if tokens_avail else "unavailable",
            record.input_tokens if tokens_avail else "n/a",
            record.output_tokens if tokens_avail else "n/a",
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Record failure of LLM invocation and check for rate limits."""
        rid_str = str(run_id)
        run_info = self._runs.pop(rid_str, None)
        start_time = run_info["start_time"] if run_info else time.perf_counter()
        model_name = run_info["model"] if run_info else "unknown"
        ctx = run_info["context"] if run_info else get_current_telemetry_context()

        end_time = time.perf_counter()
        duration = end_time - start_time

        rl_info = extract_rate_limit_info(error) if isinstance(error, Exception) else RateLimitInfo()

        record = LLMRequestRecord(
            request_id=rid_str,
            component=ctx.get("component", "unknown"),
            workflow_phase=ctx.get("phase", "unknown"),
            operation=ctx.get("operation", "unknown"),
            section=ctx.get("section"),
            model=model_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=False,
            status_code=rl_info.status_code,
            is_rate_limit=rl_info.is_rate_limit,
            is_retry=ctx.get("is_retry", False),
            retry_count=ctx.get("retry_count", 0),
            error_type=rl_info.error_type or error.__class__.__name__,
            error_message=rl_info.error_message or str(error),
            rate_limit_info=rl_info.to_dict() if rl_info.is_rate_limit else None,
        )

        tracker = get_current_telemetry_tracker()
        if tracker:
            tracker.record_request(record)

        if rl_info.is_rate_limit:
            retry_msg = (
                f"retry_after={rl_info.retry_after}s"
                if rl_info.retry_after is not None
                else "provider retry_after header unavailable"
            )
            reset_msg = (
                f"reset_tokens={rl_info.reset_tokens}"
                if rl_info.reset_tokens
                else "reset headers unavailable"
            )
            logger.warning(
                "OpenAI rate limit (HTTP 429) encountered [component=%s, phase=%s, section=%s, op=%s, model=%s]: duration=%.2fs, %s, %s",
                record.component,
                record.workflow_phase,
                record.section or "none",
                record.operation,
                record.model,
                duration,
                retry_msg,
                reset_msg,
            )
        else:
            logger.error(
                "LLM request failed [component=%s, phase=%s, section=%s, op=%s, model=%s]: duration=%.2fs, error_type=%s, error=%s",
                record.component,
                record.workflow_phase,
                record.section or "none",
                record.operation,
                record.model,
                duration,
                record.error_type,
                record.error_message,
            )


_global_telemetry_callback_handler = TelemetryCallbackHandler()


def get_telemetry_callback_handler() -> TelemetryCallbackHandler:
    """Return singleton callback handler instance for LangChain model attachment."""
    return _global_telemetry_callback_handler


# ---------------------------------------------------------------------------
# Safe Bounded 429 Retry Wrapper
# ---------------------------------------------------------------------------

async def execute_with_rate_limit_retry_async(
    coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    *,
    component: str = "unknown",
    phase: str = "unknown",
    section: Optional[str] = None,
    operation: str = "unknown",
    operation_name: Optional[str] = None,
    max_retries: int = 1,
    initial_backoff: float = 2.0,
    max_backoff: float = 12.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Execute an async coroutine with safe bounded exponential backoff on HTTP 429 rate limit errors.

    Respects provider Retry-After headers if available. Does not produce unbounded retry storms.
    """
    op_name = operation_name if operation_name is not None else operation
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as exc:
            rl_info = extract_rate_limit_info(exc)
            if not rl_info.is_rate_limit or attempt >= max_retries:
                if rl_info.is_rate_limit:
                    logger.error(
                        "Rate limit (HTTP 429) persisted after %d retries for %s [section=%s]. Aborting operation.",
                        attempt,
                        op_name,
                        section or "none",
                    )
                raise

            attempt += 1
            if rl_info.retry_after is not None and rl_info.retry_after > 0:
                backoff = min(rl_info.retry_after, max_backoff)
            else:
                backoff = min(initial_backoff * (backoff_factor ** (attempt - 1)), max_backoff)

            logger.warning(
                "Rate limit (HTTP 429) hit during %s [section=%s, attempt=%d/%d]. "
                "Backing off for %.2fs before retry (retry_after_header=%s, reset_tokens=%s)...",
                op_name,
                section or "none",
                attempt,
                max_retries,
                backoff,
                rl_info.retry_after,
                rl_info.reset_tokens or "n/a",
            )
            await asyncio.sleep(backoff)
