"""Post-processing service providing deterministic normalization of generation results."""

import json
import re
import time
from typing import Any, Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.generation import PostProcessingValidationError, StructuredOutputError
from generation.llm.models import LLMResult, LLMUsage
from generation.postprocessing.base import BasePostProcessor
from generation.postprocessing.config import PostProcessingConfig
from generation.postprocessing.models import ProcessedResponse

logger = get_logger(__name__)

_post_processing_service: Optional["PostProcessingService"] = None

# Mapping of known provider-specific finish reasons to standard normalized representations
_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "end_turn": "stop",
    "complete": "stop",
    "completed": "stop",
    "length": "length",
    "max_tokens": "length",
    "content_filter": "content_filter",
    "safety": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
}

# Regex matching markdown code blocks for JSON extraction
_JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)


class PostProcessingService(BasePostProcessor):
    """Application-level service orchestrating deterministic post-processing.

    Positions immediately following LLM Interface execution and immediately
    preceding Groundedness / Safety Checks in the RAG generation pipeline.

    Responsibilities:
    - Extract content from LLMResult, dicts, strings, or duck-typed agent results.
    - Validate content presence and invariants (non-empty unless configured).
    - Deterministically normalize line endings and whitespace.
    - Normalize provider finish reasons to standard application tokens.
    - Faithfully preserve token usage metrics and execution metadata.
    - Extract and parse structured output (JSON) when configured.
    - Record performance and observability metrics without leaking sensitive data.

    Non-Responsibilities:
    - Never calls an LLM or performs model inference.
    - Never performs semantic rewriting, quality correction, or summarization.
    - Never retrieves external data or queries databases.
    - Never performs groundedness or safety evaluation (delegated downstream).
    """

    def __init__(
        self,
        config: Optional[PostProcessingConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize PostProcessingService.

        Args:
            config: Optional PostProcessingConfig instance.
            settings: Optional application Settings instance.
        """
        self._settings = settings or get_settings()
        resolved_config = config or PostProcessingConfig.from_settings(self._settings)
        super().__init__(config=resolved_config)

    def _extract_inputs(
        self,
        result: Any,
    ) -> tuple[str | None, str | None, LLMUsage | None, str | None, dict[str, Any]]:
        """Extract raw components from supported input shapes.

        Supports:
        - LLMResult (canonical LLM Interface output)
        - Python dicts containing "content" or "text"
        - Raw strings
        - Duck-typed objects exposing "content", "generated_content", or "text"

        Returns:
            tuple of (raw_content, finish_reason, usage, model, metadata)

        Raises:
            PostProcessingValidationError: If input is None or unsupported type.
        """
        if result is None:
            raise PostProcessingValidationError("Generation result cannot be None.")

        if isinstance(result, LLMResult):
            raw_content = result.content
            finish_reason = result.finish_reason
            usage = result.usage
            metadata = dict(result.metadata)
            model = metadata.get("model") or metadata.get("model_name")
            return raw_content, finish_reason, usage, model, metadata

        if isinstance(result, str):
            return result, "stop", None, None, {}

        if isinstance(result, dict):
            raw_content = result.get("content")
            if raw_content is None:
                raw_content = result.get("text")
            finish_reason = result.get("finish_reason")
            usage_val = result.get("usage")
            usage: Optional[LLMUsage] = None
            if isinstance(usage_val, LLMUsage):
                usage = usage_val
            elif isinstance(usage_val, dict):
                usage = LLMUsage(
                    input_tokens=usage_val.get("input_tokens"),
                    output_tokens=usage_val.get("output_tokens"),
                    total_tokens=usage_val.get("total_tokens"),
                )
            metadata = dict(result.get("metadata") or {})
            model = result.get("model") or metadata.get("model") or metadata.get("model_name")
            return raw_content, finish_reason, usage, model, metadata

        # Duck-typed objects (e.g. Agent results, custom containers)
        content_attr = None
        for attr in ("content", "generated_content", "text"):
            if hasattr(result, attr):
                content_attr = getattr(result, attr)
                break

        if content_attr is not None or hasattr(result, "content"):
            raw_content = content_attr
            finish_reason = getattr(result, "finish_reason", None)
            usage_attr = getattr(result, "usage", None)
            usage = usage_attr if isinstance(usage_attr, LLMUsage) else None
            metadata_attr = getattr(result, "metadata", {}) or {}
            metadata = dict(metadata_attr) if isinstance(metadata_attr, dict) else {}
            model = getattr(result, "model", None) or metadata.get("model")
            return raw_content, finish_reason, usage, model, metadata

        raise PostProcessingValidationError(
            f"Unsupported generation result type '{type(result).__name__}'. "
            "Expected LLMResult, dict, str, or object with 'content' attribute."
        )

    def _normalize_content(
        self,
        raw_content: Any,
        config: PostProcessingConfig,
    ) -> str:
        """Validate and apply deterministic text normalization.

        Args:
            raw_content: Raw extracted content candidate.
            config: Active PostProcessingConfig.

        Returns:
            str: Deterministically normalized text content.

        Raises:
            PostProcessingValidationError: If content is missing, invalid type, or empty when disallowed.
        """
        if raw_content is None:
            if config.allow_empty_content:
                return ""
            raise PostProcessingValidationError("Generated response content is missing or None.")

        if not isinstance(raw_content, str):
            raise PostProcessingValidationError(
                f"Generated response content must be a string, got '{type(raw_content).__name__}'."
            )

        content = raw_content

        # Normalize line endings: CRLF (\r\n) or CR (\r) -> LF (\n)
        if config.normalize_line_endings:
            content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize whitespace (strip leading/trailing whitespace without altering internal indentation)
        if config.strip_whitespace:
            content = content.strip()

        # Validate non-empty constraint
        if not config.allow_empty_content and not content:
            raise PostProcessingValidationError(
                "Generated response content is empty or contains only whitespace."
            )

        # Optional maximum content length guard
        if config.max_content_length is not None and len(content) > config.max_content_length:
            content = content[: config.max_content_length]

        return content

    def _normalize_finish_reason(
        self,
        finish_reason: str | None,
        config: PostProcessingConfig,
        metadata: dict[str, Any],
    ) -> str | None:
        """Normalize finish reason to standard application tokens."""
        if finish_reason is None or not config.normalize_finish_reason:
            return finish_reason

        raw_str = finish_reason.strip().lower()
        normalized = _FINISH_REASON_MAP.get(raw_str, raw_str)
        if normalized != finish_reason:
            metadata["raw_finish_reason"] = finish_reason
        return normalized

    def _extract_and_parse_json(
        self,
        content: str,
        config: PostProcessingConfig,
        metadata: dict[str, Any],
    ) -> Any | None:
        """Extract and parse structured JSON output if configured."""
        if not (config.parse_json or config.require_json):
            return None

        target_text = content

        # Strip markdown code fences if configured
        if config.strip_code_fences_for_json:
            match = _JSON_CODE_BLOCK_PATTERN.search(target_text)
            if match:
                target_text = match.group(1).strip()

        try:
            parsed = json.loads(target_text)
            metadata["structured_output_parsed"] = True
            return parsed
        except (json.JSONDecodeError, TypeError) as err:
            metadata["structured_output_parsed"] = False
            metadata["json_parse_error"] = str(err)
            if config.require_json:
                raise StructuredOutputError(
                    f"Failed to parse structured JSON output: {err}",
                    original_error=err,
                ) from err
            logger.debug("Optional structured JSON parsing failed: %s", err)
            return None

    def process(
        self,
        result: Any,
        *,
        config: Optional[PostProcessingConfig] = None,
        parse_json: Optional[bool] = None,
        require_json: Optional[bool] = None,
        allow_empty_content: Optional[bool] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProcessedResponse:
        """Process, normalize, and validate generation results synchronously.

        Args:
            result: Raw or application-level generation result (LLMResult, dict, str, etc.).
            config: Optional PostProcessingConfig override.
            parse_json: Optional runtime override to parse structured JSON.
            require_json: Optional runtime override to strictly require valid JSON.
            allow_empty_content: Optional runtime override for empty content tolerance.
            model: Optional model identifier override.
            **kwargs: Additional metadata parameters to merge into the response.

        Returns:
            ProcessedResponse: Normalized application-level response.

        Raises:
            PostProcessingValidationError: If input or content violates invariants.
            StructuredOutputError: If structured output is required but parsing fails.
        """
        start_time = time.perf_counter()

        active_config = config or self._config
        if (
            parse_json is not None
            or require_json is not None
            or allow_empty_content is not None
        ):
            active_config = PostProcessingConfig(
                strip_whitespace=active_config.strip_whitespace,
                normalize_line_endings=active_config.normalize_line_endings,
                normalize_finish_reason=active_config.normalize_finish_reason,
                allow_empty_content=(
                    allow_empty_content
                    if allow_empty_content is not None
                    else active_config.allow_empty_content
                ),
                parse_json=(
                    parse_json if parse_json is not None else active_config.parse_json
                ),
                require_json=(
                    require_json if require_json is not None else active_config.require_json
                ),
                strip_code_fences_for_json=active_config.strip_code_fences_for_json,
                max_content_length=active_config.max_content_length,
            )

        # 1. Extract raw inputs
        raw_content, raw_finish_reason, usage, resolved_model, metadata = self._extract_inputs(
            result
        )

        # Merge additional kwargs into metadata
        if kwargs:
            metadata.update(kwargs)

        if model:
            resolved_model = model

        # 2. Normalize content
        normalized_content = self._normalize_content(raw_content, active_config)

        # 3. Normalize finish reason
        finish_reason = self._normalize_finish_reason(
            raw_finish_reason, active_config, metadata
        )

        # 4. Extract structured output if configured
        structured_output = self._extract_and_parse_json(
            normalized_content, active_config, metadata
        )

        # 5. Record execution duration and metrics
        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
        metadata["postprocessing_duration_ms"] = duration_ms
        metadata["raw_content_length"] = len(raw_content) if isinstance(raw_content, str) else 0
        metadata["processed_content_length"] = len(normalized_content)
        metadata["postprocessing_success"] = True

        logger.debug(
            "Post-processing completed in %.2fms (model=%s, finish_reason=%s, length=%d)",
            duration_ms,
            resolved_model,
            finish_reason,
            len(normalized_content),
        )

        return ProcessedResponse(
            content=normalized_content,
            finish_reason=finish_reason,
            usage=usage,
            model=resolved_model,
            structured_output=structured_output,
            raw_content=raw_content if isinstance(raw_content, str) else "",
            metadata=metadata,
        )

    async def process_async(
        self,
        result: Any,
        *,
        config: Optional[PostProcessingConfig] = None,
        parse_json: Optional[bool] = None,
        require_json: Optional[bool] = None,
        allow_empty_content: Optional[bool] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProcessedResponse:
        """Async convenience wrapper for process in asynchronous generation pipelines.

        Retains fast in-memory synchronous execution internally without artificial delay.
        """
        return self.process(
            result,
            config=config,
            parse_json=parse_json,
            require_json=require_json,
            allow_empty_content=allow_empty_content,
            model=model,
            **kwargs,
        )


def get_post_processing_service(
    config: Optional[PostProcessingConfig] = None,
    settings: Optional[Settings] = None,
) -> PostProcessingService:
    """Retrieve or initialize the singleton PostProcessingService instance.

    Args:
        config: Optional PostProcessingConfig override.
        settings: Optional Settings override.

    Returns:
        PostProcessingService: Configured service instance.
    """
    global _post_processing_service
    if _post_processing_service is None or config is not None or settings is not None:
        service = PostProcessingService(config=config, settings=settings)
        if config is None and settings is None:
            _post_processing_service = service
        return service
    return _post_processing_service


def reset_post_processing_service() -> None:
    """Reset the cached singleton PostProcessingService instance (useful for tests)."""
    global _post_processing_service
    _post_processing_service = None


def post_process(
    result: Any,
    *,
    config: Optional[PostProcessingConfig] = None,
    parse_json: Optional[bool] = None,
    require_json: Optional[bool] = None,
    allow_empty_content: Optional[bool] = None,
    model: Optional[str] = None,
    service: Optional[PostProcessingService] = None,
    **kwargs: Any,
) -> ProcessedResponse:
    """Functional convenience entrypoint to normalize generation results into ProcessedResponse.

    Args:
        result: Upstream generation result (LLMResult, dict, str, or compatible).
        config: Optional PostProcessingConfig override.
        parse_json: Optional runtime override to parse structured JSON.
        require_json: Optional runtime override to strictly require valid JSON.
        allow_empty_content: Optional runtime override for empty content tolerance.
        model: Optional model identifier override.
        service: Optional PostProcessingService override.
        **kwargs: Additional metadata parameters.

    Returns:
        ProcessedResponse: Normalized application-level response.
    """
    active_service = service or get_post_processing_service()
    return active_service.process(
        result,
        config=config,
        parse_json=parse_json,
        require_json=require_json,
        allow_empty_content=allow_empty_content,
        model=model,
        **kwargs,
    )


async def post_process_async(
    result: Any,
    *,
    config: Optional[PostProcessingConfig] = None,
    parse_json: Optional[bool] = None,
    require_json: Optional[bool] = None,
    allow_empty_content: Optional[bool] = None,
    model: Optional[str] = None,
    service: Optional[PostProcessingService] = None,
    **kwargs: Any,
) -> ProcessedResponse:
    """Async convenience wrapper for post_process in asynchronous RAG pipelines."""
    return post_process(
        result,
        config=config,
        parse_json=parse_json,
        require_json=require_json,
        allow_empty_content=allow_empty_content,
        model=model,
        service=service,
        **kwargs,
    )
