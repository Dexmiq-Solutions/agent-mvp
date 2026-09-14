"""Groundedness and Safety evaluation service acting as a generation quality gate."""

import asyncio
import concurrent.futures
import json
import re
import time
from typing import Any, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.generation import (
    EvaluationError,
    EvaluationOutputError,
    EvaluationProviderError,
    EvaluationTimeoutError,
    EvaluationValidationError,
    LLMError,
    LLMProviderError,
    LLMTimeoutError,
    RegenerationExhaustedError,
)
from rag.generation.evaluation.base import BaseEvaluator
from rag.generation.evaluation.config import EvaluationConfig
from rag.generation.evaluation.models import EvaluationResult
from rag.generation.formatting.models import FormattedContext
from rag.generation.llm.interface import BaseLLMInterface
from rag.generation.llm.models import LLMResult
from rag.generation.llm.service import get_llm_service
from rag.generation.postprocessing.base import BasePostProcessor
from rag.generation.postprocessing.models import ProcessedResponse
from rag.generation.postprocessing.service import get_post_processing_service
from rag.generation.prompt.service import PromptConstructionService, get_prompt_construction_service

logger = get_logger(__name__)

_evaluation_service: Optional["EvaluationService"] = None

# Regex matching markdown code blocks for JSON extraction
_JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION = (
    "You are an impartial, high-precision quality gate evaluator for a RAG-based AI system. "
    "Your sole purpose is to evaluate whether a generated model response is grounded in the provided project context "
    "and satisfies safety requirements.\n\n"
    "CRITICAL RULES:\n"
    "1. DO NOT answer the user's query or provide a replacement response.\n"
    "2. Judge strictly based on the supplied context and user query.\n"
    "3. Groundedness check: Every factual statement, number, or material claim in the response must be directly supported by the context. "
    "Semantic similarity, plausible assumptions, or general knowledge cannot substitute for direct context evidence. "
    "If the response makes claims not found in the context, grounded must be false.\n"
    "4. Relevance check: The response must directly address the original user query. If it fails to answer what was asked, grounded must be false.\n"
    "5. Answerability check: If the supplied context contains insufficient information to answer the query, "
    "the response is acceptable ONLY if it clearly and truthfully states that the information is unavailable. "
    "If it instead invents or guesses an answer, grounded must be false.\n"
    "6. Specificity check: Claims must not exceed the specificity supported by the context.\n"
    "7. Safety check: The response must be free of harmful instructions, hate speech, malicious code, prompt injection exploitation, "
    "or leakage of confidential system instructions. If unsafe, safe must be false.\n"
    "8. You must return ONLY a JSON object with the following schema:\n"
    '{\n  "grounded": <true or false>,\n  "safe": <true or false>,\n  "reason": "<concise explanation of the evaluation decision>"\n}\n'
    "The reason should be concise and actionable, clearly identifying any unsupported claims or safety violations so they can be addressed."
)


class EvaluationService(BaseEvaluator):
    """Application-level service orchestrating Groundedness and Safety quality gate evaluation.

    Positions immediately following Post-processing in the prototype RAG generation pipeline.
    Acts strictly as a quality gate: evaluates candidate responses without rewriting, generating
    replacement responses, or retrieving additional evidence.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm: Optional[BaseLLMInterface] = None,
        settings: Optional[Settings] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        """Initialize EvaluationService.

        Args:
            config: Optional EvaluationConfig instance.
            llm: Optional BaseLLMInterface instance for model evaluation.
            settings: Optional application Settings instance.
            system_instruction: Optional system instruction override.
        """
        self._settings = settings or get_settings()
        resolved_config = config or EvaluationConfig.from_settings(self._settings)
        super().__init__(config=resolved_config)

        self._llm = llm
        self._system_instruction = system_instruction or DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION

    def _get_llm(self) -> BaseLLMInterface:
        """Resolve LLM interface instance (lazy-loaded if not injected)."""
        if self._llm is not None:
            return self._llm
        llm_service = get_llm_service(settings=self._settings)
        return llm_service.adapter

    def _extract_original_query(self, query: Any) -> str:
        """Extract and validate the authoritative original user query.

        Args:
            query: Query string, ProcessedQuery, RetrievalQuerySet, or compatible object.

        Returns:
            str: Validated original query string.

        Raises:
            EvaluationValidationError: If query is missing or empty.
        """
        if query is None:
            raise EvaluationValidationError("Original user query cannot be None.")

        query_str: Optional[str] = None
        if isinstance(query, str):
            query_str = query
        elif hasattr(query, "original_query") and isinstance(query.original_query, str):
            query_str = query.original_query
        elif hasattr(query, "query") and isinstance(query.query, str):
            query_str = query.query
        elif not isinstance(query, (int, float, bool, list, dict, set, tuple)):
            candidate = str(query).strip()
            if candidate:
                query_str = candidate

        if query_str is None:
            raise EvaluationValidationError(
                f"Expected str or object with 'original_query', got '{type(query).__name__}'."
            )

        clean_query = query_str.strip()
        if not clean_query:
            raise EvaluationValidationError(
                "Original user query cannot be empty or contain only whitespace."
            )

        return clean_query

    def _extract_context(self, context: Any) -> tuple[str, Optional[str]]:
        """Extract formatted context text and project isolation identifier.

        Args:
            context: FormattedContext, AssembledContext, sequence of items, or string.

        Returns:
            tuple of (context_text, project_id)

        Raises:
            EvaluationValidationError: If context is None or invalid type.
        """
        if context is None:
            raise EvaluationValidationError("Context cannot be None.")

        if isinstance(context, str):
            return context.strip(), None

        if isinstance(context, FormattedContext):
            return context.text.strip(), context.project_id

        if hasattr(context, "text") and hasattr(context, "project_id"):
            return str(context.text).strip(), getattr(context, "project_id", None)

        if isinstance(context, (list, tuple)):
            items_text: list[str] = []
            extracted_project_id: Optional[str] = None
            for item in context:
                item_text = (
                    getattr(item, "text", None)
                    or getattr(item, "content", None)
                    or (item.get("text") if isinstance(item, dict) else None)
                    or (item.get("content") if isinstance(item, dict) else None)
                    or str(item)
                )
                items_text.append(str(item_text).strip())
                if extracted_project_id is None:
                    extracted_project_id = getattr(item, "project_id", None) or (
                        item.get("project_id") if isinstance(item, dict) else None
                    )
            return "\n\n".join(items_text), extracted_project_id

        if hasattr(context, "items") and isinstance(context.items, (list, tuple)):
            return self._extract_context(context.items)

        raise EvaluationValidationError(
            f"Unsupported context type '{type(context).__name__}'. "
            "Expected FormattedContext, AssembledContext, list of items, or str."
        )

    def _extract_response(self, response: Any) -> tuple[str, Optional[str], dict[str, Any]]:
        """Extract response content, model, and metadata from supported response shapes.

        Args:
            response: ProcessedResponse, LLMResult, dict, str, or duck-typed response.

        Returns:
            tuple of (content, model, metadata)

        Raises:
            EvaluationValidationError: If response is None or invalid type.
        """
        if response is None:
            raise EvaluationValidationError("Generated response cannot be None.")

        if isinstance(response, ProcessedResponse):
            return response.content, response.model, dict(response.metadata)

        if isinstance(response, LLMResult):
            model = response.metadata.get("model") if response.metadata else None
            return response.content, model, dict(response.metadata)

        if isinstance(response, dict):
            content = response.get("content")
            if content is None:
                content = response.get("text")
            if content is None:
                raise EvaluationValidationError("Response dict must contain 'content' or 'text'.")
            metadata = dict(response.get("metadata") or {})
            model = response.get("model") or metadata.get("model")
            return str(content), model, metadata

        if isinstance(response, str):
            return response, None, {}

        # Duck-typed response
        content_attr = None
        for attr in ("content", "generated_content", "text"):
            if hasattr(response, attr):
                content_attr = getattr(response, attr)
                break

        if content_attr is not None:
            metadata = dict(getattr(response, "metadata", {}) or {})
            model = getattr(response, "model", None) or metadata.get("model")
            return str(content_attr), model, metadata

        raise EvaluationValidationError(
            f"Unsupported response type '{type(response).__name__}'. "
            "Expected ProcessedResponse, LLMResult, dict, or str."
        )

    def _parse_structured_output(self, raw_content: str) -> tuple[bool, bool, str, Optional[float]]:
        """Parse and validate structured JSON output from evaluator LLM.

        Args:
            raw_content: Raw response text returned by evaluator model.

        Returns:
            tuple of (grounded, safe, reason, score)

        Raises:
            EvaluationOutputError: If JSON is invalid or required fields are missing.
        """
        stripped = (raw_content or "").strip()
        if not stripped:
            raise EvaluationOutputError("Evaluator returned empty response.")

        json_str = stripped
        match = _JSON_CODE_BLOCK_PATTERN.search(stripped)
        if match:
            json_str = match.group(1).strip()

        try:
            parsed = json.loads(json_str)
        except Exception as exc:
            logger.error("Evaluator output is not valid JSON: %r", stripped[:200])
            raise EvaluationOutputError(
                f"Evaluator output is not valid JSON: {exc}",
                original_error=exc,
            ) from exc

        if not isinstance(parsed, dict):
            raise EvaluationOutputError(
                f"Evaluator output must be a JSON dictionary, got '{type(parsed).__name__}'."
            )

        if "grounded" not in parsed:
            raise EvaluationOutputError("Evaluator output is missing required 'grounded' field.")

        if "safe" not in parsed:
            raise EvaluationOutputError("Evaluator output is missing required 'safe' field.")

        def _to_bool(val: Any, field_name: str) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in ("true", "yes", "1", "pass", "passed", "y"):
                    return True
                if lowered in ("false", "no", "0", "fail", "failed", "n"):
                    return False
            raise EvaluationOutputError(
                f"Evaluator field '{field_name}' must be boolean, got '{val!r}'."
            )

        grounded = _to_bool(parsed["grounded"], "grounded")
        safe = _to_bool(parsed["safe"], "safe")
        reason = str(parsed.get("reason", "")).strip()

        if not reason:
            if not (grounded and safe):
                raise EvaluationOutputError("Evaluator rejected response but provided no reason.")
            reason = "Response is supported by the supplied context and satisfies safety criteria."

        score_val = parsed.get("score")
        score: Optional[float] = None
        if score_val is not None:
            try:
                score = float(score_val)
            except (ValueError, TypeError):
                score = None

        return grounded, safe, reason, score

    async def evaluate_async(
        self,
        query: Any,
        context: Any,
        response: Any,
        *,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Asynchronously evaluate generated response against supplied query and context."""
        start_time = time.perf_counter()

        # 1. Validate and extract original user query
        clean_query = self._extract_original_query(query)

        # 2. Validate and extract context, verifying tenant isolation
        clean_context, ctx_project_id = self._extract_context(context)

        eff_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise EvaluationValidationError("project_id must be a non-empty string when specified.")
            eff_project_id = project_id.strip()

        if eff_project_id is not None and ctx_project_id is not None and eff_project_id != ctx_project_id:
            raise EvaluationValidationError(
                f"Context project_id '{ctx_project_id}' violates requested project boundary '{eff_project_id}'."
            )

        resolved_project_id = eff_project_id or ctx_project_id

        # 3. Validate and extract response content
        clean_response, resp_model, resp_metadata = self._extract_response(response)

        # Fast path: Empty response fails groundedness (unresponsive/no evidence delivered)
        if not clean_response.strip():
            logger.warning("Evaluator received empty response; rejecting quality gate.")
            return EvaluationResult(
                grounded=False,
                safe=True,
                reason="The generated response is empty and does not answer the user's query.",
                metadata={
                    "empty_response": True,
                    "duration_ms": 0.0,
                    "project_id": resolved_project_id,
                },
            )

        # Fast path: Evaluation disabled in config
        if not self._config.enabled:
            logger.info("Groundedness / Safety evaluation is disabled by configuration.")
            return EvaluationResult(
                grounded=True,
                safe=True,
                reason="Evaluation bypassed (disabled by configuration).",
                metadata={
                    "evaluation_enabled": False,
                    "duration_ms": 0.0,
                    "project_id": resolved_project_id,
                },
            )

        # 4. Construct Evaluator-Specific Prompt
        evaluator_user_prompt = (
            f"[ORIGINAL USER QUERY]\n{clean_query}\n\n"
            f"[RETRIEVED PROJECT CONTEXT]\n{clean_context or '[No context provided]'}\n\n"
            f"[GENERATED RESPONSE TO EVALUATE]\n{clean_response}"
        )

        messages = [
            {"role": "system", "content": self._system_instruction},
            {"role": "user", "content": evaluator_user_prompt},
        ]

        llm = self._get_llm()
        effective_model = self._config.model or llm.model_name

        # 5. Execute LLM Evaluation Call with Bounded Timeout & Structured Output
        try:
            llm_result = await llm.generate(
                messages,
                temperature=self._config.temperature,
                timeout=self._config.timeout,
                max_tokens=500,
                response_format={"type": "json_object"},
                **kwargs,
            )
        except LLMTimeoutError as exc:
            logger.error("Evaluator request timed out after %.2fs", self._config.timeout)
            raise EvaluationTimeoutError(
                f"Evaluation request timed out after {self._config.timeout:.2f}s.",
                status_code=408,
                original_error=exc,
            ) from exc
        except LLMProviderError as exc:
            logger.error("Evaluator provider failed: %s", exc)
            raise EvaluationProviderError(
                f"Evaluation provider error: {exc.message}",
                status_code=exc.status_code,
                original_error=exc,
            ) from exc
        except LLMError as exc:
            logger.error("Evaluator LLM error: %s", exc)
            raise EvaluationProviderError(
                f"Evaluation error: {exc.message}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during evaluation: %s", exc)
            raise EvaluationError(
                f"Unexpected evaluation error: {exc}",
                original_error=exc,
            ) from exc

        # 6. Parse and Validate Structured Output
        grounded, safe, reason, score = self._parse_structured_output(llm_result.content)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        model_used = llm_result.metadata.get("model") or effective_model

        logger.info(
            "Evaluation completed in %.2fms: grounded=%s, safe=%s, passed=%s, model='%s'",
            elapsed_ms,
            grounded,
            safe,
            grounded and safe,
            model_used,
        )

        return EvaluationResult(
            grounded=grounded,
            safe=safe,
            reason=reason,
            score=score,
            metadata={
                "latency_ms": round(elapsed_ms, 2),
                "model": model_used,
                "project_id": resolved_project_id,
                "evaluator": "EvaluationService",
                "tokens": llm_result.usage.to_dict() if llm_result.usage else None,
            },
        )

    def evaluate(
        self,
        query: Any,
        context: Any,
        response: Any,
        *,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Synchronously evaluate generated response against supplied query and context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.evaluate_async(query, context, response, project_id=project_id, **kwargs),
                )
                return future.result()

        return asyncio.run(
            self.evaluate_async(query, context, response, project_id=project_id, **kwargs)
        )


def get_evaluation_service(
    config: Optional[EvaluationConfig] = None,
    llm: Optional[BaseLLMInterface] = None,
    settings: Optional[Settings] = None,
) -> EvaluationService:
    """Retrieve or initialize the singleton EvaluationService instance."""
    global _evaluation_service
    if _evaluation_service is None or any(arg is not None for arg in (config, llm, settings)):
        service = EvaluationService(
            config=config,
            llm=llm,
            settings=settings,
        )
        if all(arg is None for arg in (config, llm, settings)):
            _evaluation_service = service
        return service
    return _evaluation_service


def reset_evaluation_service() -> None:
    """Reset the cached singleton EvaluationService instance (useful for test isolation)."""
    global _evaluation_service
    _evaluation_service = None


async def evaluate_async(
    query: Any,
    context: Any,
    response: Any,
    *,
    project_id: Optional[str] = None,
    service: Optional[EvaluationService] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Module-level convenience function for async Groundedness and Safety evaluation."""
    active_service = service or get_evaluation_service()
    return await active_service.evaluate_async(
        query=query,
        context=context,
        response=response,
        project_id=project_id,
        **kwargs,
    )


def evaluate(
    query: Any,
    context: Any,
    response: Any,
    *,
    project_id: Optional[str] = None,
    service: Optional[EvaluationService] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Module-level convenience function for sync Groundedness and Safety evaluation."""
    active_service = service or get_evaluation_service()
    return active_service.evaluate(
        query=query,
        context=context,
        response=response,
        project_id=project_id,
        **kwargs,
    )


async def generate_with_evaluation_async(
    query: Any,
    context: Any,
    *,
    project_id: Optional[str] = None,
    system_instruction: Optional[str] = None,
    max_attempts: Optional[int] = None,
    prompt_service: Optional[PromptConstructionService] = None,
    llm_service: Optional[BaseLLMInterface] = None,
    post_processing_service: Optional[BasePostProcessor] = None,
    evaluation_service: Optional[EvaluationService] = None,
    **llm_kwargs: Any,
) -> ProcessedResponse:
    """Execute bounded generation pipeline with quality gate evaluation and feedback regeneration.

    Flow:
    1. Construct prompt using original user query and retrieved context (incorporating evaluation
       feedback on regeneration attempts).
    2. Execute LLM generation via LLM Interface.
    3. Post-process and normalize generated output.
    4. Evaluate normalized output via Groundedness / Safety Check quality gate.
    5. If passed (grounded AND safe): accept and return final ProcessedResponse.
    6. If failed: propagate evaluator reason as feedback into next attempt.
    7. If maximum attempts exhausted: raise RegenerationExhaustedError without fabricating answers.

    Args:
        query: Original user query (untransformed).
        context: Authoritative retrieval context (AssembledContext, FormattedContext, or sequence).
        project_id: Optional project boundary identifier.
        system_instruction: Optional system instruction override.
        max_attempts: Maximum total generation attempts (defaults to 1 + max_regeneration_attempts).
        prompt_service: Optional PromptConstructionService override.
        llm_service: Optional BaseLLMInterface override.
        post_processing_service: Optional BasePostProcessor override.
        evaluation_service: Optional EvaluationService override.
        **llm_kwargs: Additional arguments passed to LLM generation.

    Returns:
        ProcessedResponse: Accepted, normalized response with evaluation metadata attached.

    Raises:
        RegenerationExhaustedError: If all allowed generation attempts fail evaluation.
        EvaluationError: If an unrecoverable evaluation error occurs.
    """
    p_service = prompt_service or get_prompt_construction_service()
    l_service = llm_service or get_llm_service()
    post_service = post_processing_service or get_post_processing_service()
    eval_service = evaluation_service or get_evaluation_service()

    configured_limit = 1 + max(0, eval_service.config.max_regeneration_attempts)
    total_attempts = max_attempts if max_attempts is not None and max_attempts > 0 else configured_limit

    evaluation_feedback: Optional[str] = None
    last_evaluation: Optional[EvaluationResult] = None
    last_processed: Optional[ProcessedResponse] = None

    for attempt in range(1, total_attempts + 1):
        # 1. Prompt Construction incorporating evaluation feedback if regenerating
        constructed_prompt = p_service.construct(
            query=query,
            context=context,
            system_instruction=system_instruction,
            project_id=project_id,
            evaluation_feedback=evaluation_feedback,
        )

        # 2. LLM Interface Execution
        llm_result = await l_service.generate(constructed_prompt, **llm_kwargs)

        # 3. Post-Processing Normalization
        processed = post_service.process(llm_result)
        last_processed = processed

        # 4. Groundedness & Safety Evaluation Quality Gate
        eval_result = await eval_service.evaluate_async(
            query=query,
            context=context,
            response=processed,
            project_id=project_id,
        )
        last_evaluation = eval_result

        # 5. Quality Gate Acceptance Decision
        if eval_result.passed:
            logger.info(
                "Generation passed quality gate on attempt %d/%d (grounded=%s, safe=%s)",
                attempt,
                total_attempts,
                eval_result.grounded,
                eval_result.safe,
            )
            enriched_metadata = dict(processed.metadata)
            enriched_metadata["evaluation"] = eval_result.to_dict()
            enriched_metadata["evaluation_passed"] = True
            enriched_metadata["evaluation_attempts"] = attempt

            return ProcessedResponse(
                content=processed.content,
                finish_reason=processed.finish_reason,
                usage=processed.usage,
                model=processed.model,
                structured_output=processed.structured_output,
                raw_content=processed.raw_content,
                metadata=enriched_metadata,
            )

        logger.warning(
            "Generation attempt %d/%d rejected by quality gate: %s",
            attempt,
            total_attempts,
            eval_result.reason,
        )

        # 6. Regeneration Feedback Setup
        if attempt < total_attempts:
            evaluation_feedback = eval_result.reason

    # 7. Exhaustion: All attempts failed quality gate
    raise RegenerationExhaustedError(
        f"Generation failed groundedness/safety evaluation after {total_attempts} attempts. "
        f"Last failure reason: {last_evaluation.reason if last_evaluation else 'Unknown'}",
        attempts=total_attempts,
        last_evaluation=last_evaluation,
        last_response=last_processed,
    )
