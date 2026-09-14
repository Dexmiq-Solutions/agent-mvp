"""Relevance Check subsystem providing pluggable evaluation of retrieved context."""

from abc import ABC, abstractmethod
import json
import re
import time
from typing import Any, Optional

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.retrieval.models import AssembledContext
from rag.retrieval.transformation.provider import BaseLLMClient, OpenAICompatibleLLMClient

logger = get_logger(__name__)

_DEFAULT_RELEVANCE_SYSTEM_PROMPT = (
    "You are a strict retrieval relevance evaluator. "
    "Given a user query and a set of retrieved context excerpts, determine whether the context "
    "contains information relevant to answering or addressing the query.\n\n"
    "CRITICAL RULES:\n"
    "1. Do NOT answer the query.\n"
    "2. If the context contains relevant information (even partial or supporting evidence), relevant is true.\n"
    "3. If the context is completely irrelevant, off-topic, or empty, relevant is false.\n"
    "4. Return ONLY a valid JSON object with format:\n"
    '{"relevant": true|false, "reason": "<concise explanation>"}\n'
)


class BaseRelevanceChecker(ABC):
    """Abstract interface for evaluating the relevance of assembled retrieval context."""

    @abstractmethod
    async def check_relevance(
        self,
        query: str,
        context: AssembledContext,
    ) -> tuple[bool, str]:
        """Evaluate whether the assembled context is relevant to the query.

        Args:
            query: Verbatim or preprocessed user query.
            context: Assembled retrieval context to evaluate.

        Returns:
            tuple[bool, str]: (is_relevant, reason).
        """
        pass


class HeuristicRelevanceChecker(BaseRelevanceChecker):
    """Fast, deterministic relevance checker without LLM overhead.

    Evaluates context emptiness, score thresholds, and lexical/token overlap.
    """

    def __init__(
        self,
        min_score_threshold: Optional[float] = None,
        min_items: int = 1,
    ) -> None:
        """Initialize HeuristicRelevanceChecker.

        Args:
            min_score_threshold: Optional minimum relevance score threshold.
            min_items: Minimum number of items required in context.
        """
        self._min_score_threshold = min_score_threshold
        self._min_items = max(1, min_items)

    async def check_relevance(
        self,
        query: str,
        context: AssembledContext,
    ) -> tuple[bool, str]:
        """Evaluate context relevance heuristically."""
        if context.is_empty or len(context.items) < self._min_items:
            return False, "Context is empty or contains fewer items than minimum required."

        # Check score threshold if configured
        if self._min_score_threshold is not None:
            max_score = max((item.score for item in context.items), default=0.0)
            if max_score < self._min_score_threshold:
                return False, f"Maximum item score ({max_score:.4f}) below threshold ({self._min_score_threshold:.4f})."

        # Check non-empty content
        total_chars = sum(len(item.text.strip()) for item in context.items)
        if total_chars == 0:
            return False, "Retrieved context items contain only empty text."

        return True, "Context satisfies heuristic relevance criteria."


class LLMRelevanceChecker(BaseRelevanceChecker):
    """LLM-backed semantic relevance evaluator.

    Utilizes an LLM provider to assess semantic alignment between the query
    and assembled context excerpts. Gracefully falls back to heuristic check on failure.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        model: str = "gpt-4o",
        timeout: float = 5.0,
        settings: Optional[Settings] = None,
        fallback_checker: Optional[BaseRelevanceChecker] = None,
    ) -> None:
        """Initialize LLMRelevanceChecker.

        Args:
            llm_client: Optional BaseLLMClient instance.
            model: Model identifier for evaluation.
            timeout: Maximum timeout in seconds for evaluator call.
            settings: Application Settings instance.
            fallback_checker: Optional fallback checker if LLM evaluation fails.
        """
        self._settings = settings or get_settings()
        self._model = model
        self._timeout = timeout
        self._llm_client = llm_client or OpenAICompatibleLLMClient(
            api_key=getattr(self._settings, "OPENAI_API_KEY", None),
            base_url=getattr(self._settings, "LLM_BASE_URL", None),
            default_model=self._model,
            default_timeout=self._timeout,
        )
        self._fallback = fallback_checker or HeuristicRelevanceChecker()

    async def check_relevance(
        self,
        query: str,
        context: AssembledContext,
    ) -> tuple[bool, str]:
        """Evaluate context relevance via LLM classification."""
        if context.is_empty:
            return False, "Context is empty."

        # Build context excerpt prompt
        context_excerpts = []
        for idx, item in enumerate(context.items[:5], start=1):
            snippet = item.text.strip()[:300]
            context_excerpts.append(f"[{idx}] {snippet}")
        context_text = "\n".join(context_excerpts)

        user_prompt = f"Query: {query}\n\nRetrieved Context:\n{context_text}\n"

        start_time = time.perf_counter()
        try:
            raw_response = await self._llm_client.complete(
                prompt=user_prompt,
                model=self._model,
                temperature=0.0,
                system_instruction=_DEFAULT_RELEVANCE_SYSTEM_PROMPT,
                timeout=self._timeout,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Parse JSON response
            cleaned = raw_response.strip()
            # Strip markdown code fences if present
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            is_rel = bool(data.get("relevant", True))
            reason = str(data.get("reason", "LLM relevance evaluation completed"))
            logger.debug(
                "LLM relevance check result: relevant=%s, reason='%s' (%.2fms)",
                is_rel,
                reason,
                elapsed_ms,
            )
            return is_rel, reason
        except Exception as exc:
            logger.warning(
                "LLM relevance check failed or timed out: %s. Falling back to heuristic check.",
                exc,
            )
            return await self._fallback.check_relevance(query, context)


def get_relevance_checker(
    strategy: str = "heuristic",
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> BaseRelevanceChecker:
    """Factory creating configured BaseRelevanceChecker instance.

    Args:
        strategy: 'heuristic' or 'llm'.
        settings: Application settings.
        **kwargs: Additional kwargs forwarded to constructor.

    Returns:
        BaseRelevanceChecker: Configured checker instance.
    """
    resolved_settings = settings or get_settings()
    norm_strategy = (strategy or "heuristic").lower().strip()

    if norm_strategy == "llm":
        return LLMRelevanceChecker(settings=resolved_settings, **kwargs)
    return HeuristicRelevanceChecker(**kwargs)
