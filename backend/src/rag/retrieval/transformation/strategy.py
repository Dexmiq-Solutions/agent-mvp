"""Transformation Strategy implementing LLM-based query rewriting."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from observability.logging import get_logger
from exceptions.retrieval import TransformationValidationError
from rag.retrieval.models import ProcessedQuery
from rag.retrieval.transformation.provider import BaseLLMClient
from rag.retrieval.transformation.validator import TransformationOutputValidator

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a retrieval query rewrite specialist for an enterprise RAG system.
Your sole task is to reformulate the user's current query into a standalone, search-optimized retrieval query.

CRITICAL INSTRUCTIONS:
1. DO NOT answer the user's question or explain anything. Output ONLY the rewritten retrieval query.
2. Resolve ambiguous pronouns (e.g., "that", "this", "it", "they") and incomplete follow-up phrases using the conversation context if provided.
3. Preserve all technical identifiers, requirement IDs (e.g., BRD-102), API names (e.g., API-v2), error codes (e.g., ERR-404), version numbers, and system names verbatim. Never generalize or drop them.
4. Do NOT invent facts, assumptions, or fabricated terminology.
5. If the user query is already clear and self-contained, return it as-is or minimally clarified.
6. Output strictly the single rewritten query string. No quotes, no markdown, no preamble."""


class BaseTransformationStrategy(ABC):
    """Abstract interface for query transformation strategies.

    Answers: 'How should this query be transformed?'
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy identifier."""

    @abstractmethod
    async def transform(
        self,
        query: ProcessedQuery,
        conversation_context: Optional[Any] = None,
    ) -> Optional[str]:
        """Transform a processed query into an alternative retrieval representation.

        Args:
            query: Preprocessed user query.
            conversation_context: Optional relevant conversation context.

        Returns:
            Optional[str]: Transformed retrieval query, or None if transformation should be skipped.

        Raises:
            TransformationValidationError: If generated query fails output validation.
            TransformationProviderError: If external LLM provider fails.
        """


class LLMQueryRewriteStrategy(BaseTransformationStrategy):
    """Query transformation strategy that reformulates queries using an LLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: Optional[str] = None,
        validator: Optional[TransformationOutputValidator] = None,
        temperature: float = 0.0,
        max_tokens: int = 250,
        timeout: Optional[float] = None,
    ) -> None:
        self._llm_client = llm_client
        self._model = model
        self._validator = validator or TransformationOutputValidator()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def strategy_name(self) -> str:
        return "llm_rewrite"

    def _format_context(self, conversation_context: Any) -> str:
        """Format conversation context compactly."""
        if not conversation_context:
            return ""

        if isinstance(conversation_context, str):
            return conversation_context.strip()

        if isinstance(conversation_context, (list, tuple)):
            turns: list[str] = []
            for item in conversation_context:
                if isinstance(item, str):
                    turns.append(item.strip())
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    turns.append(f"{role}: {content}")
                else:
                    turns.append(str(item))
            return "\n".join(turns)

        return str(conversation_context)

    def _build_user_prompt(self, query: ProcessedQuery, conversation_context: Any) -> str:
        """Construct compact user prompt for rewriting."""
        formatted_context = self._format_context(conversation_context)
        parts: list[str] = []

        if formatted_context:
            parts.append(f"Conversation Context:\n{formatted_context}\n")

        parts.append(f"Current User Query: {query.processed_query}")
        parts.append("Rewritten Retrieval Query:")
        return "\n".join(parts)

    async def transform(
        self,
        query: ProcessedQuery,
        conversation_context: Optional[Any] = None,
    ) -> Optional[str]:
        """Execute LLM query rewriting and validate output."""
        prompt = self._build_user_prompt(query, conversation_context)

        raw_output = await self._llm_client.complete(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )

        val_result = self._validator.validate(raw_output, query.original_query)
        if not val_result.is_valid:
            logger.warning(
                "Query transformation output rejected by validator: reason=%s, raw_preview=%r",
                val_result.reason,
                raw_output[:100],
            )
            raise TransformationValidationError(
                f"Transformed query failed validation: {val_result.reason}"
            )

        return val_result.cleaned_query
