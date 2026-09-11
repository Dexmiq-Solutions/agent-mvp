"""Transformation Policy determining whether a retrieval query requires transformation."""

from abc import ABC, abstractmethod
import re
from typing import Any, Optional

from retrieval.models import PolicyDecision, ProcessedQuery


class BaseTransformationPolicy(ABC):
    """Abstract interface for transformation policies.

    Answers: 'Should this query be transformed?'
    Decoupled from transformation strategies and downstream retrieval.
    """

    @abstractmethod
    def evaluate(
        self,
        query: ProcessedQuery,
        conversation_context: Optional[Any] = None,
        is_fallback: bool = False,
    ) -> PolicyDecision:
        """Evaluate whether a processed query should be transformed.

        Args:
            query: Preprocessed user query.
            conversation_context: Optional conversational history/turns.
            is_fallback: True if re-entering from an insufficient retrieval fallback.

        Returns:
            PolicyDecision: High-confidence decision indicating whether to transform.
        """


class AdaptiveTransformationPolicy(BaseTransformationPolicy):
    """Adaptive, deterministic transformation policy.

    Uses simple, high-confidence signals to detect obvious context-dependent
    or conversational follow-up queries and retrieval fallbacks, while allowing
    self-contained queries to bypass LLM calls with zero overhead.
    """

    # Conversational continuation prefixes indicating context-dependence
    CONTINUATION_STARTERS: tuple[str, ...] = (
        "and ",
        "what about",
        "how about",
        "what if",
        "also ",
        "and the ",
        "and for ",
        "so what",
        "what did we decide",
        "how does that affect",
        "what happens if",
        "why did that",
        "why did it",
    )

    # Standalone deictic / anaphoric pronoun patterns referencing prior context
    DEICTIC_PATTERN: re.Pattern = re.compile(
        r"\b(that|this|these|those|it|its|they|them|the former|the latter)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        query: ProcessedQuery,
        conversation_context: Optional[Any] = None,
        is_fallback: bool = False,
    ) -> PolicyDecision:
        """Evaluate query context-dependence deterministically."""
        # 1. Fallback re-entry always warrants transformation
        if is_fallback:
            return PolicyDecision(
                should_transform=True,
                reason="retrieval_fallback",
                confidence=1.0,
            )

        text = query.processed_query.strip()
        lower_text = text.lower()

        # 2. Check for explicit conversational continuation starters
        for starter in self.CONTINUATION_STARTERS:
            if lower_text.startswith(starter):
                return PolicyDecision(
                    should_transform=True,
                    reason="conversational_continuation",
                    confidence=0.95,
                )

        # 3. Check for standalone deictic / anaphoric pronouns
        deictic_matches = self.DEICTIC_PATTERN.findall(lower_text)
        if deictic_matches:
            # If conversation context is present OR query is brief, highly likely context-dependent
            if conversation_context or len(text.split()) <= 8:
                return PolicyDecision(
                    should_transform=True,
                    reason="context_dependent_reference",
                    confidence=0.90,
                )

        # 4. Self-contained query: pass through to retrieval unchanged
        return PolicyDecision(
            should_transform=False,
            reason="self_contained",
            confidence=1.0,
        )
