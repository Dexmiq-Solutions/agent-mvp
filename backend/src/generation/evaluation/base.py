"""Abstract base interface defining the contract for Groundedness and Safety evaluators."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from generation.evaluation.config import EvaluationConfig
from generation.evaluation.models import EvaluationResult


class BaseEvaluator(ABC):
    """Abstract base class for all Groundedness and Safety evaluators.

    Follows the Strategy Pattern to decouple quality gate validation from
    concrete LLM providers, embedding models, or future Agent frameworks
    (such as LangGraph or DeepAgents).
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        """Initialize evaluator with configuration.

        Args:
            config: Optional EvaluationConfig instance.
        """
        self._config = config or EvaluationConfig()

    @property
    def config(self) -> EvaluationConfig:
        """Return active evaluation configuration."""
        return self._config

    @abstractmethod
    def evaluate(
        self,
        query: Any,
        context: Any,
        response: Any,
        *,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Synchronously evaluate generated response against supplied query and context.

        Args:
            query: Original user query as string or query container.
            context: Final context as FormattedContext, AssembledContext, items, or text.
            response: Generated response as ProcessedResponse, LLMResult, dict, or text.
            project_id: Optional project/tenant identifier to enforce isolation.
            **kwargs: Additional evaluation options.

        Returns:
            EvaluationResult: Structured quality gate decision (grounded, safe, reason, passed).
        """

    @abstractmethod
    async def evaluate_async(
        self,
        query: Any,
        context: Any,
        response: Any,
        *,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Asynchronously evaluate generated response against supplied query and context.

        Args:
            query: Original user query as string or query container.
            context: Final context as FormattedContext, AssembledContext, items, or text.
            response: Generated response as ProcessedResponse, LLMResult, dict, or text.
            project_id: Optional project/tenant identifier to enforce isolation.
            **kwargs: Additional evaluation options.

        Returns:
            EvaluationResult: Structured quality gate decision (grounded, safe, reason, passed).
        """
