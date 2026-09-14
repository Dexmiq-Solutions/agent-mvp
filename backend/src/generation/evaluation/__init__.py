"""Groundedness and Safety evaluation package acting as a quality gate in RAG generation."""

from exceptions.generation import (
    EvaluationError,
    EvaluationOutputError,
    EvaluationProviderError,
    EvaluationTimeoutError,
    EvaluationValidationError,
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

__all__ = [
    # Domain Models
    "EvaluationResult",
    "EvaluationRequest",
    # Configuration
    "EvaluationConfig",
    "DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION",
    # Base Interface
    "BaseEvaluator",
    # Service & Functional Entrypoints
    "EvaluationService",
    "get_evaluation_service",
    "reset_evaluation_service",
    "evaluate",
    "evaluate_async",
    "generate_with_evaluation_async",
    # Exceptions
    "EvaluationError",
    "EvaluationValidationError",
    "EvaluationProviderError",
    "EvaluationTimeoutError",
    "EvaluationOutputError",
    "RegenerationExhaustedError",
]
