"""Generation and prompt construction domain exceptions."""

# Re-export LLM exceptions for backwards compatibility
from exceptions.llm import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMValidationError,
)


class GenerationError(Exception):
    """Base exception for all generation pipeline errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class ContextFormattingError(GenerationError):
    """Base exception for context formatting stage failures."""


class ContextFormattingValidationError(ContextFormattingError):
    """Raised when context formatting inputs fail validation or contract invariants."""


class PromptConstructionError(GenerationError):
    """Base exception for prompt construction stage failures."""


class PromptConstructionValidationError(PromptConstructionError):
    """Raised when prompt construction inputs (query, context) fail validation."""


class PostProcessingError(GenerationError):
    """Base exception for all post-processing stage failures."""


class PostProcessingValidationError(PostProcessingError):
    """Raised when post-processing inputs (result, content) fail validation or contract invariants."""


class StructuredOutputError(PostProcessingError):
    """Raised when structured output parsing or schema validation fails."""


class EvaluationError(GenerationError):
    """Base exception for all Groundedness and Safety evaluation stage failures."""


class EvaluationValidationError(EvaluationError):
    """Raised when evaluation inputs (query, context, response) fail validation or contract invariants."""


class EvaluationProviderError(EvaluationError):
    """Raised when an upstream LLM provider fails during evaluation execution."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error=original_error)
        self.status_code = status_code


class EvaluationTimeoutError(EvaluationProviderError):
    """Raised when an evaluation LLM request exceeds the bounded timeout."""


class EvaluationOutputError(EvaluationError):
    """Raised when evaluator output cannot be parsed as valid JSON or lacks required fields."""


class RegenerationExhaustedError(GenerationError):
    """Raised when bounded regeneration attempts are exhausted without passing evaluation."""

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_evaluation: object | None = None,
        last_response: object | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error=original_error)
        self.attempts = attempts
        self.last_evaluation = last_evaluation
        self.last_response = last_response
