"""Generation and prompt construction domain exceptions."""


class GenerationError(Exception):
    """Base exception for all generation pipeline errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class PromptConstructionError(GenerationError):
    """Base exception for prompt construction stage failures."""


class PromptConstructionValidationError(PromptConstructionError):
    """Raised when prompt construction inputs (query, context) fail validation."""
