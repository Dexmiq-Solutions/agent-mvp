"""Generation package encompassing prompt construction and RAG inference components."""

from exceptions.generation import (
    GenerationError,
    PromptConstructionError,
    PromptConstructionValidationError,
)
from generation.prompt import (
    DEFAULT_SYSTEM_INSTRUCTION,
    ConstructedPrompt,
    GenerationPrompt,
    PromptConstructionConfig,
    PromptConstructionService,
    StructuredPrompt,
    construct_prompt,
    construct_prompt_async,
    get_prompt_construction_service,
    reset_prompt_construction_service,
)

__all__ = [
    # Domain Models
    "ConstructedPrompt",
    "GenerationPrompt",
    "StructuredPrompt",
    # Configuration
    "PromptConstructionConfig",
    "DEFAULT_SYSTEM_INSTRUCTION",
    # Service & Functional Entrypoints
    "PromptConstructionService",
    "get_prompt_construction_service",
    "reset_prompt_construction_service",
    "construct_prompt",
    "construct_prompt_async",
    # Domain Exceptions
    "GenerationError",
    "PromptConstructionError",
    "PromptConstructionValidationError",
]
