"""Prompt Construction package for the RAG generation pipeline.

Responsible for assembling original user queries and authoritative retrieval context
into structured, provider-independent prompts for downstream LLM/agent inference.
"""

from exceptions.generation import (
    GenerationError,
    PromptConstructionError,
    PromptConstructionValidationError,
)
from rag.generation.prompt.config import (
    DEFAULT_SYSTEM_INSTRUCTION,
    PromptConstructionConfig,
)
from rag.generation.prompt.models import (
    ConstructedPrompt,
    GenerationPrompt,
    StructuredPrompt,
)
from rag.generation.prompt.service import (
    PromptConstructionService,
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
    # Exceptions
    "GenerationError",
    "PromptConstructionError",
    "PromptConstructionValidationError",
]
