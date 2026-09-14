"""Generation package encompassing context formatting, prompt construction, and RAG inference components."""

from exceptions.generation import (
    ContextFormattingError,
    ContextFormattingValidationError,
    GenerationError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMValidationError,
    PromptConstructionError,
    PromptConstructionValidationError,
)
from generation.formatting import (
    DEFAULT_CONTEXT_FOOTER,
    DEFAULT_CONTEXT_HEADER,
    DEFAULT_EMPTY_CONTEXT_TEXT,
    DEFAULT_ITEM_TEMPLATE,
    BaseContextFormatter,
    ContextFormattingConfig,
    ContextFormattingService,
    FormattedContext,
    FormattedContextItem,
    TextContextFormatter,
    format_context,
    format_context_async,
    get_context_formatting_service,
    reset_context_formatting_service,
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
from generation.llm import (
    BaseLLMInterface,
    LLMConfig,
    LLMResult,
    LLMService,
    LLMStreamEvent,
    LLMUsage,
    OpenAICompatibleLLMAdapter,
    generate_async,
    generate_stream_async,
    get_llm_service,
    reset_llm_service,
)

__all__ = [
    # Context Formatting Domain Models
    "FormattedContext",
    "FormattedContextItem",
    # Context Formatting Configuration
    "ContextFormattingConfig",
    "DEFAULT_CONTEXT_HEADER",
    "DEFAULT_CONTEXT_FOOTER",
    "DEFAULT_EMPTY_CONTEXT_TEXT",
    "DEFAULT_ITEM_TEMPLATE",
    # Context Formatters
    "BaseContextFormatter",
    "TextContextFormatter",
    # Context Formatting Service & Entrypoints
    "ContextFormattingService",
    "get_context_formatting_service",
    "reset_context_formatting_service",
    "format_context",
    "format_context_async",
    # Prompt Construction Domain Models
    "ConstructedPrompt",
    "GenerationPrompt",
    "StructuredPrompt",
    # Prompt Construction Configuration
    "PromptConstructionConfig",
    "DEFAULT_SYSTEM_INSTRUCTION",
    # Prompt Construction Service & Entrypoints
    "PromptConstructionService",
    "get_prompt_construction_service",
    "reset_prompt_construction_service",
    "construct_prompt",
    "construct_prompt_async",
    # LLM Interface Domain Models
    "LLMResult",
    "LLMStreamEvent",
    "LLMUsage",
    # LLM Interface Configuration
    "LLMConfig",
    # LLM Interfaces & Adapters
    "BaseLLMInterface",
    "OpenAICompatibleLLMAdapter",
    # LLM Service & Entrypoints
    "LLMService",
    "get_llm_service",
    "reset_llm_service",
    "generate_async",
    "generate_stream_async",
    # Domain Exceptions
    "GenerationError",
    "ContextFormattingError",
    "ContextFormattingValidationError",
    "PromptConstructionError",
    "PromptConstructionValidationError",
    "LLMError",
    "LLMConfigurationError",
    "LLMValidationError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMStreamError",
]


