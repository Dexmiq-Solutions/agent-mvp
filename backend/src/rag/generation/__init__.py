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
    PostProcessingError,
    PostProcessingValidationError,
    StructuredOutputError,
    EvaluationError,
    EvaluationValidationError,
    EvaluationProviderError,
    EvaluationTimeoutError,
    EvaluationOutputError,
    RegenerationExhaustedError,
)
from rag.generation.formatting import (
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
from rag.generation.prompt import (
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
from rag.generation.llm import (
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
from rag.generation.postprocessing import (
    BasePostProcessor,
    PostProcessedResponse,
    PostProcessingConfig,
    PostProcessingService,
    ProcessedResponse,
    get_post_processing_service,
    post_process,
    post_process_async,
    reset_post_processing_service,
)
from rag.generation.evaluation import (
    DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION,
    BaseEvaluator,
    EvaluationConfig,
    EvaluationRequest,
    EvaluationResult,
    EvaluationService,
    evaluate,
    evaluate_async,
    generate_with_evaluation_async,
    get_evaluation_service,
    reset_evaluation_service,
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
    # Post-processing Domain Models
    "ProcessedResponse",
    "PostProcessedResponse",
    # Post-processing Configuration
    "PostProcessingConfig",
    # Post-processing Interfaces & Implementations
    "BasePostProcessor",
    # Post-processing Service & Entrypoints
    "PostProcessingService",
    "get_post_processing_service",
    "reset_post_processing_service",
    "post_process",
    "post_process_async",
    # Groundedness & Safety Evaluation Domain Models
    "EvaluationResult",
    "EvaluationRequest",
    # Groundedness & Safety Evaluation Configuration
    "EvaluationConfig",
    "DEFAULT_EVALUATOR_SYSTEM_INSTRUCTION",
    # Groundedness & Safety Evaluation Interfaces & Implementations
    "BaseEvaluator",
    # Groundedness & Safety Evaluation Service & Entrypoints
    "EvaluationService",
    "get_evaluation_service",
    "reset_evaluation_service",
    "evaluate",
    "evaluate_async",
    "generate_with_evaluation_async",
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
    "PostProcessingError",
    "PostProcessingValidationError",
    "StructuredOutputError",
    "EvaluationError",
    "EvaluationValidationError",
    "EvaluationProviderError",
    "EvaluationTimeoutError",
    "EvaluationOutputError",
    "RegenerationExhaustedError",
]


