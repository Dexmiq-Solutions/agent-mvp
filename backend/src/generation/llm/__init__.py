"""LLM Generation package providing provider-independent model execution."""

from exceptions.generation import (
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
from generation.llm.adapters.openai import OpenAICompatibleLLMAdapter
from generation.llm.config import LLMConfig
from generation.llm.interface import BaseLLMInterface
from generation.llm.models import LLMResult, LLMStreamEvent, LLMUsage
from generation.llm.service import (
    LLMService,
    generate_async,
    generate_stream_async,
    get_llm_service,
    reset_llm_service,
)

__all__ = [
    # Domain Models
    "LLMResult",
    "LLMStreamEvent",
    "LLMUsage",
    # Configuration
    "LLMConfig",
    # Interfaces & Adapters
    "BaseLLMInterface",
    "OpenAICompatibleLLMAdapter",
    # Service & Execution
    "LLMService",
    "get_llm_service",
    "reset_llm_service",
    "generate_async",
    "generate_stream_async",
    # Exceptions
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
