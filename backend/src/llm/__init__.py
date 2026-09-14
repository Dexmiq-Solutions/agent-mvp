"""LLM subsystem providing provider-independent model execution and adapter interfaces."""

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
from llm.adapters.openai import OpenAICompatibleLLMAdapter
from llm.config import LLMConfig
from llm.interface import BaseLLMInterface
from llm.models import LLMResult, LLMStreamEvent, LLMUsage
from llm.service import (
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
