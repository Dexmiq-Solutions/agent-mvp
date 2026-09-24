"""Context Formatting package for the RAG retrieval pipeline."""

from exceptions.retrieval import (
    ContextFormattingError,
    ContextFormattingValidationError,
)
from rag.retrieval.formatting.base import BaseContextFormatter
from rag.retrieval.formatting.config import (
    DEFAULT_CONTEXT_FOOTER,
    DEFAULT_CONTEXT_HEADER,
    DEFAULT_EMPTY_CONTEXT_TEXT,
    DEFAULT_ITEM_TEMPLATE,
    ContextFormattingConfig,
)
from rag.retrieval.formatting.models import FormattedContext, FormattedContextItem
from rag.retrieval.formatting.service import (
    ContextFormattingService,
    format_context,
    format_context_async,
    get_context_formatting_service,
    reset_context_formatting_service,
)
from rag.retrieval.formatting.text import TextContextFormatter

__all__ = [
    # Domain Models
    "FormattedContext",
    "FormattedContextItem",
    # Configuration
    "ContextFormattingConfig",
    "DEFAULT_CONTEXT_HEADER",
    "DEFAULT_CONTEXT_FOOTER",
    "DEFAULT_EMPTY_CONTEXT_TEXT",
    "DEFAULT_ITEM_TEMPLATE",
    # Formatter Strategies
    "BaseContextFormatter",
    "TextContextFormatter",
    # Service & Functional Entrypoints
    "ContextFormattingService",
    "get_context_formatting_service",
    "reset_context_formatting_service",
    "format_context",
    "format_context_async",
    # Domain Exceptions
    "ContextFormattingError",
    "ContextFormattingValidationError",
]
