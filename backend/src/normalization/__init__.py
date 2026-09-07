"""Document normalization layer for the RAG indexing pipeline.

Sits strictly between Cleaning and future Chunking.
Executes Structure-Aware Deterministic Normalization to make valid content consistently
represented without altering meaning, document structure, hierarchy, or intentional
formatting (code blocks, tables, lists).
"""

from exceptions.normalization import (
    InvalidNormalizationInputError,
    NormalizationConfigurationError,
    NormalizationError,
    NormalizationProcessingError,
)
from normalization.models import (
    NormalizationDecision,
    NormalizationReport,
    NormalizationRuleType,
    NormalizedDocument,
)
from normalization.rules import (
    BaseElementNormalizer,
    CodeBlockNormalizer,
    DefaultNormalizer,
    HeadingNormalizer,
    ListItemNormalizer,
    NormalizationConfig,
    ParagraphNormalizer,
    TableNormalizer,
    collapse_whitespace_in_line,
    normalize_blank_lines,
    normalize_line_endings,
    normalize_unicode_nfc,
    strip_safe_control_characters,
)
from normalization.service import (
    DocumentNormalizationService,
    get_normalization_service,
    reset_normalization_service,
)

__all__ = [
    # Service and Orchestrator
    "DocumentNormalizationService",
    "get_normalization_service",
    "reset_normalization_service",
    # Domain Models
    "NormalizedDocument",
    "NormalizationReport",
    "NormalizationDecision",
    "NormalizationRuleType",
    # Configuration and Rules
    "NormalizationConfig",
    "BaseElementNormalizer",
    "HeadingNormalizer",
    "ParagraphNormalizer",
    "CodeBlockNormalizer",
    "TableNormalizer",
    "ListItemNormalizer",
    "DefaultNormalizer",
    # Deterministic Functions
    "normalize_line_endings",
    "normalize_unicode_nfc",
    "strip_safe_control_characters",
    "collapse_whitespace_in_line",
    "normalize_blank_lines",
    # Domain Exceptions
    "NormalizationError",
    "InvalidNormalizationInputError",
    "NormalizationProcessingError",
    "NormalizationConfigurationError",
]
