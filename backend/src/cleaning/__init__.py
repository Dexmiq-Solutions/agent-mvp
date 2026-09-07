"""Document cleaning layer for the RAG indexing pipeline.

Sits strictly between Parsing & Extraction and future Normalization/Chunking.
Executes Structure-Aware Hybrid Cleaning combining deterministic rules for
high-confidence artifacts with structure-aware heuristics for pattern-based noise,
governed by a conservative preservation policy.
"""

from cleaning.context import DocumentStructureContext
from cleaning.heuristics import (
    BaseHeuristicEvaluator,
    ExtractionDuplicationHeuristic,
    NavigationBoilerplateHeuristic,
    RepetitiveHeaderFooterHeuristic,
)
from cleaning.models import (
    CleanedDocument,
    CleaningAction,
    CleaningCategory,
    CleaningDecision,
    CleaningReport,
    DecisionSource,
)
from cleaning.rules import (
    BaseDeterministicRule,
    EmptyContentRule,
    ExplicitPageNumberRule,
    InvalidControlCharsRule,
    MalformedStructuralArtifactRule,
)
from cleaning.service import (
    DocumentCleaningService,
    get_cleaning_service,
    reset_cleaning_service,
)
from exceptions.cleaning import (
    CleaningConfigurationError,
    CleaningError,
    CleaningProcessingError,
    InvalidCleaningInputError,
)

__all__ = [
    # Service and Orchestrator
    "DocumentCleaningService",
    "get_cleaning_service",
    "reset_cleaning_service",
    # Domain Models
    "CleanedDocument",
    "CleaningReport",
    "CleaningDecision",
    "CleaningAction",
    "CleaningCategory",
    "DecisionSource",
    # Context
    "DocumentStructureContext",
    # Deterministic Rules
    "BaseDeterministicRule",
    "EmptyContentRule",
    "InvalidControlCharsRule",
    "MalformedStructuralArtifactRule",
    "ExplicitPageNumberRule",
    # Heuristic Evaluators
    "BaseHeuristicEvaluator",
    "RepetitiveHeaderFooterHeuristic",
    "NavigationBoilerplateHeuristic",
    "ExtractionDuplicationHeuristic",
    # Domain Exceptions
    "CleaningError",
    "InvalidCleaningInputError",
    "CleaningProcessingError",
    "CleaningConfigurationError",
]
