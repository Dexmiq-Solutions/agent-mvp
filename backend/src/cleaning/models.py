"""Domain models for the document cleaning layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from parsing.models import ElementType, ParsedDocument, ParsedElement


class CleaningCategory(str, Enum):
    """Categories of removable unwanted content and extraction artifacts."""

    EMPTY_CONTENT = "empty_content"
    EXTRACTION_ARTIFACT = "extraction_artifact"
    INVALID_CONTROL_CHARS = "invalid_control_chars"
    PAGE_NUMBER = "page_number"
    REPETITIVE_HEADER = "repetitive_header"
    REPETITIVE_FOOTER = "repetitive_footer"
    NAVIGATION_BOILERPLATE = "navigation_boilerplate"
    DUPLICATE = "duplicate"


class DecisionSource(str, Enum):
    """Source that determined the cleaning action for an element."""

    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"


class CleaningAction(str, Enum):
    """Action to take on a document element."""

    REMOVE = "remove"
    PRESERVE = "preserve"


@dataclass(frozen=True)
class CleaningDecision:
    """Detailed record of a cleaning decision made for an individual element.

    Provides complete explainability and traceability for auditing why an element
    was removed or preserved.
    """

    element_id: str
    action: CleaningAction
    decision_source: DecisionSource
    rule_name: str
    reason: str
    confidence: float
    element_type: ElementType
    category: CleaningCategory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize cleaning decision to a dictionary."""
        return {
            "element_id": self.element_id,
            "action": self.action.value,
            "category": self.category.value if self.category else None,
            "decision_source": self.decision_source.value,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "element_type": self.element_type.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CleaningReport:
    """Summary report detailing the results of the cleaning stage.

    Contains metrics, removal counts, category breakdowns, and individual decisions
    without persisting or leaking raw document text.
    """

    total_input_elements: int = 0
    total_cleaned_elements: int = 0
    removed_count: int = 0
    preserved_count: int = 0
    decisions: list[CleaningDecision] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def removed_elements(self) -> list[CleaningDecision]:
        """Return decisions that resulted in element removal."""
        return [d for d in self.decisions if d.action == CleaningAction.REMOVE]

    @property
    def category_counts(self) -> dict[str, int]:
        """Return element removal count grouped by cleaning category."""
        counts: dict[str, int] = {}
        for d in self.removed_elements:
            if d.category:
                key = d.category.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def decision_source_counts(self) -> dict[str, int]:
        """Return element removal count grouped by decision source."""
        counts: dict[str, int] = {}
        for d in self.removed_elements:
            key = d.decision_source.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self, include_decisions: bool = True) -> dict[str, Any]:
        """Serialize cleaning report to a dictionary."""
        data: dict[str, Any] = {
            "total_input_elements": self.total_input_elements,
            "total_cleaned_elements": self.total_cleaned_elements,
            "removed_count": self.removed_count,
            "preserved_count": self.preserved_count,
            "category_counts": self.category_counts,
            "decision_source_counts": self.decision_source_counts,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
        if include_decisions:
            data["decisions"] = [d.to_dict() for d in self.decisions]
        return data

    def __repr__(self) -> str:
        """Safe representation without exposing sensitive contents."""
        return (
            f"CleaningReport(total_input={self.total_input_elements}, "
            f"cleaned={self.total_cleaned_elements}, "
            f"removed={self.removed_count}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms)"
        )


@dataclass(frozen=True)
class CleanedDocument(ParsedDocument):
    """Cleaned representation of a document produced by the Cleaning stage.

    Directly subclasses ParsedDocument to maintain 100% structural compatibility
    with downstream stages (Normalization, Chunking) while attaching the CleaningReport.
    """

    cleaning_report: CleaningReport = field(default_factory=CleaningReport)

    def to_dict(
        self,
        include_elements: bool = True,
        include_report: bool = True,
    ) -> dict[str, Any]:
        """Serialize cleaned document to a dictionary.

        Args:
            include_elements: Whether to include the list of serialized elements.
            include_report: Whether to include the cleaning report summary.
        """
        data = super().to_dict(include_elements=include_elements)
        if include_report:
            data["cleaning_report"] = self.cleaning_report.to_dict(
                include_decisions=include_elements
            )
        return data

    def __repr__(self) -> str:
        """Safe string representation omitting raw document contents."""
        return (
            f"CleanedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_elements={self.total_elements}, "
            f"removed_elements={self.cleaning_report.removed_count}, "
            f"document_version_id={self.document_version_id!r})"
        )
