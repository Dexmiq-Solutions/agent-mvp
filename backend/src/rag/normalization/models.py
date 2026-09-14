"""Domain models for the document normalization layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag.cleaning.models import CleanedDocument
from rag.parsing.models import ElementType


class NormalizationRuleType(str, Enum):
    """Enumeration of deterministic normalization rule categories."""

    LINE_ENDINGS = "line_endings"
    UNICODE_NFC = "unicode_nfc"
    WHITESPACE = "whitespace"
    CONTROL_CHARS = "control_chars"
    BLANK_LINES = "blank_lines"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class NormalizationDecision:
    """Detailed record of normalization applied to an individual document element.

    Provides full explainability and auditability for representation changes made
    to structural elements without storing full raw text.
    """

    element_id: str
    element_type: ElementType
    changed: bool
    applied_rules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalization decision to a dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "changed": self.changed,
            "applied_rules": list(self.applied_rules),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NormalizationReport:
    """Summary report detailing the results of the normalization stage.

    Contains aggregate metrics, change counts, and rule breakdowns without
    persisting or leaking raw document contents.
    """

    total_elements: int = 0
    normalized_count: int = 0
    unchanged_count: int = 0
    applied_rule_counts: dict[str, int] = field(default_factory=dict)
    decisions: list[NormalizationDecision] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self, include_decisions: bool = True) -> dict[str, Any]:
        """Serialize normalization report to a dictionary.

        Args:
            include_decisions: Whether to include the list of individual element decisions.
        """
        data: dict[str, Any] = {
            "total_elements": self.total_elements,
            "normalized_count": self.normalized_count,
            "unchanged_count": self.unchanged_count,
            "applied_rule_counts": dict(self.applied_rule_counts),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
        if include_decisions:
            data["decisions"] = [d.to_dict() for d in self.decisions]
        return data

    def __repr__(self) -> str:
        """Safe representation omitting raw document contents."""
        return (
            f"NormalizationReport(total_elements={self.total_elements}, "
            f"normalized={self.normalized_count}, "
            f"unchanged={self.unchanged_count}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms)"
        )


@dataclass(frozen=True)
class NormalizedDocument(CleanedDocument):
    """Normalized representation of a document produced by the Normalization stage.

    Directly subclasses CleanedDocument (and ParsedDocument) to maintain 100%
    structural compatibility with subsequent stages (Chunking, Embedding) while
    attaching the NormalizationReport and preserving any CleaningReport.
    """

    normalization_report: NormalizationReport = field(default_factory=NormalizationReport)

    def to_dict(
        self,
        include_elements: bool = True,
        include_reports: bool = True,
    ) -> dict[str, Any]:
        """Serialize normalized document to a dictionary.

        Args:
            include_elements: Whether to include the list of serialized elements.
            include_reports: Whether to include the cleaning and normalization reports.
        """
        data = super().to_dict(
            include_elements=include_elements,
            include_report=include_reports,
        )
        if include_reports:
            data["normalization_report"] = self.normalization_report.to_dict(
                include_decisions=include_elements
            )
        return data

    def __repr__(self) -> str:
        """Safe string representation omitting raw document text contents."""
        return (
            f"NormalizedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_elements={self.total_elements}, "
            f"normalized_elements={self.normalization_report.normalized_count}, "
            f"document_version_id={self.document_version_id!r})"
        )
