"""Deterministic cleaning rules for high-confidence artifact removal."""

from abc import ABC, abstractmethod
import re
from typing import ClassVar

from rag.cleaning.context import DocumentStructureContext
from rag.cleaning.models import (
    CleaningAction,
    CleaningCategory,
    CleaningDecision,
    DecisionSource,
)
from rag.parsing.models import ElementType, ParsedElement

# Unicode invisible / zero-width and control characters to check
ZERO_WIDTH_CHARS = frozenset(
    [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\ufeff",  # byte order mark / zero-width no-break space
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u00a0",  # non-breaking space
    ]
)

# Regex matching explicit page numbers
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^page\s+\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^--\s*\d+\s*--$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^\[\s*\d+\s*\]$"),
]

EMPTY_TABLE_REGEX = re.compile(r"^[\s\|\-:]+$")
EMPTY_HEADING_MARKER_REGEX = re.compile(r"^#{1,6}\s*$")


class BaseDeterministicRule(ABC):
    """Abstract interface for high-confidence deterministic cleaning rules."""

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Unique identifier for the rule."""

    @abstractmethod
    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        """Evaluate element against the deterministic rule.

        Returns:
            CleaningDecision with action=REMOVE if rule matches with certainty,
            otherwise None.
        """


class EmptyContentRule(BaseDeterministicRule):
    """Identifies elements containing only empty strings, whitespace, or zero-width characters."""

    @property
    def rule_name(self) -> str:
        return "empty_content_rule"

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        content = element.content

        # 1. Direct empty or whitespace string check
        if not content or content.strip() == "":
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=CleaningCategory.EMPTY_CONTENT,
                decision_source=DecisionSource.DETERMINISTIC,
                rule_name=self.rule_name,
                reason="Element content is empty or contains only whitespace.",
                confidence=1.0,
                element_type=element.element_type,
            )

        # 2. Check if content contains only invisible/zero-width characters
        cleaned_chars = "".join(c for c in content if c not in ZERO_WIDTH_CHARS and not c.isspace())
        if not cleaned_chars:
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=CleaningCategory.EMPTY_CONTENT,
                decision_source=DecisionSource.DETERMINISTIC,
                rule_name=self.rule_name,
                reason="Element content contains exclusively zero-width or non-printable whitespace characters.",
                confidence=1.0,
                element_type=element.element_type,
            )

        return None


class InvalidControlCharsRule(BaseDeterministicRule):
    """Identifies elements containing only invalid control characters and no printable text."""

    @property
    def rule_name(self) -> str:
        return "invalid_control_chars_rule"

    # Control chars excluding standard whitespace (\t, \n, \r)
    CONTROL_CHAR_REGEX: ClassVar[re.Pattern[str]] = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        content = element.content
        if not self.CONTROL_CHAR_REGEX.search(content):
            return None

        # Remove control characters and whitespace
        stripped_printable = self.CONTROL_CHAR_REGEX.sub("", content).strip()
        if not stripped_printable:
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=CleaningCategory.INVALID_CONTROL_CHARS,
                decision_source=DecisionSource.DETERMINISTIC,
                rule_name=self.rule_name,
                reason="Element content contains exclusively invalid control characters and no printable text.",
                confidence=1.0,
                element_type=element.element_type,
            )

        return None


class MalformedStructuralArtifactRule(BaseDeterministicRule):
    """Identifies malformed structural extraction remnants.

    Examples include empty code fences, standalone heading markers with no text,
    and empty table delimiter lines with no content cells.
    """

    @property
    def rule_name(self) -> str:
        return "malformed_structural_artifact_rule"

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        stripped = element.content.strip()

        # 1. Empty markdown code fence block (e.g. ``` or ```python\n``` or empty content)
        if element.element_type == ElementType.CODE_BLOCK:
            lines = [l.strip() for l in stripped.split("\n") if l.strip()]
            if not stripped or all(
                l.startswith("```") or l.startswith("~~~") for l in lines
            ):
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.EXTRACTION_ARTIFACT,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason="Code block element contains no code content.",
                    confidence=1.0,
                    element_type=element.element_type,
                )

        # 2. Heading marker with no title text (e.g. "#", "###")
        if element.element_type == ElementType.HEADING:
            if EMPTY_HEADING_MARKER_REGEX.match(stripped) or not stripped:
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.EXTRACTION_ARTIFACT,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason="Heading element contains heading marker hashes without heading title.",
                    confidence=1.0,
                    element_type=element.element_type,
                )

        # 3. Empty table containing only pipe, dash, and colon delimiters
        if element.element_type == ElementType.TABLE:
            # Check if all rows are merely pipes, dashes, or spaces
            lines = [line.strip() for line in stripped.split("\n") if line.strip()]
            if lines and all(EMPTY_TABLE_REGEX.match(line) for line in lines):
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.EXTRACTION_ARTIFACT,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason="Table element contains only delimiter characters without substantive cell content.",
                    confidence=1.0,
                    element_type=element.element_type,
                )

        return None


class ExplicitPageNumberRule(BaseDeterministicRule):
    """Identifies explicit page numbers appearing as standalone elements."""

    @property
    def rule_name(self) -> str:
        return "explicit_page_number_rule"

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        # Tables and Code Blocks cannot be page numbers
        if element.element_type in (ElementType.TABLE, ElementType.CODE_BLOCK):
            return None

        stripped = element.content.strip()

        # Check explicit page number regex patterns
        for pattern in PAGE_NUMBER_PATTERNS:
            if pattern.match(stripped):
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.PAGE_NUMBER,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason=f"Standalone page number pattern detected: '{stripped}'.",
                    confidence=1.0,
                    element_type=element.element_type,
                    metadata={"pattern": pattern.pattern},
                )

        # Standalone integer (e.g., '1', '12') on page boundaries or isolated
        if stripped.isdigit() and len(stripped) <= 4:
            # If element has page metadata and is first or last in that page
            if context.is_first_in_page(element.element_id) or context.is_last_in_page(
                element.element_id
            ):
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.PAGE_NUMBER,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason=f"Isolated page number integer '{stripped}' located at page boundary.",
                    confidence=1.0,
                    element_type=element.element_type,
                )

            # Or standalone integer at document boundary
            if context.is_first_in_document(
                element.element_id
            ) or context.is_last_in_document(element.element_id):
                return CleaningDecision(
                    element_id=element.element_id,
                    action=CleaningAction.REMOVE,
                    category=CleaningCategory.PAGE_NUMBER,
                    decision_source=DecisionSource.DETERMINISTIC,
                    rule_name=self.rule_name,
                    reason=f"Isolated page number integer '{stripped}' located at document boundary.",
                    confidence=1.0,
                    element_type=element.element_type,
                )

        return None
