"""Structure-aware heuristic cleaning layer for multi-signal artifact detection."""

from abc import ABC, abstractmethod
import re

from cleaning.context import DocumentStructureContext
from cleaning.models import (
    CleaningAction,
    CleaningCategory,
    CleaningDecision,
    DecisionSource,
)
from parsing.models import ElementType, ParsedElement

# Normative/requirement terms that must protect content from aggressive removal
REQUIREMENT_KEYWORDS = frozenset(
    [
        "shall",
        "must",
        "required",
        "mandatory",
        "prohibited",
        "warning:",
        "caution:",
        "important:",
        "compliance",
    ]
)

# Common navigation phrase patterns
NAVIGATION_PATTERNS = [
    re.compile(r"^(?:home\s*>\s*)+.*", re.IGNORECASE),
    re.compile(r"^back to top\s*$", re.IGNORECASE),
    re.compile(r"^skip to (?:main\s+)?content\s*$", re.IGNORECASE),
    re.compile(r"^previous(?:\s+page)?\s*\|\s*next(?:\s+page)?$", re.IGNORECASE),
    re.compile(r"^\s*«\s*previous\s*\|\s*next\s*»\s*$", re.IGNORECASE),
]

CONFIDENCE_THRESHOLD = 0.85


class BaseHeuristicEvaluator(ABC):
    """Abstract interface for structure-aware heuristic evaluators."""

    @property
    @abstractmethod
    def heuristic_name(self) -> str:
        """Unique identifier for the heuristic."""

    @abstractmethod
    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        """Evaluate element against structural heuristic signals.

        Returns:
            CleaningDecision with action=REMOVE if multiple signals provide high confidence,
            otherwise None (allowing fallback to conservative preservation).
        """


class RepetitiveHeaderFooterHeuristic(BaseHeuristicEvaluator):
    """Detects repetitive running headers and footers using multiple structural signals.

    Combines:
    - Page-level or section-level repetition frequency
    - Structural positioning (page boundary or section boundary)
    - Element type constraints (excludes substantive headings and tables)
    - Length constraints (running headers/footers are short)
    - Content characteristics (absence of normative requirement clauses)
    """

    @property
    def heuristic_name(self) -> str:
        return "repetitive_header_footer_heuristic"

    def _contains_requirement_language(self, text: str) -> bool:
        lower = text.lower()
        return any(keyword in lower for keyword in REQUIREMENT_KEYWORDS)

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        # Never remove substantive Headings, Tables, or Code Blocks via running header/footer heuristic
        if element.element_type in (ElementType.HEADING, ElementType.TABLE, ElementType.CODE_BLOCK):
            return None

        stripped = element.content.strip()
        length = len(stripped)

        # Running headers and footers are short (typically < 120 characters)
        if length > 120 or length == 0:
            return None

        # Guard against removing legitimate requirements or normative statements
        if self._contains_requirement_language(stripped):
            return None

        is_page_boundary = context.is_first_in_page(element.element_id) or context.is_last_in_page(
            element.element_id
        )
        is_first_page = context.is_first_in_page(element.element_id)
        is_last_page = context.is_last_in_page(element.element_id)

        rep_count = context.get_repetition_count(stripped)
        pages = context.get_page_distribution(stripped)

        score = 0.0
        reasons: list[str] = []

        # Signal 1: Page-level repetition across multiple distinct pages
        if context.has_page_info and len(pages) >= 2:
            score += 0.50
            reasons.append(f"repeated across {len(pages)} pages")
            if is_page_boundary:
                score += 0.40
                pos_str = "top" if is_first_page else "bottom"
                reasons.append(f"positioned at page {pos_str}")
        elif not context.has_page_info and rep_count >= 3:
            # Signal 2: Repetition across document without page info
            score += 0.40
            reasons.append(f"repeated {rep_count} times across document")
            if context.is_first_in_section(element.element_id) or context.is_last_in_section(
                element.element_id
            ):
                score += 0.45
                reasons.append("positioned at section boundary")
            elif context.is_first_in_document(element.element_id) or context.is_last_in_document(
                element.element_id
            ):
                score += 0.45
                reasons.append("positioned at document boundary")

        # Signal 3: Length brevity bonus (< 60 chars is typical for headers/footers)
        if length < 60 and score > 0.3:
            score += 0.10

        if score >= CONFIDENCE_THRESHOLD:
            category = (
                CleaningCategory.REPETITIVE_HEADER
                if (is_first_page or context.is_first_in_section(element.element_id))
                else CleaningCategory.REPETITIVE_FOOTER
            )
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=category,
                decision_source=DecisionSource.HEURISTIC,
                rule_name=self.heuristic_name,
                reason=f"Repetitive header/footer artifact ({', '.join(reasons)}).",
                confidence=min(score, 1.0),
                element_type=element.element_type,
                metadata={"score": round(score, 2), "repetition_count": rep_count},
            )

        return None


class NavigationBoilerplateHeuristic(BaseHeuristicEvaluator):
    """Detects website navigation breadcrumbs and repeated boilerplate structures."""

    @property
    def heuristic_name(self) -> str:
        return "navigation_boilerplate_heuristic"

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        if element.element_type in (ElementType.TABLE, ElementType.CODE_BLOCK):
            return None

        stripped = element.content.strip()
        length = len(stripped)

        if length > 120 or length == 0:
            return None

        score = 0.0
        reasons: list[str] = []

        # Check navigation patterns
        matched_nav = False
        for pattern in NAVIGATION_PATTERNS:
            if pattern.match(stripped):
                matched_nav = True
                score += 0.60
                reasons.append(f"matches navigation pattern '{pattern.pattern}'")
                break

        if not matched_nav:
            return None

        # Structural context signals
        if context.is_first_in_document(element.element_id) or context.is_last_in_document(
            element.element_id
        ):
            score += 0.30
            reasons.append("located at document boundary")
        elif context.is_first_in_page(element.element_id) or context.is_last_in_page(
            element.element_id
        ):
            score += 0.30
            reasons.append("located at page boundary")

        rep_count = context.get_repetition_count(stripped)
        if rep_count >= 2:
            score += 0.20
            reasons.append(f"repeated {rep_count} times")

        if score >= CONFIDENCE_THRESHOLD:
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=CleaningCategory.NAVIGATION_BOILERPLATE,
                decision_source=DecisionSource.HEURISTIC,
                rule_name=self.heuristic_name,
                reason=f"Navigation boilerplate artifact ({', '.join(reasons)}).",
                confidence=min(score, 1.0),
                element_type=element.element_type,
                metadata={"score": round(score, 2)},
            )

        return None


class ExtractionDuplicationHeuristic(BaseHeuristicEvaluator):
    """Detects immediately consecutive duplicate extractions produced by parser hiccups.

    Strictly conservative: only flags identical consecutive non-heading elements
    with identical content and structure. Legitimate non-consecutive repetitions
    are preserved.
    """

    @property
    def heuristic_name(self) -> str:
        return "extraction_duplication_heuristic"

    def evaluate(
        self,
        element: ParsedElement,
        context: DocumentStructureContext,
    ) -> CleaningDecision | None:
        prev_elem = context.get_previous_element(element.element_id)
        if not prev_elem:
            return None

        # Only evaluate identical consecutive elements of the same type
        if element.element_type != prev_elem.element_type:
            return None

        # Do not remove consecutive headings or code blocks (could be legitimately structured)
        if element.element_type in (ElementType.HEADING, ElementType.CODE_BLOCK):
            return None

        content_curr = element.content.strip()
        content_prev = prev_elem.content.strip()

        if not content_curr or content_curr != content_prev:
            return None

        # Confirm identical parent / section context
        if element.section_path == prev_elem.section_path:
            return CleaningDecision(
                element_id=element.element_id,
                action=CleaningAction.REMOVE,
                category=CleaningCategory.DUPLICATE,
                decision_source=DecisionSource.HEURISTIC,
                rule_name=self.heuristic_name,
                reason=(
                    f"Immediate consecutive duplicate of element '{prev_elem.element_id}' "
                    f"under section '{'/'.join(element.section_path) or 'root'}'."
                ),
                confidence=0.95,
                element_type=element.element_type,
                metadata={"duplicate_of": prev_elem.element_id},
            )

        return None
