"""Document cleaning service orchestrating deterministic and heuristic artifact removal."""

import asyncio
import time
from typing import Sequence

from app.core.logging import get_logger
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
from exceptions.cleaning import (
    CleaningProcessingError,
    InvalidCleaningInputError,
)
from parsing.models import ElementType, ParsedDocument, ParsedElement

logger = get_logger(__name__)


def _default_deterministic_rules() -> list[BaseDeterministicRule]:
    """Instantiate default deterministic cleaning rules."""
    return [
        MalformedStructuralArtifactRule(),
        EmptyContentRule(),
        InvalidControlCharsRule(),
        ExplicitPageNumberRule(),
    ]


def _default_heuristic_evaluators() -> list[BaseHeuristicEvaluator]:
    """Instantiate default structure-aware heuristic evaluators."""
    return [
        ExtractionDuplicationHeuristic(),
        RepetitiveHeaderFooterHeuristic(),
        NavigationBoilerplateHeuristic(),
    ]


class DocumentCleaningService:
    """Service coordinating structure-aware document cleaning.

    Applies deterministic rules for high-confidence artifacts and structure-aware
    heuristics for pattern-based noise, operating strictly on the intermediate
    ParsedDocument representation while preserving meaningful project knowledge.
    """

    def __init__(
        self,
        deterministic_rules: Sequence[BaseDeterministicRule] | None = None,
        heuristic_evaluators: Sequence[BaseHeuristicEvaluator] | None = None,
    ) -> None:
        self._rules = (
            list(deterministic_rules)
            if deterministic_rules is not None
            else _default_deterministic_rules()
        )
        self._heuristics = (
            list(heuristic_evaluators)
            if heuristic_evaluators is not None
            else _default_heuristic_evaluators()
        )

    def clean_sync(self, document: ParsedDocument) -> CleanedDocument:
        """Clean a ParsedDocument synchronously.

        Evaluates each element using deterministic rules and heuristic evaluators,
        removes identified artifacts, re-links broken hierarchy pointers, and returns
        a CleanedDocument.

        Args:
            document: Structured ParsedDocument from the parsing stage.

        Returns:
            CleanedDocument containing retained elements and the cleaning report.

        Raises:
            InvalidCleaningInputError: If document is not a ParsedDocument instance.
            CleaningProcessingError: If cleaning fails unexpectedly.
        """
        if not isinstance(document, ParsedDocument):
            raise InvalidCleaningInputError(
                f"Expected ParsedDocument, got '{type(document).__name__}'."
            )

        start_time = time.perf_counter()
        doc_id = document.document_id
        project_id = document.project_id
        doc_type_val = document.document_type.value

        logger.info(
            "Starting cleaning for document '%s' (project: '%s', type: '%s', elements: %d)",
            doc_id,
            project_id,
            doc_type_val,
            document.total_elements,
        )

        try:
            # Handle empty document edge case
            if document.total_elements == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                report = CleaningReport(
                    total_input_elements=0,
                    total_cleaned_elements=0,
                    removed_count=0,
                    preserved_count=0,
                    decisions=[],
                    elapsed_ms=elapsed_ms,
                )
                return CleanedDocument(
                    document_id=doc_id,
                    project_id=project_id,
                    document_type=document.document_type,
                    elements=[],
                    source_metadata=dict(document.source_metadata),
                    parser_metadata=dict(document.parser_metadata),
                    document_version_id=document.document_version_id,
                    cleaning_report=report,
                )

            # Build structural context in O(n) pass
            context = DocumentStructureContext(document)

            decisions: list[CleaningDecision] = []
            removed_element_ids: set[str] = set()
            removed_headings_info: dict[str, tuple[str | None, str]] = {}

            for elem in document.elements:
                decision: CleaningDecision | None = None

                # 1. Evaluate deterministic rules
                for rule in self._rules:
                    decision = rule.evaluate(elem, context)
                    if decision is not None and decision.action == CleaningAction.REMOVE:
                        break

                # 2. Evaluate heuristic evaluators if no deterministic removal
                if decision is None or decision.action != CleaningAction.REMOVE:
                    for evaluator in self._heuristics:
                        decision = evaluator.evaluate(elem, context)
                        if decision is not None and decision.action == CleaningAction.REMOVE:
                            break

                # 3. Fallback: Conservative preservation
                if decision is None or decision.action != CleaningAction.REMOVE:
                    decision = CleaningDecision(
                        element_id=elem.element_id,
                        action=CleaningAction.PRESERVE,
                        decision_source=DecisionSource.DETERMINISTIC,
                        rule_name="conservative_preservation",
                        reason="Element preserved as substantive content.",
                        confidence=1.0,
                        element_type=elem.element_type,
                    )

                decisions.append(decision)

                if decision.action == CleaningAction.REMOVE:
                    removed_element_ids.add(elem.element_id)
                    if elem.element_type == ElementType.HEADING:
                        removed_headings_info[elem.element_id] = (elem.parent_id, elem.content)
                    logger.debug(
                        "Removed element '%s' (category: %s, rule: %s, reason: %s)",
                        elem.element_id,
                        decision.category.value if decision.category else "none",
                        decision.rule_name,
                        decision.reason,
                    )

            # Re-construct elements list, repairing hierarchy if any heading was removed
            retained_elements: list[ParsedElement] = []
            for elem in document.elements:
                if elem.element_id in removed_element_ids:
                    continue

                parent_id = elem.parent_id
                section_path = elem.section_path

                # Check if parent was removed; repair parent link
                if parent_id in removed_headings_info:
                    parent_id = removed_headings_info[parent_id][0]

                # Remove any removed heading title from section_path
                if removed_headings_info:
                    removed_titles = {info[1] for info in removed_headings_info.values()}
                    section_path = tuple(t for t in section_path if t not in removed_titles)

                new_order = len(retained_elements)
                if new_order != elem.order or parent_id != elem.parent_id or section_path != elem.section_path:
                    retained_elements.append(
                        ParsedElement(
                            element_id=elem.element_id,
                            element_type=elem.element_type,
                            content=elem.content,
                            order=new_order,
                            heading_level=elem.heading_level,
                            parent_id=parent_id,
                            section_path=section_path,
                            metadata=dict(elem.metadata),
                        )
                    )
                else:
                    retained_elements.append(elem)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = CleaningReport(
                total_input_elements=document.total_elements,
                total_cleaned_elements=len(retained_elements),
                removed_count=len(removed_element_ids),
                preserved_count=len(retained_elements),
                decisions=decisions,
                elapsed_ms=elapsed_ms,
            )

            det_removals = report.decision_source_counts.get(DecisionSource.DETERMINISTIC.value, 0)
            heur_removals = report.decision_source_counts.get(DecisionSource.HEURISTIC.value, 0)

            logger.info(
                "Successfully cleaned document '%s' in %.2fms "
                "(retained: %d, removed: %d [deterministic: %d, heuristic: %d])",
                doc_id,
                elapsed_ms,
                len(retained_elements),
                len(removed_element_ids),
                det_removals,
                heur_removals,
            )

            return CleanedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                elements=retained_elements,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                document_version_id=document.document_version_id,
                cleaning_report=report,
            )

        except InvalidCleaningInputError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure cleaning document '%s': %s",
                doc_id,
                str(exc),
            )
            raise CleaningProcessingError(
                f"Unexpected failure cleaning document '{doc_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    async def clean(self, document: ParsedDocument) -> CleanedDocument:
        """Clean a ParsedDocument asynchronously via controlled thread offloading.

        Args:
            document: Structured ParsedDocument.

        Returns:
            CleanedDocument.
        """
        return await asyncio.to_thread(self.clean_sync, document)

    async def clean_batch(self, documents: list[ParsedDocument]) -> list[CleanedDocument]:
        """Clean multiple ParsedDocuments concurrently.

        Args:
            documents: List of ParsedDocuments to clean.

        Returns:
            List of CleanedDocument results preserving input order.
        """
        tasks = [self.clean(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_cleaning_service: DocumentCleaningService | None = None


def get_cleaning_service(
    deterministic_rules: Sequence[BaseDeterministicRule] | None = None,
    heuristic_evaluators: Sequence[BaseHeuristicEvaluator] | None = None,
) -> DocumentCleaningService:
    """Get or create the default DocumentCleaningService instance."""
    global _default_cleaning_service
    if deterministic_rules is not None or heuristic_evaluators is not None:
        return DocumentCleaningService(
            deterministic_rules=deterministic_rules,
            heuristic_evaluators=heuristic_evaluators,
        )
    if _default_cleaning_service is None:
        _default_cleaning_service = DocumentCleaningService()
    return _default_cleaning_service


def reset_cleaning_service() -> None:
    """Reset the singleton cleaning service instance. Useful for test isolation."""
    global _default_cleaning_service
    _default_cleaning_service = None
