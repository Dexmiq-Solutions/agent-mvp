"""Document normalization service orchestrating structure-aware deterministic normalization."""

import asyncio
from collections import defaultdict
import time
from typing import Mapping

from app.core.logging import get_logger
from rag.cleaning.models import CleanedDocument, CleaningReport
from exceptions.normalization import (
    InvalidNormalizationInputError,
    NormalizationProcessingError,
)
from rag.normalization.models import (
    NormalizationDecision,
    NormalizationReport,
    NormalizedDocument,
)
from rag.normalization.rules import (
    BaseElementNormalizer,
    CodeBlockNormalizer,
    DefaultNormalizer,
    HeadingNormalizer,
    ListItemNormalizer,
    NormalizationConfig,
    ParagraphNormalizer,
    TableNormalizer,
)
from rag.parsing.models import ElementType, ParsedDocument, ParsedElement

logger = get_logger(__name__)


def _default_normalizers() -> dict[ElementType, BaseElementNormalizer]:
    """Instantiate default element-specific normalizers."""
    return {
        ElementType.HEADING: HeadingNormalizer(),
        ElementType.PARAGRAPH: ParagraphNormalizer(),
        ElementType.TEXT_BLOCK: ParagraphNormalizer(),
        ElementType.CODE_BLOCK: CodeBlockNormalizer(),
        ElementType.TABLE: TableNormalizer(),
        ElementType.LIST_ITEM: ListItemNormalizer(),
    }


class DocumentNormalizationService:
    """Service orchestrating structure-aware deterministic document normalization.

    Processes cleaned documents in a single O(n) pass, applying element-specific
    deterministic rules to standardize whitespace, line endings, blank lines, and
    Unicode representation while strictly preserving hierarchy, intentional formatting,
    and semantic meaning.
    """

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        normalizers: Mapping[ElementType, BaseElementNormalizer] | None = None,
    ) -> None:
        self._config = config or NormalizationConfig()
        self._normalizers = dict(normalizers) if normalizers is not None else _default_normalizers()
        self._default_normalizer = DefaultNormalizer()
        self._heading_normalizer = self._normalizers.get(ElementType.HEADING) or HeadingNormalizer()

    def normalize_sync(self, document: ParsedDocument) -> NormalizedDocument:
        """Normalize a ParsedDocument or CleanedDocument synchronously.

        Iterates over all document elements in a single O(n) pass, resolves the
        appropriate structure-specific normalizer, updates representation-level
        inconsistencies, and returns an idempotent NormalizedDocument.

        Args:
            document: ParsedDocument or CleanedDocument to normalize.

        Returns:
            NormalizedDocument containing normalized elements and NormalizationReport.

        Raises:
            InvalidNormalizationInputError: If document is not a ParsedDocument instance.
            NormalizationProcessingError: If normalization fails unexpectedly.
        """
        if not isinstance(document, ParsedDocument):
            raise InvalidNormalizationInputError(
                f"Expected ParsedDocument or CleanedDocument, got '{type(document).__name__}'."
            )

        start_time = time.perf_counter()
        doc_id = document.document_id
        project_id = document.project_id
        doc_type_val = document.document_type.value

        logger.info(
            "Starting normalization for document '%s' (project: '%s', type: '%s', elements: %d)",
            doc_id,
            project_id,
            doc_type_val,
            document.total_elements,
        )

        try:
            # Preserve existing CleaningReport if present on input
            cleaning_report = getattr(document, "cleaning_report", CleaningReport())

            # Handle empty document edge case
            if document.total_elements == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                report = NormalizationReport(
                    total_elements=0,
                    normalized_count=0,
                    unchanged_count=0,
                    applied_rule_counts={},
                    decisions=[],
                    elapsed_ms=elapsed_ms,
                )
                return NormalizedDocument(
                    document_id=doc_id,
                    project_id=project_id,
                    document_type=document.document_type,
                    elements=[],
                    source_metadata=dict(document.source_metadata),
                    parser_metadata=dict(document.parser_metadata),
                    document_version_id=document.document_version_id,
                    cleaning_report=cleaning_report,
                    normalization_report=report,
                )

            normalized_elements: list[ParsedElement] = []
            decisions: list[NormalizationDecision] = []
            applied_rule_counts: dict[str, int] = defaultdict(int)
            normalized_count = 0
            unchanged_count = 0

            # Single O(n) pass over elements
            for elem in document.elements:
                normalizer = self._normalizers.get(elem.element_type, self._default_normalizer)
                norm_content, applied_rules = normalizer.normalize(elem.content, self._config)

                changed = norm_content != elem.content
                if changed:
                    normalized_count += 1
                    for rule in applied_rules:
                        applied_rule_counts[rule] += 1
                else:
                    unchanged_count += 1

                decisions.append(
                    NormalizationDecision(
                        element_id=elem.element_id,
                        element_type=elem.element_type,
                        changed=changed,
                        applied_rules=applied_rules,
                    )
                )

                # Sync character count in element metadata if tracked
                meta = dict(elem.metadata)
                if "character_count" in meta:
                    meta["character_count"] = len(norm_content)

                # Normalize section path breadcrumb strings consistently with headings
                norm_section_path = elem.section_path
                if elem.section_path:
                    norm_section_path = tuple(
                        self._heading_normalizer.normalize(crumb, self._config)[0]
                        for crumb in elem.section_path
                    )

                normalized_elements.append(
                    ParsedElement(
                        element_id=elem.element_id,
                        element_type=elem.element_type,
                        content=norm_content,
                        order=elem.order,
                        heading_level=elem.heading_level,
                        parent_id=elem.parent_id,
                        section_path=norm_section_path,
                        metadata=meta,
                    )
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = NormalizationReport(
                total_elements=document.total_elements,
                normalized_count=normalized_count,
                unchanged_count=unchanged_count,
                applied_rule_counts=dict(applied_rule_counts),
                decisions=decisions,
                elapsed_ms=elapsed_ms,
            )

            logger.info(
                "Successfully normalized document '%s' in %.2fms "
                "(total: %d, normalized: %d, unchanged: %d, rules: %s)",
                doc_id,
                elapsed_ms,
                document.total_elements,
                normalized_count,
                unchanged_count,
                dict(applied_rule_counts),
            )

            return NormalizedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                elements=normalized_elements,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                document_version_id=document.document_version_id,
                cleaning_report=cleaning_report,
                normalization_report=report,
            )

        except InvalidNormalizationInputError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure normalizing document '%s': %s",
                doc_id,
                str(exc),
            )
            raise NormalizationProcessingError(
                f"Unexpected failure normalizing document '{doc_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    async def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        """Normalize a ParsedDocument or CleanedDocument asynchronously.

        Offloads CPU-bound string transformations to a thread pool via asyncio.to_thread.

        Args:
            document: Structured ParsedDocument or CleanedDocument.

        Returns:
            NormalizedDocument.
        """
        return await asyncio.to_thread(self.normalize_sync, document)

    async def normalize_batch(
        self,
        documents: list[ParsedDocument],
    ) -> list[NormalizedDocument]:
        """Normalize multiple documents concurrently.

        Args:
            documents: List of ParsedDocuments or CleanedDocuments.

        Returns:
            List of NormalizedDocument results preserving input order.
        """
        tasks = [self.normalize(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_normalization_service: DocumentNormalizationService | None = None


def get_normalization_service(
    config: NormalizationConfig | None = None,
    normalizers: Mapping[ElementType, BaseElementNormalizer] | None = None,
) -> DocumentNormalizationService:
    """Get or create the default DocumentNormalizationService instance."""
    global _default_normalization_service
    if config is not None or normalizers is not None:
        return DocumentNormalizationService(config=config, normalizers=normalizers)
    if _default_normalization_service is None:
        _default_normalization_service = DocumentNormalizationService()
    return _default_normalization_service


def reset_normalization_service() -> None:
    """Reset the singleton normalization service instance. Useful for test isolation."""
    global _default_normalization_service
    _default_normalization_service = None
