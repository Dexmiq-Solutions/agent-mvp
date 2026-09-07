"""Comprehensive unit tests for the Document Cleaning layer."""

import pytest

from acquisition.formats import DocumentType
from cleaning import (
    CleanedDocument,
    CleaningAction,
    CleaningCategory,
    CleaningReport,
    DecisionSource,
    DocumentCleaningService,
    EmptyContentRule,
    ExplicitPageNumberRule,
    ExtractionDuplicationHeuristic,
    InvalidCleaningInputError,
    InvalidControlCharsRule,
    MalformedStructuralArtifactRule,
    NavigationBoilerplateHeuristic,
    RepetitiveHeaderFooterHeuristic,
    get_cleaning_service,
    reset_cleaning_service,
)
from parsing.models import ElementType, ParsedDocument, ParsedElement


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_cleaning_service()
    yield
    reset_cleaning_service()


def make_parsed_doc(
    elements: list[ParsedElement],
    doc_id: str = "doc-clean-1",
    project_id: str = "proj-test",
    doc_type: DocumentType = DocumentType.MARKDOWN,
) -> ParsedDocument:
    """Helper to construct a ParsedDocument test fixture."""
    return ParsedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        elements=elements,
        source_metadata={"source": "test"},
        parser_metadata={"parser": "test_parser"},
        document_version_id="v1",
    )


def make_elem(
    element_id: str,
    content: str,
    element_type: ElementType = ElementType.PARAGRAPH,
    order: int = 0,
    heading_level: int | None = None,
    parent_id: str | None = None,
    section_path: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> ParsedElement:
    """Helper to construct a ParsedElement."""
    return ParsedElement(
        element_id=element_id,
        element_type=element_type,
        content=content,
        order=order,
        heading_level=heading_level,
        parent_id=parent_id,
        section_path=section_path,
        metadata=metadata or {},
    )


# ==============================================================================
# 1. Deterministic Cleaning Tests
# ==============================================================================


def test_empty_content_rule_removes_whitespace_and_empty():
    """Verify empty strings and whitespace-only elements are deterministically removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "# Title", ElementType.HEADING, 0, heading_level=1, section_path=("Title",)),
        make_elem("e2", "", ElementType.PARAGRAPH, 1, parent_id="e1", section_path=("Title",)),
        make_elem("e3", "   \t\n  ", ElementType.PARAGRAPH, 2, parent_id="e1", section_path=("Title",)),
        make_elem("e4", "Meaningful body paragraph.", ElementType.PARAGRAPH, 3, parent_id="e1", section_path=("Title",)),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    assert cleaned.cleaning_report.removed_count == 2
    assert cleaned.elements[0].content == "# Title"
    assert cleaned.elements[1].content == "Meaningful body paragraph."
    assert cleaned.elements[1].order == 1

    removals = cleaned.cleaning_report.removed_elements
    assert all(r.category == CleaningCategory.EMPTY_CONTENT for r in removals)
    assert all(r.decision_source == DecisionSource.DETERMINISTIC for r in removals)


def test_empty_content_rule_removes_zero_width_characters():
    """Verify elements consisting solely of invisible zero-width characters are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "\u200b\u200c\u200d\ufeff", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Valid content here.", ElementType.PARAGRAPH, 1),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 1
    assert cleaned.elements[0].content == "Valid content here."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.EMPTY_CONTENT.value] == 1


def test_invalid_control_characters_rule():
    """Verify elements with only control characters are deterministically removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "\x00\x01\x02\x0b\x0c", ElementType.TEXT_BLOCK, 0),
        make_elem("e2", "Valid paragraph.", ElementType.PARAGRAPH, 1),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 1
    assert cleaned.elements[0].content == "Valid paragraph."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.INVALID_CONTROL_CHARS.value] == 1


def test_malformed_code_fence_and_heading_marker_removal():
    """Verify empty code fences, empty heading markers, and empty tables are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "###", ElementType.HEADING, 0, heading_level=3),
        make_elem("e2", "", ElementType.CODE_BLOCK, 1),
        make_elem("e3", "|---|---|", ElementType.TABLE, 2),
        make_elem("e4", "|   |   |", ElementType.TABLE, 3),
        make_elem("e5", "# Substantive Heading", ElementType.HEADING, 4, heading_level=1),
        make_elem("e6", "Substantive paragraph.", ElementType.PARAGRAPH, 5),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    assert cleaned.elements[0].content == "# Substantive Heading"
    assert cleaned.elements[1].content == "Substantive paragraph."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.EXTRACTION_ARTIFACT.value] == 4


def test_explicit_page_numbers_removal():
    """Verify standalone page numbers in various common formats are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "Page 1", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Page 2 of 42", ElementType.PARAGRAPH, 1),
        make_elem("e3", "- 3 -", ElementType.PARAGRAPH, 2),
        make_elem("e4", "-- 4 --", ElementType.PARAGRAPH, 3),
        make_elem("e5", "5 / 50", ElementType.PARAGRAPH, 4),
        make_elem("e6", "[ 6 ]", ElementType.PARAGRAPH, 5),
        make_elem("e7", "Page 7: System Requirements and Architecture", ElementType.PARAGRAPH, 6),
        make_elem("e8", "Substantive report content.", ElementType.PARAGRAPH, 7),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    assert cleaned.elements[0].content == "Page 7: System Requirements and Architecture"
    assert cleaned.elements[1].content == "Substantive report content."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.PAGE_NUMBER.value] == 6


def test_isolated_page_integer_boundary_removal():
    """Verify standalone page integers on page boundaries are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "1", ElementType.PARAGRAPH, 0, metadata={"page_number": 1}),
        make_elem("e2", "First page content paragraph.", ElementType.PARAGRAPH, 1, metadata={"page_number": 1}),
        make_elem("e3", "Second page content paragraph.", ElementType.PARAGRAPH, 2, metadata={"page_number": 2}),
        make_elem("e4", "2", ElementType.PARAGRAPH, 3, metadata={"page_number": 2}),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    assert cleaned.elements[0].content == "First page content paragraph."
    assert cleaned.elements[1].content == "Second page content paragraph."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.PAGE_NUMBER.value] == 2


# ==============================================================================
# 2. Heuristic Cleaning Tests
# ==============================================================================


def test_repetitive_running_header_and_footer_across_pages():
    """Verify repetitive headers and footers repeating across pages are heuristically removed."""
    service = DocumentCleaningService()
    elements = [
        # Page 1
        make_elem("e1", "Company Confidential - Q3 Report", ElementType.PARAGRAPH, 0, metadata={"page_number": 1}),
        make_elem("e2", "Introduction and scope of work.", ElementType.PARAGRAPH, 1, metadata={"page_number": 1}),
        make_elem("e3", "Copyright 2026 Acme Corp", ElementType.PARAGRAPH, 2, metadata={"page_number": 1}),
        # Page 2
        make_elem("e4", "Company Confidential - Q3 Report", ElementType.PARAGRAPH, 3, metadata={"page_number": 2}),
        make_elem("e5", "Detailed system architecture specifications.", ElementType.PARAGRAPH, 4, metadata={"page_number": 2}),
        make_elem("e6", "Copyright 2026 Acme Corp", ElementType.PARAGRAPH, 5, metadata={"page_number": 2}),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    assert cleaned.elements[0].content == "Introduction and scope of work."
    assert cleaned.elements[1].content == "Detailed system architecture specifications."

    report = cleaned.cleaning_report
    assert report.removed_count == 4
    assert report.decision_source_counts[DecisionSource.HEURISTIC.value] == 4
    assert CleaningCategory.REPETITIVE_HEADER.value in report.category_counts
    assert CleaningCategory.REPETITIVE_FOOTER.value in report.category_counts


def test_repetitive_header_footer_without_page_metadata():
    """Verify repetition at section boundaries in documents without explicit page info is removed."""
    service = DocumentCleaningService()
    elements = [
        # Section 1
        make_elem("h1", "Architecture", ElementType.HEADING, 0, heading_level=1, section_path=("Architecture",)),
        make_elem("e1", "ACME INTERNAL USE ONLY", ElementType.PARAGRAPH, 1, parent_id="h1", section_path=("Architecture",)),
        make_elem("e2", "Microservices overview description.", ElementType.PARAGRAPH, 2, parent_id="h1", section_path=("Architecture",)),
        # Section 2
        make_elem("h2", "Database", ElementType.HEADING, 3, heading_level=1, section_path=("Database",)),
        make_elem("e3", "ACME INTERNAL USE ONLY", ElementType.PARAGRAPH, 4, parent_id="h2", section_path=("Database",)),
        make_elem("e4", "PostgreSQL schema specifications.", ElementType.PARAGRAPH, 5, parent_id="h2", section_path=("Database",)),
        # Section 3
        make_elem("h3", "Security", ElementType.HEADING, 6, heading_level=1, section_path=("Security",)),
        make_elem("e5", "ACME INTERNAL USE ONLY", ElementType.PARAGRAPH, 7, parent_id="h3", section_path=("Security",)),
        make_elem("e6", "OAuth token validation flow.", ElementType.PARAGRAPH, 8, parent_id="h3", section_path=("Security",)),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    # All 3 substantive paragraphs and 3 headings must be preserved
    assert cleaned.total_elements == 6
    retained_contents = [e.content for e in cleaned.elements]
    assert "ACME INTERNAL USE ONLY" not in retained_contents
    assert "Microservices overview description." in retained_contents
    assert "PostgreSQL schema specifications." in retained_contents
    assert "OAuth token validation flow." in retained_contents


def test_navigation_boilerplate_heuristic():
    """Verify repeated navigation breadcrumbs and 'back to top' links are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "Home > Products > Cloud > Database", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Database service documentation and usage.", ElementType.PARAGRAPH, 1),
        make_elem("e3", "Back to top", ElementType.PARAGRAPH, 2),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 1
    assert cleaned.elements[0].content == "Database service documentation and usage."
    assert cleaned.cleaning_report.category_counts[CleaningCategory.NAVIGATION_BOILERPLATE.value] == 2


def test_consecutive_duplicate_extractions_removed():
    """Verify consecutive duplicate element extractions from parser bugs are removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "First unique paragraph.", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Duplicated paragraph text.", ElementType.PARAGRAPH, 1),
        make_elem("e3", "Duplicated paragraph text.", ElementType.PARAGRAPH, 2),  # bug duplicate
        make_elem("e4", "Third unique paragraph.", ElementType.PARAGRAPH, 3),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 3
    assert [e.content for e in cleaned.elements] == [
        "First unique paragraph.",
        "Duplicated paragraph text.",
        "Third unique paragraph.",
    ]
    assert cleaned.cleaning_report.category_counts[CleaningCategory.DUPLICATE.value] == 1


# ==============================================================================
# 3. Preservation & Conservative Policy Tests
# ==============================================================================


def test_legitimate_repeated_warnings_and_requirements_preserved():
    """Verify legitimately repeated requirements and safety warnings are NOT removed."""
    service = DocumentCleaningService()
    elements = [
        # Repeated safety warning across 3 different sections
        make_elem("h1", "Power System", ElementType.HEADING, 0, heading_level=1, section_path=("Power System",)),
        make_elem("w1", "Warning: Disconnect power before servicing equipment.", ElementType.PARAGRAPH, 1, parent_id="h1", section_path=("Power System",)),
        make_elem("p1", "Connect phase A to terminal 1.", ElementType.PARAGRAPH, 2, parent_id="h1", section_path=("Power System",)),
        make_elem("h2", "Battery System", ElementType.HEADING, 3, heading_level=1, section_path=("Battery System",)),
        make_elem("w2", "Warning: Disconnect power before servicing equipment.", ElementType.PARAGRAPH, 4, parent_id="h2", section_path=("Battery System",)),
        make_elem("p2", "Verify terminal voltage is 48V.", ElementType.PARAGRAPH, 5, parent_id="h2", section_path=("Battery System",)),
        make_elem("h3", "Inverter", ElementType.HEADING, 6, heading_level=1, section_path=("Inverter",)),
        make_elem("w3", "Warning: Disconnect power before servicing equipment.", ElementType.PARAGRAPH, 7, parent_id="h3", section_path=("Inverter",)),
        make_elem("p3", "Inverter must maintain 99.9% uptime.", ElementType.PARAGRAPH, 8, parent_id="h3", section_path=("Inverter",)),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    # All 9 elements must remain intact!
    assert cleaned.total_elements == 9
    assert cleaned.cleaning_report.removed_count == 0
    warnings = [e for e in cleaned.elements if "Disconnect power" in e.content]
    assert len(warnings) == 3


def test_conservative_ambiguous_content_preserved():
    """Verify single-occurrence and ambiguous potential boilerplate is preserved."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "Overview of company procedures.", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Navigation between modules is enabled via the main sidebar.", ElementType.PARAGRAPH, 1),
        make_elem("e3", "The administrator must approve all access requests.", ElementType.PARAGRAPH, 2),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    # All elements preserved
    assert cleaned.total_elements == 3
    assert cleaned.cleaning_report.removed_count == 0


def test_substantive_headings_tables_code_blocks_preserved():
    """Verify substantive headings, tables, and code blocks are never stripped by heuristics."""
    service = DocumentCleaningService()
    elements = [
        make_elem("h1", "Authentication Architecture", ElementType.HEADING, 0, heading_level=1),
        make_elem("t1", "User | Role | Access\nadmin | admin | all\nguest | guest | read", ElementType.TABLE, 1),
        make_elem("c1", "def authenticate(token):\n    return verify(token)", ElementType.CODE_BLOCK, 2),
        make_elem("p1", "Authentication uses JWT Bearer tokens.", ElementType.PARAGRAPH, 3),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 4
    assert cleaned.cleaning_report.removed_count == 0


# ==============================================================================
# 4. Hierarchy Preservation Tests
# ==============================================================================


def test_hierarchy_repair_when_empty_heading_removed():
    """Verify child elements have their parent_id re-linked when a malformed heading is removed."""
    service = DocumentCleaningService()
    elements = [
        make_elem("h1", "Root Section", ElementType.HEADING, 0, heading_level=1, section_path=("Root Section",)),
        make_elem("h2_bad", "###", ElementType.HEADING, 1, heading_level=3, parent_id="h1", section_path=("Root Section", "###")),
        make_elem("p1", "Child paragraph originally under malformed heading.", ElementType.PARAGRAPH, 2, parent_id="h2_bad", section_path=("Root Section", "###")),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 2
    root = cleaned.elements[0]
    child = cleaned.elements[1]

    assert root.element_id == "h1"
    assert child.element_id == "p1"
    # Parent re-linked to h1
    assert child.parent_id == "h1"
    # Breadcrumbs repaired to exclude "###"
    assert child.section_path == ("Root Section",)
    assert child.order == 1


# ==============================================================================
# 5. Idempotency Tests
# ==============================================================================


def test_cleaning_is_idempotent():
    """Verify clean(clean(doc)) == clean(doc)."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Page 1 of 5", ElementType.PARAGRAPH, 1),
        make_elem("e3", "# System Overview", ElementType.HEADING, 2, heading_level=1),
        make_elem("e4", "The system uses Python 3.12.", ElementType.PARAGRAPH, 3),
        make_elem("e5", "   \n  ", ElementType.PARAGRAPH, 4),
    ]
    doc = make_parsed_doc(elements)

    # First clean
    cleaned_pass_1 = service.clean_sync(doc)
    assert cleaned_pass_1.total_elements == 2
    assert cleaned_pass_1.cleaning_report.removed_count == 3

    # Second clean on the already cleaned document
    cleaned_pass_2 = service.clean_sync(cleaned_pass_1)
    assert cleaned_pass_2.total_elements == 2
    assert cleaned_pass_2.cleaning_report.removed_count == 0
    assert cleaned_pass_2.cleaning_report.preserved_count == 2

    # Verify elements are identical
    for elem1, elem2 in zip(cleaned_pass_1.elements, cleaned_pass_2.elements, strict=True):
        assert elem1.element_id == elem2.element_id
        assert elem1.content == elem2.content
        assert elem1.order == elem2.order
        assert elem1.element_type == elem2.element_type


# ==============================================================================
# 6. Multi-Format Compatibility Tests
# ==============================================================================


def test_txt_parsed_document_cleaning():
    """Verify cleaning works seamlessly on TXTParser output format."""
    service = DocumentCleaningService()
    elements = [
        make_elem("elem-0", "Header block.", ElementType.TEXT_BLOCK, 0),
        make_elem("elem-1", "", ElementType.TEXT_BLOCK, 1),
        make_elem("elem-2", "Valid paragraph content.", ElementType.TEXT_BLOCK, 2),
    ]
    doc = make_parsed_doc(elements, doc_type=DocumentType.TXT)
    cleaned = service.clean_sync(doc)

    assert isinstance(cleaned, CleanedDocument)
    assert cleaned.document_type == DocumentType.TXT
    assert cleaned.total_elements == 2
    assert cleaned.elements[0].content == "Header block."
    assert cleaned.elements[1].content == "Valid paragraph content."


def test_docx_parsed_document_cleaning():
    """Verify cleaning works seamlessly on DOCXParser output format."""
    service = DocumentCleaningService()
    elements = [
        make_elem("elem-0", "Title", ElementType.HEADING, 0, heading_level=1, metadata={"style_name": "Title"}),
        make_elem("elem-1", "   ", ElementType.PARAGRAPH, 1, metadata={"style_name": "Normal"}),
        make_elem("elem-2", "Cell A | Cell B", ElementType.TABLE, 2, metadata={"row_count": 1, "col_count": 2}),
    ]
    doc = make_parsed_doc(elements, doc_type=DocumentType.DOCX)
    cleaned = service.clean_sync(doc)

    assert isinstance(cleaned, CleanedDocument)
    assert cleaned.document_type == DocumentType.DOCX
    assert cleaned.total_elements == 2
    assert cleaned.elements[1].element_type == ElementType.TABLE


# ==============================================================================
# 7. Asynchronous & Batch API Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_async_clean_and_clean_batch():
    """Verify async clean() and clean_batch() execute successfully."""
    service = get_cleaning_service()
    doc1 = make_parsed_doc([
        make_elem("d1_e1", "Content A", ElementType.PARAGRAPH, 0),
        make_elem("d1_e2", "", ElementType.PARAGRAPH, 1),
    ], doc_id="doc-1")

    doc2 = make_parsed_doc([
        make_elem("d2_e1", "Content B", ElementType.PARAGRAPH, 0),
        make_elem("d2_e2", "Page 2", ElementType.PARAGRAPH, 1),
    ], doc_id="doc-2")

    # Async single
    cleaned1 = await service.clean(doc1)
    assert cleaned1.total_elements == 1
    assert cleaned1.elements[0].content == "Content A"

    # Async batch
    batch_results = await service.clean_batch([doc1, doc2])
    assert len(batch_results) == 2
    assert batch_results[0].document_id == "doc-1"
    assert batch_results[0].total_elements == 1
    assert batch_results[1].document_id == "doc-2"
    assert batch_results[1].total_elements == 1


# ==============================================================================
# 8. Error Handling & Traceability Tests
# ==============================================================================


def test_invalid_cleaning_input_error():
    """Verify passing non-ParsedDocument raises InvalidCleaningInputError."""
    service = DocumentCleaningService()
    with pytest.raises(InvalidCleaningInputError, match="Expected ParsedDocument"):
        service.clean_sync("not a document")  # type: ignore

    with pytest.raises(InvalidCleaningInputError):
        service.clean_sync(None)  # type: ignore


def test_empty_document_handling():
    """Verify empty ParsedDocument produces clean CleanedDocument with empty report."""
    service = DocumentCleaningService()
    doc = make_parsed_doc([])
    cleaned = service.clean_sync(doc)

    assert cleaned.total_elements == 0
    assert cleaned.cleaning_report.total_input_elements == 0
    assert cleaned.cleaning_report.removed_count == 0
    assert cleaned.cleaning_report.preserved_count == 0


def test_safe_repr_does_not_leak_text():
    """Verify CleanedDocument and CleaningReport __repr__ do not leak document text."""
    service = DocumentCleaningService()
    secret_text = "SECRET_CREDENTIAL_XYZ_12345"
    elements = [
        make_elem("e1", secret_text, ElementType.PARAGRAPH, 0),
        make_elem("e2", "", ElementType.PARAGRAPH, 1),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    repr_str = repr(cleaned)
    assert secret_text not in repr_str
    assert "CleanedDocument" in repr_str

    report_repr = repr(cleaned.cleaning_report)
    assert secret_text not in report_repr
    assert "CleaningReport" in report_repr


def test_serialization_to_dict():
    """Verify to_dict serialization on CleanedDocument and CleaningReport."""
    service = DocumentCleaningService()
    elements = [
        make_elem("e1", "Substantive content.", ElementType.PARAGRAPH, 0),
        make_elem("e2", "Page 1", ElementType.PARAGRAPH, 1),
    ]
    doc = make_parsed_doc(elements)
    cleaned = service.clean_sync(doc)

    data = cleaned.to_dict()
    assert data["document_id"] == "doc-clean-1"
    assert "cleaning_report" in data
    assert data["cleaning_report"]["removed_count"] == 1
    assert data["cleaning_report"]["total_cleaned_elements"] == 1
    assert len(data["elements"]) == 1
