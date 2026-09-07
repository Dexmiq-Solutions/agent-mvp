"""Comprehensive unit and integration tests for the Document Normalization layer."""

import pytest

from acquisition.formats import DocumentType
from cleaning import (
    CleanedDocument,
    CleaningReport,
    DocumentCleaningService,
)
from exceptions.normalization import (
    InvalidNormalizationInputError,
    NormalizationError,
)
from normalization import (
    CodeBlockNormalizer,
    DefaultNormalizer,
    DocumentNormalizationService,
    HeadingNormalizer,
    ListItemNormalizer,
    NormalizationConfig,
    NormalizationDecision,
    NormalizationReport,
    NormalizationRuleType,
    NormalizedDocument,
    ParagraphNormalizer,
    TableNormalizer,
    collapse_whitespace_in_line,
    get_normalization_service,
    normalize_blank_lines,
    normalize_line_endings,
    normalize_unicode_nfc,
    reset_normalization_service,
    strip_safe_control_characters,
)
from parsing.models import ElementType, ParsedDocument, ParsedElement


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_normalization_service()
    yield
    reset_normalization_service()


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
    """Helper to construct a ParsedElement test fixture."""
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


def make_parsed_doc(
    elements: list[ParsedElement],
    doc_id: str = "doc-norm-1",
    project_id: str = "proj-test",
    doc_type: DocumentType = DocumentType.MARKDOWN,
) -> ParsedDocument:
    """Helper to construct a ParsedDocument test fixture."""
    return ParsedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        elements=elements,
        source_metadata={"source": "test_source"},
        parser_metadata={"parser": "test_parser"},
        document_version_id="v1",
    )


def make_cleaned_doc(
    elements: list[ParsedElement],
    doc_id: str = "doc-clean-1",
    project_id: str = "proj-test",
    doc_type: DocumentType = DocumentType.MARKDOWN,
) -> CleanedDocument:
    """Helper to construct a CleanedDocument test fixture."""
    report = CleaningReport(
        total_input_elements=len(elements),
        total_cleaned_elements=len(elements),
        removed_count=0,
        preserved_count=len(elements),
        decisions=[],
        elapsed_ms=1.5,
    )
    return CleanedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        elements=elements,
        source_metadata={"source": "test_source"},
        parser_metadata={"parser": "test_parser"},
        document_version_id="v1",
        cleaning_report=report,
    )


# ==============================================================================
# 1. Low-Level Deterministic Transforms Tests
# ==============================================================================


def test_normalize_line_endings():
    """Verify conversion of CRLF and CR to LF."""
    # CRLF
    text_crlf = "First line\r\nSecond line\r\nThird line"
    norm, changed = normalize_line_endings(text_crlf)
    assert norm == "First line\nSecond line\nThird line"
    assert changed is True

    # CR
    text_cr = "Line 1\rLine 2"
    norm, changed = normalize_line_endings(text_cr)
    assert norm == "Line 1\nLine 2"
    assert changed is True

    # Already LF
    text_lf = "Line 1\nLine 2"
    norm, changed = normalize_line_endings(text_lf)
    assert norm == text_lf
    assert changed is False


def test_normalize_unicode_nfc():
    """Verify Unicode canonical composition (NFC) produces bitwise identical representations."""
    # Decomposed: 'e' + combining acute accent (\u0301)
    decomposed = "caf\u0065\u0301"
    # Precomposed: 'é' (\u00e9)
    precomposed = "caf\u00e9"

    assert decomposed != precomposed
    norm, changed = normalize_unicode_nfc(decomposed, "NFC")
    assert norm == precomposed
    assert changed is True

    # Already precomposed NFC
    norm2, changed2 = normalize_unicode_nfc(precomposed, "NFC")
    assert norm2 == precomposed
    assert changed2 is False


def test_strip_safe_control_characters():
    """Verify safe removal of null bytes and zero-width characters while preserving tabs and newlines."""
    # Embedded null byte and zero-width spaces
    dirty = "Hello\x00 World\u200b!\ufeff Test\u200e."
    cleaned, changed = strip_safe_control_characters(dirty)
    assert cleaned == "Hello World! Test."
    assert changed is True

    # Tabs and newlines must remain untouched!
    code_text = "\tdef foo():\n\t    return 42\n"
    cleaned2, changed2 = strip_safe_control_characters(code_text)
    assert cleaned2 == code_text
    assert changed2 is False


def test_collapse_whitespace_in_line():
    """Verify collapsing of multiple horizontal spaces into single space."""
    line = "Customer    must    approve   the request."
    norm = collapse_whitespace_in_line(line)
    assert norm == "Customer must approve the request."


def test_normalize_blank_lines():
    """Verify collapsing of excessive consecutive blank lines."""
    text = "Section 1\n\n\n\n\nSection 2"
    norm, changed = normalize_blank_lines(text, max_consecutive=1)
    # max_consecutive=1 means at most \n\n (1 empty line between texts)
    assert norm == "Section 1\n\nSection 2"
    assert changed is True

    # Normal single blank line is preserved
    text_normal = "Section 1\n\nSection 2"
    norm2, changed2 = normalize_blank_lines(text_normal, max_consecutive=1)
    assert norm2 == text_normal
    assert changed2 is False


# ==============================================================================
# 2. Structure-Aware Element Normalizers Tests
# ==============================================================================


def test_heading_normalizer():
    """Verify heading normalization enforces single-line and collapses whitespace."""
    normalizer = HeadingNormalizer()
    config = NormalizationConfig()

    # Heading with CRLF, excessive spaces, and internal newline
    raw_heading = "  ##  System    Architecture \r\n Overview  \t  "
    norm, rules = normalizer.normalize(raw_heading, config)
    assert norm == "## System Architecture Overview"
    assert NormalizationRuleType.WHITESPACE.value in rules
    assert NormalizationRuleType.LINE_ENDINGS.value in rules

    # Pure idempotency test
    norm2, rules2 = normalizer.normalize(norm, config)
    assert norm2 == norm
    assert len(rules2) == 0


def test_paragraph_normalizer():
    """Verify paragraph normalizer handles multiline text, spaces, and blank lines."""
    normalizer = ParagraphNormalizer()
    config = NormalizationConfig()

    raw_para = (
        "  Customer     must approve\r\n"
        "the    request before    submission.  \r\n"
        "\r\n"
        "\r\n"
        "  Approval   is mandatory.  "
    )
    norm, rules = normalizer.normalize(raw_para, config)
    expected = (
        "Customer must approve\n"
        "the request before submission.\n\n"
        "Approval is mandatory."
    )
    assert norm == expected
    assert NormalizationRuleType.LINE_ENDINGS.value in rules
    assert NormalizationRuleType.WHITESPACE.value in rules
    assert NormalizationRuleType.BLANK_LINES.value in rules


def test_code_block_normalizer_preserves_indentation():
    """CRITICAL: Verify code block indentation and intentional spacing are strictly preserved."""
    normalizer = CodeBlockNormalizer()
    config = NormalizationConfig(preserve_code_indentation=True)

    code_content = (
        "def authenticate_user(token: str) -> bool:\r\n"
        "    # 4-space indentation must be preserved!\r\n"
        "    if not token:    \r\n"
        "        return False\r\n"
        "\treturn verify_signature(token)\r\n"
    )
    norm, rules = normalizer.normalize(code_content, config)

    # Indentation preserved: 4 spaces and tab preserved, trailing spaces on lines cleaned
    expected = (
        "def authenticate_user(token: str) -> bool:\n"
        "    # 4-space indentation must be preserved!\n"
        "    if not token:\n"
        "        return False\n"
        "\treturn verify_signature(token)"
    )
    assert norm == expected
    assert "    if not token:" in norm
    assert "\treturn verify_signature(token)" in norm
    assert NormalizationRuleType.LINE_ENDINGS.value in rules


def test_table_normalizer():
    """Verify table row delimiters and alignment structure are preserved while standardizing cells."""
    normalizer = TableNormalizer()
    config = NormalizationConfig()

    table_content = (
        "|   Role   |   Permissions    |   Status   |\r\n"
        "|:---|:---:|---:|\r\n"
        "|   Admin   |   Full Access    |   Active   |\r\n"
        "|   Viewer  |   Read   Only    |   Active   |\r\n"
    )
    norm, rules = normalizer.normalize(table_content, config)
    expected = (
        "| Role | Permissions | Status |\n"
        "| :--- | :---: | ---: |\n"
        "| Admin | Full Access | Active |\n"
        "| Viewer | Read Only | Active |"
    )
    assert norm == expected
    assert NormalizationRuleType.LINE_ENDINGS.value in rules
    assert NormalizationRuleType.WHITESPACE.value in rules


def test_list_item_normalizer():
    """Verify list marker and indentation are preserved while body text is normalized."""
    normalizer = ListItemNormalizer()
    config = NormalizationConfig()

    raw_item = "    -   First    item with   excessive   spaces.  "
    norm, rules = normalizer.normalize(raw_item, config)
    assert norm == "    - First item with excessive spaces."
    assert NormalizationRuleType.WHITESPACE.value in rules

    # Numbered list item
    raw_numbered = "  1.   Step   one description.  "
    norm_num, _ = normalizer.normalize(raw_numbered, config)
    assert norm_num == "  1. Step one description."


# ==============================================================================
# 3. DocumentNormalizationService Orchestration Tests
# ==============================================================================


def test_normalize_sync_cleaned_document():
    """Verify normalizing a CleanedDocument preserves CleanedDocument attributes and reports."""
    elems = [
        make_elem("elem-0", "##  Requirement   Specification  ", ElementType.HEADING, 0, heading_level=2),
        make_elem("elem-1", "Customer    must   be authenticated.\r\n", ElementType.PARAGRAPH, 1, parent_id="elem-0", section_path=("Requirement Specification",), metadata={"character_count": 35}),
        make_elem("elem-2", "    def run():\n        pass", ElementType.CODE_BLOCK, 2, parent_id="elem-0", section_path=("Requirement Specification",)),
    ]
    cleaned_doc = make_cleaned_doc(elems)
    service = DocumentNormalizationService()

    normalized_doc = service.normalize_sync(cleaned_doc)

    assert isinstance(normalized_doc, NormalizedDocument)
    assert isinstance(normalized_doc, CleanedDocument)
    assert isinstance(normalized_doc, ParsedDocument)

    # Check that report is attached and cleaning report is preserved
    assert normalized_doc.cleaning_report == cleaned_doc.cleaning_report
    assert normalized_doc.normalization_report.total_elements == 3
    assert normalized_doc.normalization_report.normalized_count == 2  # elem-0 and elem-1 changed
    assert normalized_doc.normalization_report.unchanged_count == 1  # elem-2 was already normalized code

    # Check normalized contents
    assert normalized_doc.elements[0].content == "## Requirement Specification"
    assert normalized_doc.elements[1].content == "Customer must be authenticated."
    assert normalized_doc.elements[1].metadata["character_count"] == len("Customer must be authenticated.")
    assert normalized_doc.elements[2].content == "    def run():\n        pass"

    # Check section path normalization
    assert normalized_doc.elements[1].section_path == ("Requirement Specification",)


def test_empty_document_normalization():
    """Verify normalizing an empty document is safe and fast."""
    empty_doc = make_parsed_doc([])
    service = DocumentNormalizationService()

    normalized = service.normalize_sync(empty_doc)
    assert normalized.total_elements == 0
    assert normalized.normalization_report.total_elements == 0
    assert normalized.normalization_report.normalized_count == 0
    assert normalized.normalization_report.unchanged_count == 0


def test_invalid_input_raises_error():
    """Verify non-ParsedDocument input raises InvalidNormalizationInputError."""
    service = DocumentNormalizationService()
    with pytest.raises(InvalidNormalizationInputError):
        service.normalize_sync("not a parsed document")  # type: ignore


# ==============================================================================
# 4. Strict Idempotency Tests
# ==============================================================================


def test_normalization_idempotency():
    """CRITICAL: Verify normalize(normalize(doc)) == normalize(doc)."""
    elems = [
        make_elem("e-0", "  # System   Architecture  \r\n", ElementType.HEADING, 0, heading_level=1),
        make_elem("e-1", "Line 1    with   spaces.\r\n\r\n\r\nLine 2.\r\n", ElementType.PARAGRAPH, 1),
        make_elem("e-2", "    for i in range(10):\r\n        print(i)   \r\n", ElementType.CODE_BLOCK, 2),
        make_elem("e-3", "|  Col 1   |   Col 2  |\r\n|---|---|\r\n|  Val 1  |  Val 2  |", ElementType.TABLE, 3),
        make_elem("e-4", "  -   Item   1 description  \r\n", ElementType.LIST_ITEM, 4),
    ]
    doc = make_cleaned_doc(elems)
    service = DocumentNormalizationService()

    pass1 = service.normalize_sync(doc)
    pass2 = service.normalize_sync(pass1)

    assert pass1.total_elements == pass2.total_elements
    for e1, e2 in zip(pass1.elements, pass2.elements):
        assert e1.element_id == e2.element_id
        assert e1.element_type == e2.element_type
        assert e1.content == e2.content
        assert e1.order == e2.order
        assert e1.heading_level == e2.heading_level
        assert e1.parent_id == e2.parent_id
        assert e1.section_path == e2.section_path
        assert e1.metadata == e2.metadata

    # In pass 2, zero elements should be modified
    assert pass2.normalization_report.normalized_count == 0
    assert pass2.normalization_report.unchanged_count == len(elems)


# ==============================================================================
# 5. Meaning & Structure Preservation Tests
# ==============================================================================


def test_meaning_preservation_on_realistic_requirements():
    """Verify semantic meaning of business requirements is strictly preserved."""
    service = DocumentNormalizationService()

    requirement_text = (
        "The system    shall require multi-factor authentication (MFA)   \r\n"
        "for all administrative users prior to granting access to production environments.\r\n"
        "\r\n"
        "Failure to provide valid credentials within three attempts shall result in temporary lockout."
    )
    doc = make_parsed_doc([make_elem("elem-req", requirement_text, ElementType.PARAGRAPH)])
    norm_doc = service.normalize_sync(doc)

    norm_text = norm_doc.elements[0].content
    # Content must retain all semantic terms
    assert "multi-factor authentication (MFA)" in norm_text
    assert "administrative users" in norm_text
    assert "production environments." in norm_text
    assert "three attempts shall result in temporary lockout." in norm_text


def test_hierarchy_preservation():
    """Verify document hierarchy, parent_ids, and section_paths remain intact."""
    elems = [
        make_elem("h-1", "# Security Requirements", ElementType.HEADING, 0, heading_level=1, parent_id=None, section_path=("Security Requirements",)),
        make_elem("h-2", "## Authentication", ElementType.HEADING, 1, heading_level=2, parent_id="h-1", section_path=("Security Requirements", "Authentication")),
        make_elem("p-1", "Tokens must expire after 15 minutes.", ElementType.PARAGRAPH, 2, parent_id="h-2", section_path=("Security Requirements", "Authentication")),
    ]
    doc = make_parsed_doc(elems)
    service = DocumentNormalizationService()

    norm_doc = service.normalize_sync(doc)

    assert norm_doc.elements[0].parent_id is None
    assert norm_doc.elements[1].parent_id == "h-1"
    assert norm_doc.elements[2].parent_id == "h-2"
    assert norm_doc.elements[2].section_path == ("Security Requirements", "Authentication")


# ==============================================================================
# 6. Pipeline Boundary Tests (Cleaning vs Normalization vs Chunking)
# ==============================================================================


def test_cleaning_boundary_normalization_does_not_remove_elements():
    """Verify Normalization does NOT drop elements (which is Cleaning's responsibility)."""
    # Even if an element has navigation-like text or boilerplate, Normalization preserves it!
    elems = [
        make_elem("elem-0", "Back to top", ElementType.PARAGRAPH, 0),
        make_elem("elem-1", "Page 1 of 5", ElementType.PARAGRAPH, 1),
    ]
    doc = make_parsed_doc(elems)
    service = DocumentNormalizationService()

    norm_doc = service.normalize_sync(doc)
    # Both elements must be retained by Normalization
    assert norm_doc.total_elements == 2
    assert norm_doc.elements[0].content == "Back to top"
    assert norm_doc.elements[1].content == "Page 1 of 5"


def test_chunking_boundary_normalization_does_not_split_elements():
    """Verify Normalization does not split long elements into chunks or token limits."""
    long_content = "This is a long requirement paragraph. " * 50
    doc = make_parsed_doc([make_elem("elem-long", long_content, ElementType.PARAGRAPH)])
    service = DocumentNormalizationService()

    norm_doc = service.normalize_sync(doc)
    assert norm_doc.total_elements == 1
    assert norm_doc.elements[0].content == collapse_whitespace_in_line(long_content).strip()


# ==============================================================================
# 7. Async and Batch Normalization Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_async_normalize():
    """Verify async normalize execution via thread offloading."""
    doc = make_parsed_doc([make_elem("e-1", "Hello    World\r\n", ElementType.PARAGRAPH)])
    service = DocumentNormalizationService()

    normalized = await service.normalize(doc)
    assert normalized.elements[0].content == "Hello World"


@pytest.mark.asyncio
async def test_normalize_batch():
    """Verify batch normalization across multiple documents."""
    doc1 = make_parsed_doc([make_elem("e-1", "Doc 1    content\r\n", ElementType.PARAGRAPH)], doc_id="doc-1")
    doc2 = make_parsed_doc([make_elem("e-2", "Doc 2    content\r\n", ElementType.PARAGRAPH)], doc_id="doc-2")
    service = DocumentNormalizationService()

    results = await service.normalize_batch([doc1, doc2])
    assert len(results) == 2
    assert results[0].document_id == "doc-1"
    assert results[0].elements[0].content == "Doc 1 content"
    assert results[1].document_id == "doc-2"
    assert results[1].elements[0].content == "Doc 2 content"


# ==============================================================================
# 8. Serialization & Safe Representation Tests
# ==============================================================================


def test_serialization_and_safe_repr():
    """Verify safe repr masks document content and to_dict works."""
    elems = [make_elem("e-1", "Secret confidential text", ElementType.PARAGRAPH)]
    doc = make_parsed_doc(elems)
    service = DocumentNormalizationService()

    normalized = service.normalize_sync(doc)

    repr_str = repr(normalized)
    assert "Secret confidential text" not in repr_str
    assert "NormalizedDocument" in repr_str
    assert "document_id='doc-norm-1'" in repr_str

    report_repr = repr(normalized.normalization_report)
    assert "Secret confidential text" not in report_repr
    assert "NormalizationReport" in report_repr

    doc_dict = normalized.to_dict()
    assert doc_dict["document_id"] == "doc-norm-1"
    assert "normalization_report" in doc_dict
    assert doc_dict["normalization_report"]["total_elements"] == 1


# ==============================================================================
# 9. End-to-End Pipeline Flow Integration Test
# ==============================================================================


@pytest.mark.asyncio
async def test_end_to_end_parsing_cleaning_normalization_pipeline():
    """Verify complete pipeline stage flow: ParsedDocument -> CleanedDocument -> NormalizedDocument."""
    # 1. Parsed Document containing real content + extraction artifacts
    raw_elements = [
        make_elem("elem-0", "   ", ElementType.PARAGRAPH, 0),  # Empty artifact
        make_elem("elem-1", "##    System   Overview  \r\n", ElementType.HEADING, 1, heading_level=2),
        make_elem("elem-2", "Page 1 of 10", ElementType.PARAGRAPH, 2),  # Page number artifact
        make_elem(
            "elem-3",
            "The system    shall authenticate   users via OAuth 2.0.\r\n\r\n\r\nToken validation is required.",
            ElementType.PARAGRAPH,
            3,
            parent_id="elem-1",
            section_path=("System Overview",),
        ),
        make_elem(
            "elem-4",
            "```python\r\ndef verify(token):\r\n    return token.is_valid()\r\n```",
            ElementType.CODE_BLOCK,
            4,
            parent_id="elem-1",
            section_path=("System Overview",),
        ),
    ]
    parsed_doc = make_parsed_doc(raw_elements)

    # 2. Cleaning Stage
    cleaning_service = DocumentCleaningService()
    cleaned_doc = await cleaning_service.clean(parsed_doc)

    # Empty element and page number should be removed by Cleaning
    assert cleaned_doc.total_elements == 3
    assert cleaned_doc.cleaning_report.removed_count == 2

    # 3. Normalization Stage
    norm_service = DocumentNormalizationService()
    normalized_doc = await norm_service.normalize(cleaned_doc)

    # Normalization retains all 3 cleaned elements
    assert normalized_doc.total_elements == 3
    # Check normalized contents
    assert normalized_doc.elements[0].content == "## System Overview"
    assert normalized_doc.elements[1].content == (
        "The system shall authenticate users via OAuth 2.0.\n\n"
        "Token validation is required."
    )
    assert "def verify(token):\n    return token.is_valid()" in normalized_doc.elements[2].content

    # Both reports must be intact
    assert normalized_doc.cleaning_report.removed_count == 2
    assert normalized_doc.normalization_report.total_elements == 3
    assert normalized_doc.normalization_report.normalized_count >= 2
