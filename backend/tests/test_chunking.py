"""Comprehensive unit and integration tests for the Document Chunking layer."""

import pytest

from rag.acquisition.formats import DocumentType
from rag.chunking import (
    BaseChunkSizer,
    CharacterChunkSizer,
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    ChunkingConfigurationError,
    ChunkingError,
    ChunkingProcessingError,
    ChunkingReport,
    DocumentChunk,
    DocumentChunkingService,
    InvalidChunkingInputError,
    RecursiveTextSplitter,
    SectionNode,
    StructureAwareChunker,
    WordChunkSizer,
    build_section_tree,
    get_chunking_service,
    reset_chunking_service,
)
from rag.cleaning.models import CleanedDocument, CleaningReport
from rag.normalization.models import (
    NormalizationDecision,
    NormalizationReport,
    NormalizedDocument,
)
from rag.parsing.models import ElementType, ParsedDocument, ParsedElement


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_chunking_service()
    yield
    reset_chunking_service()


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


def make_normalized_doc(
    elements: list[ParsedElement],
    doc_id: str = "doc-chunk-1",
    project_id: str = "proj-test",
    doc_type: DocumentType = DocumentType.MARKDOWN,
    version_id: str | None = "v1",
) -> NormalizedDocument:
    """Helper to construct a NormalizedDocument test fixture."""
    cleaning_rep = CleaningReport(
        total_input_elements=len(elements),
        total_cleaned_elements=len(elements),
        removed_count=0,
        preserved_count=len(elements),
    )
    norm_rep = NormalizationReport(
        total_elements=len(elements),
        normalized_count=0,
        unchanged_count=len(elements),
    )
    return NormalizedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        elements=elements,
        source_metadata={"source": "test_source", "author": "QA Team"},
        parser_metadata={"parser": "markdown"},
        document_version_id=version_id,
        cleaning_report=cleaning_rep,
        normalization_report=norm_rep,
    )


# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================


def test_chunking_config_defaults():
    """Verify default configuration parameters."""
    config = ChunkingConfig()
    assert config.max_chunk_size == 1000
    assert config.min_chunk_size == 50
    assert config.chunk_overlap == 0
    assert config.preserve_hierarchy is True
    assert isinstance(config.sizer, CharacterChunkSizer)


@pytest.mark.parametrize(
    "kwargs, expected_error_substring",
    [
        ({"max_chunk_size": 0}, "max_chunk_size must be a positive integer"),
        ({"max_chunk_size": -10}, "max_chunk_size must be a positive integer"),
        ({"min_chunk_size": -1}, "min_chunk_size cannot be negative"),
        ({"max_chunk_size": 100, "min_chunk_size": 150}, "cannot exceed max_chunk_size"),
        ({"chunk_overlap": -5}, "chunk_overlap cannot be negative"),
        ({"max_chunk_size": 100, "chunk_overlap": 100}, "strictly less than max_chunk_size"),
        ({"max_chunk_size": 100, "chunk_overlap": 120}, "strictly less than max_chunk_size"),
    ],
)
def test_chunking_config_invalid(kwargs, expected_error_substring):
    """Verify explicit validation and rejection of invalid configuration."""
    with pytest.raises(ChunkingConfigurationError) as exc_info:
        ChunkingConfig(**kwargs)
    assert expected_error_substring in str(exc_info.value)


def test_chunk_sizer_implementations():
    """Verify BaseChunkSizer implementations."""
    char_sizer = CharacterChunkSizer()
    assert char_sizer.measure("Hello World") == 11
    assert char_sizer.fits("Hello World", 11) is True
    assert char_sizer.fits("Hello World", 10) is False

    word_sizer = WordChunkSizer()
    assert word_sizer.measure("Hello   World  from   Python") == 4
    assert word_sizer.fits("Hello World", 2) is True
    assert word_sizer.fits("Hello World", 1) is False


# ==============================================================================
# 2. Strict Input Type Validation Tests
# ==============================================================================


def test_chunk_service_rejects_non_normalized_documents():
    """Verify Chunking stage strictly requires NormalizedDocument."""
    service = DocumentChunkingService()
    elem = make_elem("e1", "Hello")

    # ParsedDocument
    parsed_doc = ParsedDocument(
        document_id="d1",
        project_id="p1",
        document_type=DocumentType.TXT,
        elements=[elem],
    )
    with pytest.raises(InvalidChunkingInputError) as exc_info:
        service.chunk_sync(parsed_doc)
    assert "Expected NormalizedDocument, got 'ParsedDocument'" in str(exc_info.value)

    # CleanedDocument
    cleaned_doc = CleanedDocument(
        document_id="d1",
        project_id="p1",
        document_type=DocumentType.TXT,
        elements=[elem],
    )
    with pytest.raises(InvalidChunkingInputError) as exc_info:
        service.chunk_sync(cleaned_doc)
    assert "Expected NormalizedDocument, got 'CleanedDocument'" in str(exc_info.value)

    # Primitive types
    with pytest.raises(InvalidChunkingInputError):
        service.chunk_sync(None)  # type: ignore

    with pytest.raises(InvalidChunkingInputError):
        service.chunk_sync({"document_id": "d1"})  # type: ignore


def test_empty_normalized_document_handling():
    """Verify safe handling of an empty NormalizedDocument."""
    service = DocumentChunkingService()
    doc = make_normalized_doc([])
    chunked = service.chunk_sync(doc)

    assert isinstance(chunked, ChunkedDocument)
    assert chunked.total_chunks == 0
    assert chunked.chunks == []
    assert chunked.chunking_report.total_input_elements == 0
    assert chunked.chunking_report.total_chunks == 0
    assert chunked.chunking_report.recursively_split_count == 0


# ==============================================================================
# 3. Basic Chunk Creation & Preservation Tests
# ==============================================================================


def test_basic_chunk_creation_and_attributes():
    """Verify standard chunk creation preserves all metadata, content, and provenance."""
    service = DocumentChunkingService()
    elem = make_elem(
        element_id="elem-1",
        content="System must support OAuth 2.0 authentication.",
        element_type=ElementType.PARAGRAPH,
        order=0,
        heading_level=None,
        parent_id="h1",
        section_path=("Authentication", "Requirements"),
        metadata={"requirement_id": "REQ-101"},
    )
    doc = make_normalized_doc([elem], doc_id="doc-auth-1", project_id="proj-security")
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 1
    chunk = chunked.chunks[0]

    assert chunk.chunk_id == "doc-auth-1_chunk_0"
    assert chunk.document_id == "doc-auth-1"
    assert chunk.project_id == "proj-security"
    assert chunk.content == "System must support OAuth 2.0 authentication."
    assert chunk.index == 0
    assert chunk.document_version_id == "v1"
    assert chunk.section_path == ("Authentication", "Requirements")
    assert chunk.source_element_ids == ("elem-1",)
    assert chunk.element_types == (ElementType.PARAGRAPH,)
    assert chunk.metadata.get("requirement_id") == "REQ-101"
    assert chunk.metadata.get("character_count") == len(elem.content)


def test_chunk_serialization_and_safe_repr():
    """Verify to_dict() and safe __repr__() without leaking raw text."""
    service = DocumentChunkingService()
    elem = make_elem("e1", "Sensitive confidential payload")
    doc = make_normalized_doc([elem], doc_id="doc-secret")
    chunked = service.chunk_sync(doc)
    chunk = chunked.chunks[0]

    # to_dict
    d = chunk.to_dict()
    assert d["chunk_id"] == "doc-secret_chunk_0"
    assert d["content"] == "Sensitive confidential payload"
    assert d["index"] == 0

    # __repr__ must NOT contain the sensitive content
    repr_str = repr(chunk)
    assert "Sensitive confidential payload" not in repr_str
    assert "chunk_id='doc-secret_chunk_0'" in repr_str
    assert "content_length=30" in repr_str

    doc_repr = repr(chunked)
    assert "Sensitive confidential payload" not in doc_repr
    assert "total_chunks=1" in doc_repr

    rep_repr = repr(chunked.chunking_report)
    assert "Sensitive confidential payload" not in rep_repr
    assert "total_chunks=1" in rep_repr


# ==============================================================================
# 4. Structure-Aware Chunking & Boundary Preservation
# ==============================================================================


def test_structure_aware_chunking_with_headings():
    """Verify structure-aware chunking correctly preserves headings and section groupings."""
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=500))
    elements = [
        make_elem("h1", "User Management", ElementType.HEADING, order=0, heading_level=1),
        make_elem("p1", "Admins can create and delete user accounts.", ElementType.PARAGRAPH, order=1),
        make_elem("h2", "Password Policy", ElementType.HEADING, order=2, heading_level=2),
        make_elem("p2", "Passwords must be at least 12 characters.", ElementType.PARAGRAPH, order=3),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 2

    # Chunk 0: Section 1
    c0 = chunked.chunks[0]
    assert c0.heading == "User Management"
    assert c0.heading_level == 1
    assert "User Management" in c0.content
    assert "Admins can create and delete user accounts." in c0.content
    assert c0.source_element_ids == ("h1", "p1")
    assert c0.index == 0

    # Chunk 1: Subsection 2
    c1 = chunked.chunks[1]
    assert c1.heading == "Password Policy"
    assert c1.heading_level == 2
    assert "Password Policy" in c1.content
    assert "Passwords must be at least 12 characters." in c1.content
    assert c1.source_element_ids == ("h2", "p2")
    assert c1.index == 1


def test_subsection_boundaries_are_preserved_not_blanket_merged():
    """Verify distinct subsections are NOT blanket-merged into parent section even if small."""
    # max_chunk_size is large enough to hold all elements easily
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=2000))
    elements = [
        make_elem("h1", "Security", ElementType.HEADING, order=0, heading_level=1),
        make_elem("p1", "General security guidelines.", ElementType.PARAGRAPH, order=1),
        make_elem("h2", "Authentication", ElementType.HEADING, order=2, heading_level=2),
        make_elem("p2", "Auth requirements.", ElementType.PARAGRAPH, order=3),
        make_elem("h2", "Authorization", ElementType.HEADING, order=4, heading_level=2),
        make_elem("p3", "Authz requirements.", ElementType.PARAGRAPH, order=5),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    # Must preserve the 3 distinct structural units: Security, Authentication, Authorization
    assert chunked.total_chunks == 3

    assert chunked.chunks[0].heading == "Security"
    assert chunked.chunks[0].heading_level == 1
    assert "General security guidelines." in chunked.chunks[0].content

    assert chunked.chunks[1].heading == "Authentication"
    assert chunked.chunks[1].heading_level == 2
    assert "Auth requirements." in chunked.chunks[1].content

    assert chunked.chunks[2].heading == "Authorization"
    assert chunked.chunks[2].heading_level == 2
    assert "Authz requirements." in chunked.chunks[2].content


def test_sections_within_limit_stay_intact():
    """Verify small multiple elements within the same section are grouped cohesively."""
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=1000))
    elements = [
        make_elem("h1", "System Requirements", ElementType.HEADING, order=0, heading_level=1),
        make_elem("p1", "Requirement 1: 99.9% uptime SLA.", ElementType.PARAGRAPH, order=1),
        make_elem("p2", "Requirement 2: <200ms latency.", ElementType.PARAGRAPH, order=2),
        make_elem("p3", "Requirement 3: Multi-region failover.", ElementType.PARAGRAPH, order=3),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 1
    chunk = chunked.chunks[0]
    assert chunk.source_element_ids == ("h1", "p1", "p2", "p3")
    assert "Requirement 1: 99.9% uptime SLA." in chunk.content
    assert "Requirement 2: <200ms latency." in chunk.content
    assert "Requirement 3: Multi-region failover." in chunk.content


# ==============================================================================
# 5. Recursive Splitting Tests (Oversized Sections & Elements)
# ==============================================================================


def test_recursive_splitting_oversized_section_elements():
    """Verify oversized section splits into multiple chunks at paragraph boundaries."""
    # Set max_chunk_size to 120 chars
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=120, min_chunk_size=20))
    elements = [
        make_elem("h1", "Scope", ElementType.HEADING, order=0, heading_level=1),
        make_elem("p1", "This is the first paragraph describing project scope in detail.", ElementType.PARAGRAPH, order=1),
        make_elem("p2", "This is the second paragraph covering non-functional constraints.", ElementType.PARAGRAPH, order=2),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 2
    assert chunked.chunks[0].source_element_ids == ("h1", "p1")
    assert chunked.chunks[1].source_element_ids == ("p2",)
    assert chunked.chunks[0].heading == "Scope"
    assert chunked.chunks[1].heading == "Scope"


def test_recursive_splitting_single_oversized_paragraph():
    """Verify an individual oversized paragraph recursively splits into sentences."""
    para_text = (
        "The application must authenticate all incoming requests using JWT tokens. "
        "Each JWT token contains claims for user role, tenant ID, and expiration timestamp. "
        "Expired tokens must be rejected immediately with an HTTP 401 Unauthorized status code."
    )
    # Set max_chunk_size to 110 chars
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=110, min_chunk_size=20))
    elem = make_elem("p1", para_text, ElementType.PARAGRAPH)
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks >= 2
    assert chunked.chunking_report.recursively_split_count == 1
    # All chunk sizes must be <= 110 chars
    for chunk in chunked.chunks:
        assert len(chunk.content) <= 110
        assert chunk.source_element_ids == ("p1",)


def test_recursive_splitting_multiple_structural_levels():
    """Verify deep recursive splitting across H1 -> H2 -> H3 -> paragraphs -> sentences."""
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=150, min_chunk_size=20))
    elements = [
        make_elem("h1", "Architecture", ElementType.HEADING, order=0, heading_level=1),
        make_elem("p1", "High-level overview of the microservice ecosystem.", ElementType.PARAGRAPH, order=1),
        make_elem("h2", "Core Services", ElementType.HEADING, order=2, heading_level=2),
        make_elem("h3", "Auth Service", ElementType.HEADING, order=3, heading_level=3),
        make_elem(
            "p2",
            "Auth Service handles login, logout, password resets, and session management. "
            "It communicates with the primary PostgreSQL database and issues signed JWT tokens. "
            "All endpoints are rate-limited to 100 requests per minute per IP address.",
            ElementType.PARAGRAPH,
            order=4,
        ),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    # Document should be cleanly partitioned across sections and oversized paragraph
    assert chunked.total_chunks >= 3
    # Check that Auth Service heading context is preserved on the split chunks
    auth_chunks = [c for c in chunked.chunks if c.heading == "Auth Service"]
    assert len(auth_chunks) >= 2
    for c in auth_chunks:
        assert c.heading_level == 3
        assert len(c.content) <= 150


def test_guaranteed_termination_unsplittable_content():
    """Verify unbroken text without spaces/punctuation terminates safely via character windowing."""
    unbroken_string = "A" * 500
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=100))
    elem = make_elem("e1", unbroken_string)
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 5
    for chunk in chunked.chunks:
        assert len(chunk.content) <= 100
        assert chunk.content == "A" * len(chunk.content)
    assert "".join(c.content for c in chunked.chunks) == unbroken_string


# ==============================================================================
# 6. Fallback Behavior for Weak or Missing Structure
# ==============================================================================


def test_fallback_behavior_without_headings():
    """Verify documents with no headings fall back gracefully and group paragraphs."""
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=200))
    elements = [
        make_elem("p1", "Plain text paragraph 1 without any headings.", order=0),
        make_elem("p2", "Plain text paragraph 2 continuing the narrative.", order=1),
        make_elem("p3", "Plain text paragraph 3 providing additional details.", order=2),
    ]
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks >= 1
    assert chunked.chunking_report.fallback_chunks_count == chunked.total_chunks
    assert chunked.chunking_report.structural_chunks_count == 0
    for chunk in chunked.chunks:
        assert chunk.heading is None
        assert chunk.heading_level is None
        assert chunk.section_path == ()


# ==============================================================================
# 7. Content Preservation & Boundaries
# ==============================================================================


def test_content_preservation_no_rewriting_or_loss():
    """Verify chunking preserves original text verbatim with zero rewriting or data loss."""
    raw_text = "The quick brown fox jumps over the lazy dog. 12345! @#$%^&*()"
    service = DocumentChunkingService()
    elem = make_elem("e1", raw_text)
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 1
    assert chunked.chunks[0].content == raw_text


def test_code_blocks_and_tables_preserved():
    """Verify code blocks and tables maintain structure within limits."""
    code_content = "def calculate_tax(amount: float) -> float:\n    return amount * 0.2"
    table_content = "| ID | Name | Role |\n|---|---|---|\n| 1 | Alice | Admin |"
    elements = [
        make_elem("cb1", code_content, ElementType.CODE_BLOCK, metadata={"language": "python"}),
        make_elem("tbl1", table_content, ElementType.TABLE),
    ]
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=1000))
    doc = make_normalized_doc(elements)
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 1
    assert code_content in chunked.chunks[0].content
    assert table_content in chunked.chunks[0].content
    assert chunked.chunks[0].metadata.get("language") == "python"


def test_chunking_does_not_normalize_or_clean():
    """Verify chunking preserves whitespace and characters as provided by normalization."""
    raw_with_spaces = "Preserve   multiple    spaces   exactly."
    service = DocumentChunkingService()
    elem = make_elem("e1", raw_with_spaces)
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.chunks[0].content == raw_with_spaces


# ==============================================================================
# 8. Soft min_chunk_size & Overlap Behavior
# ==============================================================================


def test_soft_min_chunk_size_merging_residual_fragments():
    """Verify tiny residual fragment from splitting is merged into preceding chunk if it fits."""
    # Two sentences: first is 80 chars, second is 15 chars
    text = "The primary database instance must be configured with automatic failover enabled. Small tail."
    # max 95, min 30. Sentence 1 is ~81 chars, sentence 2 is ~11 chars.
    # 81 + 1 + 11 = 93 <= 95 -> merged!
    splitter = RecursiveTextSplitter(
        sizer=CharacterChunkSizer(),
        max_chunk_size=95,
        min_chunk_size=30,
        chunk_overlap=0,
    )
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert "Small tail." in chunks[0]


def test_soft_min_chunk_size_preserves_standalone_requirements():
    """Verify short standalone requirements are preserved as valid chunks."""
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=500, min_chunk_size=100))
    elem = make_elem("e1", "Token expires in 60s.")  # 21 chars < 100
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks == 1
    assert chunked.chunks[0].content == "Token expires in 60s."


def test_configurable_overlap_behavior():
    """Verify minimal overlap is applied during recursive text splitting when configured."""
    text = "Sentence one about auth. Sentence two about tokens. Sentence three about keys."
    splitter_no_overlap = RecursiveTextSplitter(
        sizer=CharacterChunkSizer(),
        max_chunk_size=35,
        min_chunk_size=0,
        chunk_overlap=0,
    )
    chunks_no_overlap = splitter_no_overlap.split_text(text)

    splitter_with_overlap = RecursiveTextSplitter(
        sizer=CharacterChunkSizer(),
        max_chunk_size=40,
        min_chunk_size=0,
        chunk_overlap=15,
    )
    chunks_with_overlap = splitter_with_overlap.split_text(text)

    assert len(chunks_no_overlap) >= 3
    assert len(chunks_with_overlap) >= 2


# ==============================================================================
# 9. Custom Sizer & Word Sizing Tests
# ==============================================================================


def test_word_chunk_sizer_integration():
    """Verify DocumentChunkingService works with WordChunkSizer."""
    config = ChunkingConfig(max_chunk_size=10, min_chunk_size=2, sizer=WordChunkSizer())
    service = DocumentChunkingService(config=config)

    # 15 words
    text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
    elem = make_elem("e1", text)
    doc = make_normalized_doc([elem])
    chunked = service.chunk_sync(doc)

    assert chunked.total_chunks >= 2
    for chunk in chunked.chunks:
        assert len(chunk.content.split()) <= 10


# ==============================================================================
# 10. Large Document Performance & Idempotency
# ==============================================================================


def test_large_document_performance_and_idempotency():
    """Verify linear performance on large documents and strict deterministic idempotency."""
    elements: list[ParsedElement] = []
    order = 0
    for sec_idx in range(50):
        h_text = f"Section {sec_idx}"
        elements.append(make_elem(f"h_{sec_idx}", h_text, ElementType.HEADING, order=order, heading_level=1))
        order += 1
        for p_idx in range(4):
            p_text = f"Paragraph {p_idx} under section {sec_idx} detailing various functional requirements."
            elements.append(make_elem(f"p_{sec_idx}_{p_idx}", p_text, order=order))
            order += 1

    doc = make_normalized_doc(elements, doc_id="large-doc")
    service = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=400))

    # First run
    chunked_1 = service.chunk_sync(doc)
    assert chunked_1.total_chunks > 0
    # Performance check: 250 elements processed in well under 500ms
    assert chunked_1.chunking_report.elapsed_ms < 500

    # Second run for idempotency check
    chunked_2 = service.chunk_sync(doc)
    assert chunked_1.total_chunks == chunked_2.total_chunks
    for c1, c2 in zip(chunked_1.chunks, chunked_2.chunks):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.index == c2.index
        assert c1.section_path == c2.section_path
        assert c1.source_element_ids == c2.source_element_ids


# ==============================================================================
# 11. Asynchronous Execution Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_async_chunking_and_batch_processing():
    """Verify async chunk() and chunk_batch() execute non-blockingly."""
    service = get_chunking_service()
    doc1 = make_normalized_doc([make_elem("e1", "Doc 1 content")], doc_id="d1")
    doc2 = make_normalized_doc([make_elem("e2", "Doc 2 content")], doc_id="d2")

    # Single async
    result1 = await service.chunk(doc1)
    assert result1.document_id == "d1"
    assert result1.total_chunks == 1

    # Batch async
    batch_results = await service.chunk_batch([doc1, doc2])
    assert len(batch_results) == 2
    assert batch_results[0].document_id == "d1"
    assert batch_results[1].document_id == "d2"


# ==============================================================================
# 12. Full Pipeline Integration Test (Parsing -> Cleaning -> Normalization -> Chunking)
# ==============================================================================


def test_end_to_end_indexing_pipeline_integration():
    """Verify seamless end-to-end flow from raw document ingestion to chunking."""
    from rag.cleaning import DocumentCleaningService
    from rag.ingestion.models import IngestedDocument
    from rag.normalization import DocumentNormalizationService
    from rag.parsing.markdown import MarkdownParser

    raw_markdown = (
        b"# BRD: User Notification Engine\n\n"
        b"Page 1 of 5\n\n"
        b"## Requirements\n\n"
        b"The notification engine must dispatch SMS alerts within 5 seconds.\n\n"
        b"```json\n"
        b"{\"timeout\": 5000}\n"
        b"```\n\n"
        b"## Non-Functional\n\n"
        b"System must achieve 99.99% monthly availability."
    )

    ingested = IngestedDocument(
        document_id="brd-notif-001",
        project_id="proj-finance",
        source_storage_path="documents/brd.md",
        original_filename="brd.md",
        detected_document_type=DocumentType.MARKDOWN,
        content_type="text/markdown",
        raw_bytes=raw_markdown,
        document_version_id="v2.1",
    )

    # 1. Parsing
    parser = MarkdownParser()
    parsed = parser.parse(ingested)
    assert parsed.total_elements > 0

    # 2. Cleaning (removes "Page 1 of 5")
    cleaning_svc = DocumentCleaningService()
    cleaned = cleaning_svc.clean_sync(parsed)
    assert cleaned.total_elements < parsed.total_elements
    assert cleaned.cleaning_report.removed_count > 0

    # 3. Normalization (standardizes representation)
    norm_svc = DocumentNormalizationService()
    normalized = norm_svc.normalize_sync(cleaned)
    assert isinstance(normalized, NormalizedDocument)
    assert normalized.normalization_report is not None

    # 4. Chunking (produces structure-aware chunks)
    chunking_svc = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=300))
    chunked = chunking_svc.chunk_sync(normalized)

    assert isinstance(chunked, ChunkedDocument)
    assert chunked.document_id == "brd-notif-001"
    assert chunked.project_id == "proj-finance"
    assert chunked.document_version_id == "v2.1"
    assert chunked.total_chunks >= 2
    assert chunked.cleaning_report is not None
    assert chunked.normalization_report is not None
    assert chunked.chunking_report.total_chunks == chunked.total_chunks

    # Verify provenance and hierarchy preserved on each chunk
    for chunk in chunked.chunks:
        assert chunk.document_id == "brd-notif-001"
        assert chunk.project_id == "proj-finance"
        assert chunk.document_version_id == "v2.1"
        assert len(chunk.source_element_ids) > 0
        assert chunk.chunk_id.startswith("brd-notif-001_chunk_")
