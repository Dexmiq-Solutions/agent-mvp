"""Comprehensive unit and integration tests for the Document Metadata Enrichment layer."""

import pytest

from acquisition.formats import DocumentType
from chunking.models import (
    ChunkedDocument,
    ChunkingConfig,
    ChunkingReport,
    DocumentChunk,
)
from cleaning.models import CleaningReport
from exceptions.metadata_enrichment import (
    InvalidMetadataEnrichmentInputError,
    MetadataEnrichmentConfigurationError,
    MetadataEnrichmentError,
    MetadataEnrichmentProcessingError,
)
from metadata_enrichment import (
    BaseMetadataEnricher,
    ChunkContentType,
    ChunkMetadata,
    DocumentEnrichmentContext,
    DocumentMetadataEnrichmentService,
    EnrichedChunk,
    EnrichedDocument,
    MetadataCategory,
    MetadataEnrichmentConfig,
    MetadataEnrichmentReport,
    get_metadata_enrichment_service,
    reset_metadata_enrichment_service,
)
from normalization.models import NormalizationReport
from parsing.models import ElementType


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_metadata_enrichment_service()
    yield
    reset_metadata_enrichment_service()


def make_chunk(
    chunk_id: str,
    content: str,
    index: int = 0,
    doc_id: str = "doc-1",
    project_id: str = "proj-alpha",
    version_id: str | None = "v1.0",
    section_path: tuple[str, ...] = (),
    heading: str | None = None,
    heading_level: int | None = None,
    parent_element_id: str | None = None,
    parent_chunk_id: str | None = None,
    source_element_ids: tuple[str, ...] = (),
    element_types: tuple[ElementType, ...] = (ElementType.PARAGRAPH,),
    metadata: dict | None = None,
) -> DocumentChunk:
    """Helper to construct a DocumentChunk test fixture."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        project_id=project_id,
        content=content,
        index=index,
        document_version_id=version_id,
        section_path=section_path,
        heading=heading,
        heading_level=heading_level,
        parent_element_id=parent_element_id,
        parent_chunk_id=parent_chunk_id,
        source_element_ids=source_element_ids,
        element_types=element_types,
        metadata=metadata or {},
    )


def make_chunked_doc(
    chunks: list[DocumentChunk],
    doc_id: str = "doc-1",
    project_id: str = "proj-alpha",
    doc_type: DocumentType = DocumentType.MARKDOWN,
    version_id: str | None = "v1.0",
    source_metadata: dict | None = None,
    parser_metadata: dict | None = None,
) -> ChunkedDocument:
    """Helper to construct a ChunkedDocument test fixture."""
    cleaning_rep = CleaningReport(
        total_input_elements=len(chunks),
        total_cleaned_elements=len(chunks),
        removed_count=0,
        preserved_count=len(chunks),
    )
    norm_rep = NormalizationReport(
        total_elements=len(chunks),
        normalized_count=0,
        unchanged_count=len(chunks),
    )
    chunking_rep = ChunkingReport(
        total_input_elements=len(chunks),
        total_chunks=len(chunks),
        structural_chunks_count=len(chunks),
    )
    default_source_meta = {
        "source_storage_path": f"{project_id}/specs/{doc_id}.md",
        "original_filename": f"{doc_id}.md",
        "content_type": "text/markdown",
    }
    return ChunkedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        chunks=chunks,
        document_version_id=version_id,
        source_metadata=source_metadata if source_metadata is not None else default_source_meta,
        parser_metadata=parser_metadata or {"parser": "markdown"},
        cleaning_report=cleaning_rep,
        normalization_report=norm_rep,
        chunking_report=chunking_rep,
    )


# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================


def test_metadata_enrichment_config_defaults():
    """Verify default configuration values."""
    config = MetadataEnrichmentConfig()
    assert config.heading_separator == " > "
    assert config.enable_provenance is True
    assert config.enable_structural is True
    assert config.enable_characteristics is True
    assert config.custom_enrichers == ()


def test_metadata_enrichment_config_invalid():
    """Verify rejection of invalid configuration parameters."""
    with pytest.raises(MetadataEnrichmentConfigurationError) as exc_info:
        MetadataEnrichmentConfig(heading_separator="")
    assert "heading_separator cannot be empty" in str(exc_info.value)


# ==============================================================================
# 2. Strict Input Type Validation & Error Handling Tests
# ==============================================================================


def test_enrich_rejects_invalid_input_type():
    """Verify service rejects inputs that are not ChunkedDocument instances."""
    service = DocumentMetadataEnrichmentService()

    with pytest.raises(InvalidMetadataEnrichmentInputError) as exc_info:
        service.enrich_sync("not a document")  # type: ignore
    assert "Expected ChunkedDocument" in str(exc_info.value)

    with pytest.raises(InvalidMetadataEnrichmentInputError) as exc_info:
        service.enrich_sync(None)  # type: ignore
    assert "Expected ChunkedDocument" in str(exc_info.value)


def test_enrich_rejects_corrupted_chunk_in_document():
    """Verify service rejects chunked documents containing non-DocumentChunk items."""
    service = DocumentMetadataEnrichmentService()
    doc = make_chunked_doc([])
    # Inject invalid item into chunks
    corrupted_doc = ChunkedDocument(
        document_id=doc.document_id,
        project_id=doc.project_id,
        document_type=doc.document_type,
        chunks=["invalid_chunk"],  # type: ignore
    )

    with pytest.raises(InvalidMetadataEnrichmentInputError) as exc_info:
        service.enrich_sync(corrupted_doc)
    assert "Expected DocumentChunk in chunks list" in str(exc_info.value)


# ==============================================================================
# 3. Content Verbatim Preservation Tests
# ==============================================================================


def test_enrichment_preserves_content_strictly_verbatim():
    """Verify Metadata Enrichment never mutates, normalizes, or summarizes chunk text."""
    service = DocumentMetadataEnrichmentService()
    original_text_1 = "The client approved SSO authentication for Phase 1 delivery."
    original_text_2 = "```json\n{\n  \"timeout\": 5000\n}\n```"

    chunk1 = make_chunk("c1", original_text_1, index=0)
    chunk2 = make_chunk("c2", original_text_2, index=1)
    doc = make_chunked_doc([chunk1, chunk2])

    enriched_doc = service.enrich_sync(doc)

    assert len(enriched_doc.chunks) == 2
    assert enriched_doc.chunks[0].content == original_text_1
    assert enriched_doc.chunks[1].content == original_text_2
    assert enriched_doc.chunks[0].content is chunk1.content or enriched_doc.chunks[0].content == chunk1.content


# ==============================================================================
# 4. Provenance & Identity Preservation Tests
# ==============================================================================


def test_provenance_preservation():
    """Verify complete preservation of project, document, version, and source references."""
    service = DocumentMetadataEnrichmentService()

    chunk = make_chunk(
        chunk_id="chk-101",
        content="Functional requirements body.",
        index=0,
        doc_id="doc-fin-99",
        project_id="proj-finance-core",
        version_id="v3.2.1",
        source_element_ids=("elem-1", "elem-2"),
        metadata={"pre_existing_key": "val123"},
    )
    source_meta = {
        "source_storage_path": "proj-finance-core/contracts/spec.docx",
        "original_filename": "spec.docx",
        "author": "Architecture Board",
    }
    doc = make_chunked_doc(
        chunks=[chunk],
        doc_id="doc-fin-99",
        project_id="proj-finance-core",
        doc_type=DocumentType.DOCX,
        version_id="v3.2.1",
        source_metadata=source_meta,
    )

    enriched_doc = service.enrich_sync(doc)
    enriched_chunk = enriched_doc.chunks[0]

    # Model level properties
    assert enriched_chunk.project_id == "proj-finance-core"
    assert enriched_chunk.document_id == "doc-fin-99"
    assert enriched_chunk.chunk_id == "chk-101"
    assert enriched_chunk.document_version_id == "v3.2.1"
    assert enriched_chunk.source_element_ids == ("elem-1", "elem-2")

    # Structured metadata level
    meta = enriched_chunk.enriched_metadata
    assert meta is not None
    assert meta.project_id == "proj-finance-core"
    assert meta.document_id == "doc-fin-99"
    assert meta.chunk_id == "chk-101"
    assert meta.document_version_id == "v3.2.1"
    assert meta.source_element_ids == ("elem-1", "elem-2")
    assert meta.document_type == "docx"
    assert meta.source_storage_path == "proj-finance-core/contracts/spec.docx"
    assert meta.original_filename == "spec.docx"
    assert meta.source_metadata["author"] == "Architecture Board"

    # Pre-existing chunk metadata is preserved
    assert enriched_chunk.metadata["pre_existing_key"] == "val123"


# ==============================================================================
# 5. Multi-Tenant Project Isolation Tests
# ==============================================================================


def test_project_isolation_rejects_mismatched_chunk_project_id():
    """Verify service immediately rejects document if any chunk has a mismatched project_id."""
    service = DocumentMetadataEnrichmentService()

    valid_chunk = make_chunk("c1", "Text 1", index=0, project_id="proj-A")
    alien_chunk = make_chunk("c2", "Alien text", index=1, project_id="proj-B")

    doc = make_chunked_doc([valid_chunk, alien_chunk], project_id="proj-A")

    with pytest.raises(MetadataEnrichmentProcessingError) as exc_info:
        service.enrich_sync(doc)
    assert "Project isolation violation" in str(exc_info.value)
    assert "proj-B" in str(exc_info.value)
    assert "proj-A" in str(exc_info.value)


def test_batch_enrichment_strictly_preserves_tenant_isolation():
    """Verify batch enrichment keeps projects strictly isolated without cross-talk."""
    service = DocumentMetadataEnrichmentService()

    doc_a = make_chunked_doc(
        [make_chunk("c_a1", "Project A Content", project_id="tenant-A")],
        doc_id="doc-A",
        project_id="tenant-A",
    )
    doc_b = make_chunked_doc(
        [make_chunk("c_b1", "Project B Content", project_id="tenant-B")],
        doc_id="doc-B",
        project_id="tenant-B",
    )

    results = service.enrich_sync(doc_a), service.enrich_sync(doc_b)

    assert results[0].project_id == "tenant-A"
    assert results[0].chunks[0].project_id == "tenant-A"
    assert results[0].chunks[0].enriched_metadata.project_id == "tenant-A"

    assert results[1].project_id == "tenant-B"
    assert results[1].chunks[0].project_id == "tenant-B"
    assert results[1].chunks[0].enriched_metadata.project_id == "tenant-B"


# ==============================================================================
# 6. Structural Hierarchy & Breadcrumb Derivation Tests
# ==============================================================================


def test_structural_metadata_derivation():
    """Verify section paths, heading breadcrumbs, hierarchy depth, and relative positions."""
    service = DocumentMetadataEnrichmentService(
        config=MetadataEnrichmentConfig(heading_separator=" / ")
    )

    chunks = [
        make_chunk(
            "c0",
            "Executive summary overview.",
            index=0,
            section_path=("Overview",),
            heading="Overview",
            heading_level=1,
            parent_element_id="elem-h1",
        ),
        make_chunk(
            "c1",
            "SSO and MFA requirements details.",
            index=1,
            section_path=("Requirements", "Security", "Authentication"),
            heading="Authentication",
            heading_level=3,
            parent_element_id="elem-h3",
        ),
        make_chunk(
            "c2",
            "Root level concluding notes.",
            index=2,
            section_path=(),
            heading=None,
            heading_level=None,
            parent_element_id=None,
        ),
    ]
    doc = make_chunked_doc(chunks)

    enriched_doc = service.enrich_sync(doc)
    assert enriched_doc.total_chunks == 3

    # Chunk 0
    c0 = enriched_doc.chunks[0].enriched_metadata
    assert c0.section_path == ("Overview",)
    assert c0.heading == "Overview"
    assert c0.heading_level == 1
    assert c0.heading_path_str == "Overview"
    assert c0.hierarchy_depth == 1
    assert c0.chunk_index == 0
    assert c0.total_chunks == 3
    assert c0.relative_position == 0.0

    # Chunk 1 (Deeply nested)
    c1 = enriched_doc.chunks[1].enriched_metadata
    assert c1.section_path == ("Requirements", "Security", "Authentication")
    assert c1.heading == "Authentication"
    assert c1.heading_level == 3
    assert c1.heading_path_str == "Requirements / Security / Authentication"
    assert c1.hierarchy_depth == 3
    assert c1.chunk_index == 1
    assert c1.total_chunks == 3
    assert c1.relative_position == round(1 / 3, 4)

    # Chunk 2 (Root level without headings)
    c2 = enriched_doc.chunks[2].enriched_metadata
    assert c2.section_path == ()
    assert c2.heading is None
    assert c2.heading_path_str == ""
    assert c2.hierarchy_depth == 0
    assert c2.chunk_index == 2
    assert c2.total_chunks == 3
    assert c2.relative_position == round(2 / 3, 4)


# ==============================================================================
# 7. Deterministic Chunk Characteristics Tests
# ==============================================================================


def test_characteristics_derivation_prose():
    """Verify deterministic metrics for standard prose chunks."""
    service = DocumentMetadataEnrichmentService()
    content = "The system must process transactions within 200 milliseconds."
    chunk = make_chunk(
        "c_prose",
        content,
        element_types=(ElementType.PARAGRAPH,),
    )
    doc = make_chunked_doc([chunk])

    enriched = service.enrich_sync(doc).chunks[0].enriched_metadata
    assert enriched.character_count == len(content)
    assert enriched.word_count == 8
    assert enriched.line_count == 1
    assert enriched.content_type == ChunkContentType.PROSE.value
    assert enriched.has_code is False
    assert enriched.has_table is False
    assert enriched.has_list is False
    assert enriched.is_header_chunk is False
    assert enriched.language is None


def test_characteristics_derivation_code_block():
    """Verify deterministic metrics and language detection for code chunks."""
    service = DocumentMetadataEnrichmentService()
    content = "```python\ndef calculate_total(subtotal: float) -> float:\n    return subtotal * 1.2\n```"
    chunk = make_chunk(
        "c_code",
        content,
        element_types=(ElementType.CODE_BLOCK,),
    )
    doc = make_chunked_doc([chunk])

    enriched = service.enrich_sync(doc).chunks[0].enriched_metadata
    assert enriched.content_type == ChunkContentType.CODE.value
    assert enriched.has_code is True
    assert enriched.has_table is False
    assert enriched.language == "python"
    assert enriched.line_count == 4


def test_characteristics_derivation_table():
    """Verify deterministic metrics for table chunks."""
    service = DocumentMetadataEnrichmentService()
    content = "| Step | Role | Action |\n| :--- | :--- | :--- |\n| 1 | Admin | Approve |"
    chunk = make_chunk(
        "c_tbl",
        content,
        element_types=(ElementType.TABLE,),
    )
    doc = make_chunked_doc([chunk])

    enriched = service.enrich_sync(doc).chunks[0].enriched_metadata
    assert enriched.content_type == ChunkContentType.TABLE.value
    assert enriched.has_table is True
    assert enriched.has_code is False


def test_characteristics_derivation_list():
    """Verify deterministic metrics for list items."""
    service = DocumentMetadataEnrichmentService()
    content = "- Item 1\n- Item 2\n- Item 3"
    chunk = make_chunk(
        "c_list",
        content,
        element_types=(ElementType.LIST_ITEM,),
    )
    doc = make_chunked_doc([chunk])

    enriched = service.enrich_sync(doc).chunks[0].enriched_metadata
    assert enriched.content_type == ChunkContentType.LIST.value
    assert enriched.has_list is True


def test_characteristics_derivation_pure_heading():
    """Verify classification when chunk consists solely of a heading."""
    service = DocumentMetadataEnrichmentService()
    content = "## Architecture Decisions"
    chunk = make_chunk(
        "c_head",
        content,
        element_types=(ElementType.HEADING,),
    )
    doc = make_chunked_doc([chunk])

    enriched = service.enrich_sync(doc).chunks[0].enriched_metadata
    assert enriched.content_type == ChunkContentType.HEADING.value
    assert enriched.is_header_chunk is True


# ==============================================================================
# 8. Missing / Optional Metadata Graceful Handling Tests
# ==============================================================================


def test_missing_optional_metadata_handling():
    """Verify sensible behavior when optional metadata is absent."""
    service = DocumentMetadataEnrichmentService()

    chunk = make_chunk(
        "c_bare",
        "Minimal content.",
        version_id=None,
        section_path=(),
        heading=None,
        heading_level=None,
    )
    # Empty source and parser metadata
    doc = make_chunked_doc(
        [chunk],
        version_id=None,
        source_metadata={},
        parser_metadata={},
    )

    enriched_doc = service.enrich_sync(doc)
    meta = enriched_doc.chunks[0].enriched_metadata

    assert meta.document_version_id is None
    assert meta.section_path == ()
    assert meta.heading is None
    assert meta.heading_path_str == ""
    assert meta.source_storage_path is None
    assert meta.original_filename is None
    assert meta.source_metadata == {}


def test_empty_document_enrichment():
    """Verify graceful handling of an empty ChunkedDocument with 0 chunks."""
    service = DocumentMetadataEnrichmentService()
    doc = make_chunked_doc([], doc_id="empty-doc")

    enriched = service.enrich_sync(doc)
    assert isinstance(enriched, EnrichedDocument)
    assert enriched.total_chunks == 0
    assert enriched.chunks == []
    assert enriched.enrichment_report.total_chunks == 0
    assert enriched.enrichment_report.enriched_count == 0


# ==============================================================================
# 9. Retrieval Payload & PostgreSQL Serialization Tests
# ==============================================================================


def test_vector_payload_and_relational_serialization():
    """Verify to_vector_payload() and to_relational_record() produce expected structures."""
    service = DocumentMetadataEnrichmentService()
    chunk = make_chunk(
        "c1",
        "Retrievable chunk text.",
        index=0,
        section_path=("Sec1",),
        heading="Sec1",
        heading_level=1,
    )
    doc = make_chunked_doc([chunk])

    enriched_chunk = service.enrich_sync(doc).chunks[0]

    # Vector Payload for Qdrant
    payload = enriched_chunk.to_vector_payload()
    assert payload["project_id"] == "proj-alpha"
    assert payload["document_id"] == "doc-1"
    assert payload["chunk_id"] == "c1"
    assert payload["document_version_id"] == "v1.0"
    assert payload["section_path"] == ["Sec1"]
    assert payload["heading"] == "Sec1"
    assert payload["heading_path_str"] == "Sec1"
    assert payload["content_type"] == "prose"
    assert payload["character_count"] == len("Retrievable chunk text.")
    assert payload["word_count"] == 3

    # Relational Record for PostgreSQL
    rel_record = enriched_chunk.to_relational_record()
    assert rel_record["project_id"] == "proj-alpha"
    assert rel_record["document_id"] == "doc-1"
    assert rel_record["chunk_id"] == "c1"
    assert rel_record["index"] == 0
    assert "metadata" in rel_record
    assert "provenance" in rel_record["metadata"]
    assert "structural" in rel_record["metadata"]


# ==============================================================================
# 10. Extensibility via Custom Enricher Tests
# ==============================================================================


class MockHeuristicEnricher(BaseMetadataEnricher):
    """Mock heuristic enricher simulating entity or keyword tagging."""

    @property
    def enricher_name(self) -> str:
        return "mock_heuristic_enricher"

    def enrich(
        self,
        chunk: DocumentChunk,
        context: DocumentEnrichmentContext,
    ) -> dict[str, str]:
        tags = []
        if "SSO" in chunk.content:
            tags.append("authentication")
        if "Phase 1" in chunk.content:
            tags.append("roadmap")
        return {"heuristic_tags": ",".join(tags)} if tags else {}


def test_extensibility_with_custom_enricher():
    """Verify that custom enrichers can be registered without modifying core service."""
    custom_enricher = MockHeuristicEnricher()
    config = MetadataEnrichmentConfig(custom_enrichers=(custom_enricher,))
    service = DocumentMetadataEnrichmentService(config=config)

    chunk = make_chunk("c1", "Client approved SSO for Phase 1.")
    doc = make_chunked_doc([chunk])

    enriched_doc = service.enrich_sync(doc)
    meta = enriched_doc.chunks[0].enriched_metadata

    assert "mock_heuristic_enricher" in enriched_doc.enrichment_report.rules_executed
    assert meta.extra.get("heuristic_tags") == "authentication,roadmap"
    assert enriched_doc.chunks[0].metadata.get("heuristic_tags") == "authentication,roadmap"


# ==============================================================================
# 11. Idempotency Tests
# ==============================================================================


def test_enrichment_idempotency():
    """Verify that re-enriching an already enriched document yields identical results."""
    service = DocumentMetadataEnrichmentService()
    chunks = [
        make_chunk("c1", "First chunk content.", index=0, section_path=("Sec1",)),
        make_chunk("c2", "Second chunk content.", index=1, section_path=("Sec2",)),
    ]
    doc = make_chunked_doc(chunks)

    # First enrichment run
    enriched_1 = service.enrich_sync(doc)
    assert enriched_1.enrichment_report.enriched_count == 2

    # Second enrichment run on already enriched document
    enriched_2 = service.enrich_sync(enriched_1)
    assert enriched_2.enrichment_report.enriched_count == 2

    for c1, c2 in zip(enriched_1.chunks, enriched_2.chunks):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.project_id == c2.project_id
        assert c1.enriched_metadata.to_dict() == c2.enriched_metadata.to_dict()
        assert c1.to_vector_payload() == c2.to_vector_payload()


# ==============================================================================
# 12. Asynchronous Execution & Batch Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_async_enrichment_and_batch_processing():
    """Verify async enrich() and enrich_batch() execute non-blockingly."""
    service = get_metadata_enrichment_service()

    doc1 = make_chunked_doc(
        [make_chunk("c1", "Doc 1 text", project_id="p1")],
        doc_id="d1",
        project_id="p1",
    )
    doc2 = make_chunked_doc(
        [make_chunk("c2", "Doc 2 text", project_id="p2")],
        doc_id="d2",
        project_id="p2",
    )

    # Single async
    res1 = await service.enrich(doc1)
    assert res1.document_id == "d1"
    assert res1.total_chunks == 1

    # Batch async
    batch_res = await service.enrich_batch([doc1, doc2])
    assert len(batch_res) == 2
    assert batch_res[0].document_id == "d1"
    assert batch_res[0].project_id == "p1"
    assert batch_res[1].document_id == "d2"
    assert batch_res[1].project_id == "p2"


# ==============================================================================
# 13. End-to-End Pipeline Integration Test
# (Ingestion -> Parsing -> Cleaning -> Normalization -> Chunking -> Metadata Enrichment)
# ==============================================================================


def test_full_pipeline_indexing_integration():
    """Verify seamless end-to-end indexing flow through Metadata Enrichment."""
    from cleaning import DocumentCleaningService
    from chunking import DocumentChunkingService
    from ingestion.models import IngestedDocument
    from normalization import DocumentNormalizationService
    from parsing.markdown import MarkdownParser

    raw_markdown = (
        b"# BRD: User Notification Engine\n\n"
        b"Page 1 of 5\n\n"
        b"## Functional Requirements\n\n"
        b"The notification engine must dispatch SMS alerts within 5 seconds.\n\n"
        b"```json\n"
        b"{\"timeout\": 5000}\n"
        b"```\n\n"
        b"## Performance SLA\n\n"
        b"System must achieve 99.99% monthly availability."
    )

    # 1. Ingestion
    ingested = IngestedDocument(
        document_id="brd-notif-001",
        project_id="proj-finance",
        source_storage_path="proj-finance/specs/brd.md",
        original_filename="brd.md",
        detected_document_type=DocumentType.MARKDOWN,
        content_type="text/markdown",
        raw_bytes=raw_markdown,
        document_version_id="v2.1",
    )

    # 2. Parsing
    parser = MarkdownParser()
    parsed = parser.parse(ingested)
    assert parsed.total_elements > 0

    # 3. Cleaning (removes 'Page 1 of 5')
    cleaned = DocumentCleaningService().clean_sync(parsed)
    assert cleaned.cleaning_report.removed_count > 0

    # 4. Normalization
    normalized = DocumentNormalizationService().normalize_sync(cleaned)
    assert normalized.normalization_report is not None

    # 5. Chunking
    chunking_svc = DocumentChunkingService(config=ChunkingConfig(max_chunk_size=300))
    chunked = chunking_svc.chunk_sync(normalized)
    assert isinstance(chunked, ChunkedDocument)
    assert chunked.total_chunks > 0

    # 6. Metadata Enrichment
    enrichment_svc = DocumentMetadataEnrichmentService()
    enriched = enrichment_svc.enrich_sync(chunked)

    assert isinstance(enriched, EnrichedDocument)
    assert enriched.document_id == "brd-notif-001"
    assert enriched.project_id == "proj-finance"
    assert enriched.document_version_id == "v2.1"
    assert enriched.total_chunks == chunked.total_chunks
    assert enriched.enrichment_report.enriched_count == enriched.total_chunks

    # Validate all chunks have preserved content, provenance, structure, and characteristics
    for chunk in enriched.chunks:
        assert isinstance(chunk, EnrichedChunk)
        assert chunk.project_id == "proj-finance"
        assert chunk.document_id == "brd-notif-001"
        assert chunk.document_version_id == "v2.1"
        assert chunk.enriched_metadata is not None
        assert chunk.enriched_metadata.character_count == len(chunk.content)
        assert chunk.enriched_metadata.word_count == len(chunk.content.split())

        # Check Qdrant payload serialization
        payload = chunk.to_vector_payload()
        assert payload["project_id"] == "proj-finance"
        assert payload["document_id"] == "brd-notif-001"
        assert payload["chunk_id"] == chunk.chunk_id
        assert "content_type" in payload
