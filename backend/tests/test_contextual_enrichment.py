"""Comprehensive unit and integration tests for Document Contextual Enrichment."""

import pytest

from acquisition.formats import DocumentType
from app.core.config import Settings
from chunking.models import ChunkingReport, DocumentChunk
from cleaning.models import CleaningReport
from contextual_enrichment import (
    BaseContextProvider,
    ContextualEnrichmentConfig,
    ContextualEnrichmentReport,
    ContextuallyEnrichedChunk,
    ContextuallyEnrichedDocument,
    DocumentContext,
    DocumentContextualEnrichmentService,
    StructuredContextProvider,
    get_contextual_enrichment_service,
    reset_contextual_enrichment_service,
)
from exceptions.contextual_enrichment import (
    ContextualEnrichmentConfigurationError,
    ContextualEnrichmentError,
    ContextualEnrichmentProcessingError,
    ContextualEnrichmentProviderError,
    InvalidContextualEnrichmentInputError,
)
from metadata_enrichment.models import (
    ChunkMetadata,
    EnrichedChunk,
    EnrichedDocument,
    MetadataEnrichmentReport,
)
from normalization.models import NormalizationReport
from parsing.models import ElementType


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_contextual_enrichment_service()
    yield
    reset_contextual_enrichment_service()


def make_enriched_chunk(
    chunk_id: str,
    content: str,
    index: int = 0,
    doc_id: str = "doc-1",
    project_id: str = "proj-alpha",
    version_id: str | None = "v1.0",
    section_path: tuple[str, ...] = ("Requirements", "Auth"),
    heading: str | None = "JWT Validation",
    heading_level: int | None = 2,
    heading_path_str: str = "Requirements > Auth",
) -> EnrichedChunk:
    """Helper to construct an EnrichedChunk test fixture."""
    meta = ChunkMetadata(
        project_id=project_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        document_version_id=version_id,
        source_element_ids=("elem-1",),
        document_type="markdown",
        section_path=section_path,
        heading=heading,
        heading_level=heading_level,
        heading_path_str=heading_path_str,
        hierarchy_depth=len(section_path),
        chunk_index=index,
        total_chunks=1,
        character_count=len(content),
        word_count=len(content.split()),
    )
    return EnrichedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        project_id=project_id,
        content=content,
        index=index,
        document_version_id=version_id,
        section_path=section_path,
        heading=heading,
        heading_level=heading_level,
        source_element_ids=("elem-1",),
        element_types=(ElementType.PARAGRAPH,),
        metadata={"custom_key": "val"},
        enriched_metadata=meta,
    )


def make_enriched_doc(
    chunks: list[EnrichedChunk],
    doc_id: str = "doc-1",
    project_id: str = "proj-alpha",
    doc_type: DocumentType = DocumentType.MARKDOWN,
    original_filename: str = "auth_spec.md",
) -> EnrichedDocument:
    """Helper to construct an EnrichedDocument test fixture."""
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
    meta_rep = MetadataEnrichmentReport(
        total_chunks=len(chunks),
        enriched_count=len(chunks),
        rules_executed=["provenance", "structural"],
    )
    return EnrichedDocument(
        document_id=doc_id,
        project_id=project_id,
        document_type=doc_type,
        chunks=chunks,
        document_version_id="v1.0",
        source_metadata={
            "source_storage_path": f"{project_id}/specs/{original_filename}",
            "original_filename": original_filename,
            "content_type": "text/markdown",
        },
        parser_metadata={"parser": "markdown"},
        cleaning_report=cleaning_rep,
        normalization_report=norm_rep,
        chunking_report=chunking_rep,
        enrichment_report=meta_rep,
    )


# ==============================================================================
# 1. Configuration & Default Behavior Tests
# ==============================================================================


def test_default_config_is_disabled():
    """Verify default configuration has contextual enrichment disabled."""
    config = ContextualEnrichmentConfig()
    assert config.enabled is False
    assert config.strategy == "structured"


def test_config_from_settings(monkeypatch):
    """Verify config derives from application Settings / environment."""
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_STRATEGY", "structured")

    settings = Settings()
    config = ContextualEnrichmentConfig.from_settings(settings)
    assert config.enabled is True
    assert config.strategy == "structured"


def test_invalid_config_raises():
    """Verify invalid strategy raises ContextualEnrichmentConfigurationError."""
    with pytest.raises(ContextualEnrichmentConfigurationError):
        ContextualEnrichmentConfig(strategy="invalid_strat")


# ==============================================================================
# 2. Document-Level Scope & Disabled Path Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_disabled_enrichment_skips_processing():
    """Verify that when disabled, enrichment is skipped and chunks are untouched."""
    chunk = make_enriched_chunk("c-1", "The token expires after 60 minutes.")
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=False)
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)

    assert isinstance(result, ContextuallyEnrichedDocument)
    assert len(result.chunks) == 1
    out_chunk = result.chunks[0]

    # Contextual fields reflect disabled state
    assert out_chunk.is_contextually_enriched is False
    assert out_chunk.context_text is None
    assert out_chunk.context_strategy is None
    assert out_chunk.contextual_content == "The token expires after 60 minutes."
    assert out_chunk.to_embedding_text() == "The token expires after 60 minutes."

    # Report reflects disabled state
    assert result.contextual_enrichment_report.enabled is False
    assert result.contextual_enrichment_report.enriched_chunks_count == 0
    assert result.contextual_enrichment_report.strategy_used == "disabled"


@pytest.mark.asyncio
async def test_enabled_enrichment_processes_document_consistently():
    """Verify that when enabled, all chunks in document are processed consistently."""
    chunk1 = make_enriched_chunk("c-1", "Token expiration is 60 min.", index=0)
    chunk2 = make_enriched_chunk("c-2", "Refresh token is valid 30 days.", index=1)
    doc = make_enriched_doc([chunk1, chunk2])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)

    assert result.contextual_enrichment_report.enabled is True
    assert result.contextual_enrichment_report.total_chunks == 2
    assert result.contextual_enrichment_report.enriched_chunks_count == 2
    assert result.contextual_enrichment_report.strategy_used == "structured"

    # Both chunks are enriched
    for c in result.chunks:
        assert c.is_contextually_enriched is True
        assert c.context_text is not None
        assert "Document: auth_spec.md" in c.context_text
        assert "Section: Requirements > Auth" in c.context_text


# ==============================================================================
# 3. Content & Provenance Preservation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_verbatim_content_preservation():
    """Verify original chunk content is preserved strictly verbatim."""
    original_text = "The service retries failed requests three times using exponential backoff."
    chunk = make_enriched_chunk("c-1", original_text)
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)
    enriched_chunk = result.chunks[0]

    # chunk.content MUST remain identical verbatim
    assert enriched_chunk.content == original_text
    # context is stored separately
    assert enriched_chunk.context_text is not None
    # representation for embedding combines them
    expected_embedding_text = f"{enriched_chunk.context_text}\n\n{original_text}"
    assert enriched_chunk.contextual_content == expected_embedding_text
    assert enriched_chunk.to_embedding_text() == expected_embedding_text


@pytest.mark.asyncio
async def test_provenance_and_hierarchy_preservation():
    """Verify all provenance, structural hierarchy, and previous reports are preserved."""
    chunk = make_enriched_chunk(
        "c-99",
        "Sensitive payload",
        index=3,
        doc_id="doc-xyz",
        project_id="tenant-1",
        version_id="v2.5",
        section_path=("A", "B", "C"),
        heading="Deep Section",
    )
    doc = make_enriched_doc([chunk], doc_id="doc-xyz", project_id="tenant-1")

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)
    c = result.chunks[0]

    assert c.chunk_id == "c-99"
    assert c.document_id == "doc-xyz"
    assert c.project_id == "tenant-1"
    assert c.document_version_id == "v2.5"
    assert c.index == 3
    assert c.section_path == ("A", "B", "C")
    assert c.heading == "Deep Section"

    # Previous reports preserved
    assert result.cleaning_report is not None
    assert result.normalization_report is not None
    assert result.chunking_report is not None
    assert result.enrichment_report is not None


# ==============================================================================
# 4. Output Representations (Vector Payload & Relational Records)
# ==============================================================================


# ==============================================================================
# 4. Output Representations (Dictionary Serialization & Embedding Representation)
# ==============================================================================


@pytest.mark.asyncio
async def test_to_dict_and_embedding_text_representation():
    """Verify to_dict() and to_embedding_text() expose contextual representations without modifying source content."""
    chunk = make_enriched_chunk("c-1", "Original chunk content")
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)
    enriched_chunk = result.chunks[0]

    # Content preserved verbatim
    assert enriched_chunk.content == "Original chunk content"
    assert enriched_chunk.is_contextually_enriched is True
    assert "Document: auth_spec.md" in enriched_chunk.context_text

    # Embedding representation combines context and verbatim content
    assert enriched_chunk.to_embedding_text() == enriched_chunk.contextual_content
    assert "Document: auth_spec.md" in enriched_chunk.to_embedding_text()
    assert "Original chunk content" in enriched_chunk.to_embedding_text()

    # Dictionary serialization contains clean domain attributes
    data = enriched_chunk.to_dict()
    assert data["content"] == "Original chunk content"
    assert data["is_contextually_enriched"] is True
    assert data["context_text"] == enriched_chunk.context_text
    assert data["context_strategy"] == "structured"
    assert data["contextual_content"] == enriched_chunk.contextual_content
    assert data["project_id"] == "proj-alpha"
    assert data["document_id"] == "doc-1"


# ==============================================================================
# 5. Project-Level Tenant Isolation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_project_isolation_mismatch_raises_error():
    """Verify that any chunk with a mismatched project_id raises ContextualEnrichmentProcessingError."""
    chunk_legit = make_enriched_chunk("c-1", "Legit chunk", project_id="proj-A")
    chunk_rogue = make_enriched_chunk("c-2", "Rogue chunk", project_id="proj-B")
    doc = make_enriched_doc([chunk_legit, chunk_rogue], project_id="proj-A")

    service = DocumentContextualEnrichmentService(
        config=ContextualEnrichmentConfig(enabled=True)
    )

    with pytest.raises(ContextualEnrichmentProcessingError) as exc_info:
        await service.enrich(doc)
    assert "Project isolation violation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_batch_enrichment_isolates_projects():
    """Verify batch enrichment processes multiple documents without cross-project leakage."""
    chunk_a = make_enriched_chunk("ca-1", "Project A content", project_id="proj-A", doc_id="doc-A")
    chunk_b = make_enriched_chunk("cb-1", "Project B content", project_id="proj-B", doc_id="doc-B")

    doc_a = make_enriched_doc([chunk_a], doc_id="doc-A", project_id="proj-A", original_filename="docA.md")
    doc_b = make_enriched_doc([chunk_b], doc_id="doc-B", project_id="proj-B", original_filename="docB.md")

    service = DocumentContextualEnrichmentService(
        config=ContextualEnrichmentConfig(enabled=True, strategy="structured")
    )

    results = await service.enrich_batch([doc_a, doc_b])

    assert len(results) == 2
    assert results[0].project_id == "proj-A"
    assert "Document: docA.md" in results[0].chunks[0].context_text
    assert "docB.md" not in results[0].chunks[0].context_text

    assert results[1].project_id == "proj-B"
    assert "Document: docB.md" in results[1].chunks[0].context_text
    assert "docA.md" not in results[1].chunks[0].context_text


# ==============================================================================
# 6. Missing Context & Minimal Documents Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_missing_context_does_not_invent():
    """Verify minimal document with missing headings and no filename does not invent context."""
    # Chunk with no section path, no heading, no title
    chunk = EnrichedChunk(
        chunk_id="c-raw",
        document_id="doc-minimal",
        project_id="proj-alpha",
        content="Standalone note.",
        index=0,
        section_path=(),
        heading=None,
    )
    doc = EnrichedDocument(
        document_id="doc-minimal",
        project_id="proj-alpha",
        document_type=DocumentType.TXT,
        chunks=[chunk],
        source_metadata={},  # No original_filename
    )

    service = DocumentContextualEnrichmentService(
        config=ContextualEnrichmentConfig(enabled=True, strategy="structured")
    )

    result = await service.enrich(doc)
    c = result.chunks[0]

    # Must NOT invent fake headings or placeholders
    assert c.is_contextually_enriched is False
    assert c.context_text is None
    assert c.contextual_content == "Standalone note."


# ==============================================================================
# 7. Provider Abstraction & Custom Provider Extension Tests
# ==============================================================================


class CustomMockContextProvider(BaseContextProvider):
    """Custom context provider to verify BaseContextProvider extension point."""

    @property
    def provider_name(self) -> str:
        return "custom_mock"

    async def generate_context(
        self,
        chunk: EnrichedChunk,
        context: DocumentContext,
    ) -> str | None:
        return f"Custom Context for {chunk.chunk_id}"


@pytest.mark.asyncio
async def test_custom_provider_extension():
    """Verify that BaseContextProvider can be implemented and injected seamlessly."""
    custom_provider = CustomMockContextProvider()
    config = ContextualEnrichmentConfig(
        enabled=True,
        strategy="custom",
        provider=custom_provider,
    )
    service = DocumentContextualEnrichmentService(config=config)

    chunk = make_enriched_chunk("c-custom", "Custom chunk content")
    doc = make_enriched_doc([chunk])

    result = await service.enrich(doc)
    c = result.chunks[0]

    assert c.is_contextually_enriched is True
    assert c.context_text == "Custom Context for c-custom"
    assert c.contextual_content == "Custom Context for c-custom\n\nCustom chunk content"
    assert c.to_embedding_text() == c.contextual_content
    assert result.contextual_enrichment_report.strategy_used == "custom_mock"
    assert result.contextual_enrichment_report.enriched_chunks_count == 1


# ==============================================================================
# 8. Sync API Execution & Edge Cases Tests
# ==============================================================================


def test_enrich_sync_execution():
    """Verify enrich_sync executes cleanly for synchronous callers."""
    chunk = make_enriched_chunk("c-1", "Sync chunk content")
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = service.enrich_sync(doc)
    assert isinstance(result, ContextuallyEnrichedDocument)
    assert result.chunks[0].is_contextually_enriched is True


def test_invalid_input_type_raises():
    """Verify passing a non-EnrichedDocument raises InvalidContextualEnrichmentInputError."""
    service = DocumentContextualEnrichmentService()
    with pytest.raises(InvalidContextualEnrichmentInputError):
        service.enrich_sync("not a document")  # type: ignore


@pytest.mark.asyncio
async def test_empty_document_handling():
    """Verify empty document with 0 chunks is handled safely."""
    doc = make_enriched_doc([])
    service = DocumentContextualEnrichmentService(
        config=ContextualEnrichmentConfig(enabled=True, strategy="structured")
    )
    result = await service.enrich(doc)

    assert result.total_chunks == 0
    assert result.contextual_enrichment_report.enriched_chunks_count == 0
    assert result.contextual_enrichment_report.total_chunks == 0


def test_singleton_management():
    """Verify get_contextual_enrichment_service and reset work as expected."""
    svc1 = get_contextual_enrichment_service()
    svc2 = get_contextual_enrichment_service()
    assert svc1 is svc2

    reset_contextual_enrichment_service()
    svc3 = get_contextual_enrichment_service()
    assert svc3 is not svc1
