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
    LLMContextProvider,
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
from llm.base import BaseLLMClient, LLMProviderError
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


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for test isolation without external network calls."""

    def __init__(
        self,
        response_text: str = "This chunk describes token expiration rules.",
        fail_error: Exception | None = None,
        model_name: str = "mock-gpt-4o",
    ) -> None:
        self._response_text = response_text
        self._fail_error = fail_error
        self._model_name = model_name
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if self._fail_error:
            raise self._fail_error
        return self._response_text


# ==============================================================================
# 1. Configuration & Default Behavior Tests
# ==============================================================================


def test_default_config_is_disabled():
    """Verify default configuration has contextual enrichment disabled."""
    config = ContextualEnrichmentConfig()
    assert config.enabled is False
    assert config.strategy == "structured"
    assert config.max_concurrency == 5


def test_config_from_settings(monkeypatch):
    """Verify config derives from application Settings / environment."""
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_STRATEGY", "structured")
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_MAX_CONCURRENCY", "10")

    settings = Settings()
    config = ContextualEnrichmentConfig.from_settings(settings)
    assert config.enabled is True
    assert config.strategy == "structured"
    assert config.max_concurrency == 10


def test_invalid_config_raises():
    """Verify invalid strategy or concurrency raises ContextualEnrichmentConfigurationError."""
    with pytest.raises(ContextualEnrichmentConfigurationError):
        ContextualEnrichmentConfig(strategy="invalid_strat")

    with pytest.raises(ContextualEnrichmentConfigurationError):
        ContextualEnrichmentConfig(max_concurrency=0)


def test_llm_strategy_without_provider_raises():
    """Verify configuring strategy='llm' without passing provider raises configuration error."""
    config = ContextualEnrichmentConfig(enabled=True, strategy="llm", provider=None)
    with pytest.raises(ContextualEnrichmentConfigurationError) as exc_info:
        DocumentContextualEnrichmentService(config=config)
    assert "LLM strategy requires an explicit LLM provider" in str(exc_info.value)


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


@pytest.mark.asyncio
async def test_to_vector_payload_includes_original_and_context():
    """Verify vector payload includes original content and contextual metadata."""
    chunk = make_enriched_chunk("c-1", "Original chunk content")
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)
    payload = result.chunks[0].to_vector_payload()

    assert payload["content"] == "Original chunk content"
    assert payload["is_contextually_enriched"] is True
    assert "Document: auth_spec.md" in payload["context_text"]
    assert payload["project_id"] == "proj-alpha"
    assert payload["document_id"] == "doc-1"


@pytest.mark.asyncio
async def test_to_relational_record_includes_context_metadata():
    """Verify relational record retains context text and audit flags."""
    chunk = make_enriched_chunk("c-1", "Original chunk content")
    doc = make_enriched_doc([chunk])

    config = ContextualEnrichmentConfig(enabled=True, strategy="structured")
    service = DocumentContextualEnrichmentService(config=config)

    result = await service.enrich(doc)
    record = result.chunks[0].to_relational_record()

    assert record["context_text"] is not None
    assert record["is_contextually_enriched"] is True
    assert "contextual" in record["metadata"]


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
# 7. LLM Provider Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_llm_provider_enrichment():
    """Verify LLM provider formats prompt, invokes client, and attaches context."""
    mock_client = MockLLMClient(response_text="This chunk describes user session timeouts.")
    llm_provider = LLMContextProvider(llm_client=mock_client)

    config = ContextualEnrichmentConfig(
        enabled=True,
        strategy="custom",
        provider=llm_provider,
    )
    service = DocumentContextualEnrichmentService(config=config)

    chunk = make_enriched_chunk("c-1", "Sessions expire after 15 minutes of inactivity.")
    doc = make_enriched_doc([chunk])

    result = await service.enrich(doc)

    assert mock_client.call_count == 1
    assert "Sessions expire after 15 minutes" in mock_client.last_prompt
    assert "auth_spec.md" in mock_client.last_prompt

    c = result.chunks[0]
    assert c.is_contextually_enriched is True
    assert c.context_text == "This chunk describes user session timeouts."
    assert "This chunk describes user session timeouts.\n\nSessions expire" in c.contextual_content


@pytest.mark.asyncio
async def test_llm_provider_handles_empty_response():
    """Verify LLM returning NONE or whitespace results in None context."""
    mock_client = MockLLMClient(response_text="NONE")
    llm_provider = LLMContextProvider(llm_client=mock_client)

    config = ContextualEnrichmentConfig(enabled=True, provider=llm_provider)
    service = DocumentContextualEnrichmentService(config=config)

    chunk = make_enriched_chunk("c-1", "Self contained statement.")
    doc = make_enriched_doc([chunk])

    result = await service.enrich(doc)
    c = result.chunks[0]

    assert c.is_contextually_enriched is False
    assert c.context_text is None
    assert c.contextual_content == "Self contained statement."


@pytest.mark.asyncio
async def test_llm_provider_failure_translates_to_domain_exception():
    """Verify LLM provider error raises ContextualEnrichmentProviderError."""
    mock_client = MockLLMClient(fail_error=LLMProviderError("Rate limit exceeded"))
    llm_provider = LLMContextProvider(llm_client=mock_client)

    config = ContextualEnrichmentConfig(enabled=True, provider=llm_provider)
    service = DocumentContextualEnrichmentService(config=config)

    chunk = make_enriched_chunk("c-1", "Content")
    doc = make_enriched_doc([chunk])

    with pytest.raises(ContextualEnrichmentProviderError) as exc_info:
        await service.enrich(doc)
    assert "Rate limit exceeded" in str(exc_info.value)


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
