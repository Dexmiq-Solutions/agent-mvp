"""Unit tests for the Ingestion layer of the document indexing pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from rag.acquisition.formats import DocumentType
from rag.acquisition.models import SourceDocument, SourceReference
from core.config import Settings
from exceptions.ingestion import (
    DocumentRetrievalError,
    IngestionConfigurationError,
    IngestionError,
    InvalidIngestionInputError,
    UnsupportedDocumentTypeError,
)
from exceptions.storage import (
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageDownloadError,
    StorageError,
)
from rag.ingestion import (
    BaseIngestionService,
    DocumentSourceReference,
    IngestedDocument,
    StorageIngestionService,
    get_ingestion_service,
    reset_ingestion_service,
    resolve_and_validate_document_type,
    resolve_and_verify_storage_path,
    validate_project_id,
)
from storage.object import BaseObjectStorage


@pytest.fixture(autouse=True)
def cleanup_ingestion_service():
    """Reset singleton ingestion service before and after each test."""
    reset_ingestion_service()
    yield
    reset_ingestion_service()


@pytest.fixture
def mock_storage():
    """Create a mock BaseObjectStorage instance."""
    storage = MagicMock(spec=BaseObjectStorage)
    storage.bucket_name = "test-documents"
    storage.download = AsyncMock()
    return storage


# ==============================================================================
# 1. Validation & Document Type Resolution Tests
# ==============================================================================


def test_validate_project_id():
    """Verify project ID validation enforces strict tenant constraints."""
    assert validate_project_id("proj-123") == "proj-123"
    assert validate_project_id("  tenant-abc  ") == "tenant-abc"
    assert validate_project_id("/proj-slash/") == "proj-slash"

    # Invalid cases
    with pytest.raises(InvalidIngestionInputError, match="non-empty string"):
        validate_project_id("")
    with pytest.raises(InvalidIngestionInputError, match="non-empty string"):
        validate_project_id(None)
    with pytest.raises(InvalidIngestionInputError, match="single path segment"):
        validate_project_id("proj/subfolder")
    with pytest.raises(InvalidIngestionInputError, match="single path segment"):
        validate_project_id("../proj")


def test_resolve_and_verify_storage_path():
    """Verify storage path normalization and project tenant isolation."""
    # Already prefixed path
    assert resolve_and_verify_storage_path("proj-1", "proj-1/docs/report.pdf") == "proj-1/docs/report.pdf"

    # Unprefixed relative filename is scoped under project_id
    assert resolve_and_verify_storage_path("proj-1", "report.pdf") == "proj-1/report.pdf"

    # Cross-project violation
    with pytest.raises(InvalidIngestionInputError, match="Cross-project access prohibited"):
        resolve_and_verify_storage_path("proj-1", "proj-2/other.pdf")

    # Project root reference
    with pytest.raises(InvalidIngestionInputError, match="points to project root"):
        resolve_and_verify_storage_path("proj-1", "proj-1")

    # Path traversal
    with pytest.raises(InvalidIngestionInputError, match="Path traversal not permitted"):
        resolve_and_verify_storage_path("proj-1", "../secret.txt")


def test_resolve_and_validate_document_type_mime():
    """Verify document type detection from explicit content type / MIME."""
    # PDF
    doc_type, mime = resolve_and_validate_document_type(content_type="application/pdf")
    assert doc_type == DocumentType.PDF
    assert mime == "application/pdf"

    # DOCX
    doc_type, mime = resolve_and_validate_document_type(
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert doc_type == DocumentType.DOCX

    # TXT
    doc_type, mime = resolve_and_validate_document_type(content_type="text/plain")
    assert doc_type == DocumentType.TXT
    assert mime == "text/plain"

    # Markdown via MIME
    doc_type, mime = resolve_and_validate_document_type(content_type="text/markdown")
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"


def test_resolve_and_validate_document_type_mime_parameters():
    """Verify MIME parameter stripping (e.g., charset=utf-8)."""
    doc_type, mime = resolve_and_validate_document_type(content_type="text/plain; charset=utf-8")
    assert doc_type == DocumentType.TXT
    assert mime == "text/plain"


def test_resolve_and_validate_document_type_text_plain_markdown_filename():
    """Verify that text/plain with .md or .markdown filename resolves to Markdown."""
    doc_type, mime = resolve_and_validate_document_type(
        content_type="text/plain",
        filename_or_path="notes.md",
    )
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"

    doc_type, mime = resolve_and_validate_document_type(
        content_type="text/plain",
        filename_or_path="readme.markdown",
    )
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"

    # Standard txt stays TXT
    doc_type, mime = resolve_and_validate_document_type(
        content_type="text/plain",
        filename_or_path="notes.txt",
    )
    assert doc_type == DocumentType.TXT
    assert mime == "text/plain"


def test_resolve_and_validate_document_type_extension_fallback():
    """Verify fallback to file extension when MIME is generic octet-stream or missing."""
    # Generic octet-stream
    doc_type, mime = resolve_and_validate_document_type(
        content_type="application/octet-stream",
        filename_or_path="contract.pdf",
    )
    assert doc_type == DocumentType.PDF
    assert mime == "application/pdf"

    # None content type
    doc_type, mime = resolve_and_validate_document_type(
        content_type=None,
        filename_or_path="specs.docx",
    )
    assert doc_type == DocumentType.DOCX
    assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # Markdown extension
    doc_type, mime = resolve_and_validate_document_type(
        content_type="",
        filename_or_path="guide.markdown",
    )
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"


def test_resolve_and_validate_document_type_explicit():
    """Verify explicit DocumentType override handling."""
    doc_type, mime = resolve_and_validate_document_type(explicit_type=DocumentType.PDF)
    assert doc_type == DocumentType.PDF
    assert mime == "application/pdf"

    doc_type, mime = resolve_and_validate_document_type(explicit_type="markdown")
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported explicit document type"):
        resolve_and_validate_document_type(explicit_type="csv")


def test_resolve_and_validate_document_type_unsupported():
    """Verify that unsupported document types and extensions fail with UnsupportedDocumentTypeError."""
    # Unsupported extension
    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported file format"):
        resolve_and_validate_document_type(filename_or_path="table.csv")

    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported file format"):
        resolve_and_validate_document_type(filename_or_path="photo.png")

    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported file format"):
        resolve_and_validate_document_type(filename_or_path="bundle.zip")

    # Unsupported explicit MIME
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported content type"):
        resolve_and_validate_document_type(content_type="image/png")

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported content type"):
        resolve_and_validate_document_type(content_type="application/json")

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported content type"):
        resolve_and_validate_document_type(content_type="text/csv")

    # Neither provided
    with pytest.raises(UnsupportedDocumentTypeError, match="neither a supported content type nor filename"):
        resolve_and_validate_document_type(content_type=None, filename_or_path=None)


# ==============================================================================
# 2. Ingestion Service Execution Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ingest_pdf_from_source_document(mock_storage):
    """Verify successful ingestion of a PDF from a SourceDocument reference."""
    pdf_bytes = b"%PDF-1.4 sample pdf binary data"
    mock_storage.download.return_value = pdf_bytes

    service = StorageIngestionService(storage=mock_storage)

    source_doc = SourceDocument(
        source_id="test-documents/proj-100/manual.pdf",
        project_id="proj-100",
        filename="manual.pdf",
        storage_path="proj-100/manual.pdf",
        storage_bucket="test-documents",
        document_type=DocumentType.PDF,
        content_type="application/pdf",
        extension=".pdf",
        size_bytes=len(pdf_bytes),
        metadata={"etag": "etag-12345", "author": "Engineering"},
    )

    ingested = await service.ingest(source_doc)

    assert isinstance(ingested, IngestedDocument)
    assert ingested.document_id == "test-documents/proj-100/manual.pdf"
    assert ingested.project_id == "proj-100"
    assert ingested.source_storage_path == "proj-100/manual.pdf"
    assert ingested.original_filename == "manual.pdf"
    assert ingested.detected_document_type == DocumentType.PDF
    assert ingested.content_type == "application/pdf"
    assert ingested.raw_bytes == pdf_bytes
    assert ingested.size_bytes == len(pdf_bytes)
    assert ingested.document_version_id == "etag-12345"
    assert ingested.source_metadata["author"] == "Engineering"
    assert ingested.source_metadata["storage_bucket"] == "test-documents"

    mock_storage.download.assert_awaited_once_with("proj-100/manual.pdf")


@pytest.mark.asyncio
async def test_ingest_docx(mock_storage):
    """Verify successful ingestion of a DOCX file."""
    docx_bytes = b"PK\x03\x04 fake docx binary data"
    mock_storage.download.return_value = docx_bytes

    service = StorageIngestionService(storage=mock_storage)

    ref = DocumentSourceReference(
        project_id="proj-200",
        storage_path="proj-200/specs/requirements.docx",
        document_version_id="v2",
        source_metadata={"classification": "internal"},
    )

    ingested = await service.ingest(ref)

    assert ingested.project_id == "proj-200"
    assert ingested.source_storage_path == "proj-200/specs/requirements.docx"
    assert ingested.original_filename == "requirements.docx"
    assert ingested.detected_document_type == DocumentType.DOCX
    assert ingested.raw_bytes == docx_bytes
    assert ingested.size_bytes == len(docx_bytes)
    assert ingested.document_version_id == "v2"
    assert ingested.source_metadata["classification"] == "internal"

    mock_storage.download.assert_awaited_once_with("proj-200/specs/requirements.docx")


@pytest.mark.asyncio
async def test_ingest_txt(mock_storage):
    """Verify successful ingestion of a plain text file."""
    txt_bytes = b"Sample plain text notes"
    mock_storage.download.return_value = txt_bytes

    service = StorageIngestionService(storage=mock_storage)

    ingested = await service.ingest("notes.txt", project_id="proj-300")

    assert ingested.project_id == "proj-300"
    assert ingested.source_storage_path == "proj-300/notes.txt"
    assert ingested.original_filename == "notes.txt"
    assert ingested.detected_document_type == DocumentType.TXT
    assert ingested.content_type == "text/plain"
    assert ingested.raw_bytes == txt_bytes
    assert ingested.size_bytes == len(txt_bytes)

    mock_storage.download.assert_awaited_once_with("proj-300/notes.txt")


@pytest.mark.asyncio
async def test_ingest_markdown(mock_storage):
    """Verify successful ingestion of a Markdown file."""
    md_bytes = b"# Architecture\n\nThis is the markdown document."
    mock_storage.download.return_value = md_bytes

    service = StorageIngestionService(storage=mock_storage)

    source_ref = SourceReference(
        bucket="test-documents",
        path="proj-400/docs/readme.md",
        etag="md-etag-999",
    )

    ingested = await service.ingest(source_ref, project_id="proj-400")

    assert ingested.project_id == "proj-400"
    assert ingested.source_storage_path == "proj-400/docs/readme.md"
    assert ingested.original_filename == "readme.md"
    assert ingested.detected_document_type == DocumentType.MARKDOWN
    assert ingested.content_type == "text/markdown"
    assert ingested.raw_bytes == md_bytes
    assert ingested.size_bytes == len(md_bytes)
    assert ingested.document_version_id == "md-etag-999"

    mock_storage.download.assert_awaited_once_with("proj-400/docs/readme.md")


# ==============================================================================
# 3. Error Handling & Storage Failure Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ingest_object_not_found(mock_storage):
    """Verify ObjectNotFoundError is translated to DocumentRetrievalError."""
    mock_storage.download.side_effect = ObjectNotFoundError("Object does not exist")

    service = StorageIngestionService(storage=mock_storage)

    with pytest.raises(DocumentRetrievalError, match="Document not found in storage"):
        await service.ingest("proj-1/missing.pdf", project_id="proj-1")


@pytest.mark.asyncio
async def test_ingest_storage_authentication_error(mock_storage):
    """Verify StorageAuthenticationError is translated to DocumentRetrievalError."""
    mock_storage.download.side_effect = StorageAuthenticationError("Unauthorized key")

    service = StorageIngestionService(storage=mock_storage)

    with pytest.raises(DocumentRetrievalError, match="Authentication failed downloading"):
        await service.ingest("proj-1/doc.pdf", project_id="proj-1")


@pytest.mark.asyncio
async def test_ingest_storage_download_error(mock_storage):
    """Verify StorageDownloadError is translated to DocumentRetrievalError."""
    mock_storage.download.side_effect = StorageDownloadError("Network stream interrupted")

    service = StorageIngestionService(storage=mock_storage)

    with pytest.raises(DocumentRetrievalError, match="Failed to download document"):
        await service.ingest("proj-1/doc.pdf", project_id="proj-1")


@pytest.mark.asyncio
async def test_ingest_unexpected_storage_error(mock_storage):
    """Verify unexpected errors during download are translated to DocumentRetrievalError."""
    mock_storage.download.side_effect = RuntimeError("Socket timeout")

    service = StorageIngestionService(storage=mock_storage)

    with pytest.raises(DocumentRetrievalError, match="Unexpected error retrieving"):
        await service.ingest("proj-1/doc.pdf", project_id="proj-1")


@pytest.mark.asyncio
async def test_ingest_invalid_payload_type(mock_storage):
    """Verify non-bytes return from download raises DocumentRetrievalError."""
    mock_storage.download.return_value = "string instead of bytes"

    service = StorageIngestionService(storage=mock_storage)

    with pytest.raises(DocumentRetrievalError, match="Invalid storage payload received"):
        await service.ingest("proj-1/doc.pdf", project_id="proj-1")


# ==============================================================================
# 4. Input Validation & Tenant Protection Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ingest_none_source(mock_storage):
    """Verify None source raises InvalidIngestionInputError."""
    service = StorageIngestionService(storage=mock_storage)
    with pytest.raises(InvalidIngestionInputError, match="Source document reference cannot be None"):
        await service.ingest(None, project_id="proj-1")


@pytest.mark.asyncio
async def test_ingest_missing_project_id(mock_storage):
    """Verify missing project_id raises InvalidIngestionInputError."""
    service = StorageIngestionService(storage=mock_storage)
    with pytest.raises(InvalidIngestionInputError, match="project_id is required"):
        await service.ingest("doc.pdf", project_id=None)


@pytest.mark.asyncio
async def test_ingest_cross_project_violation(mock_storage):
    """Verify cross-project path access is rejected."""
    service = StorageIngestionService(storage=mock_storage)
    with pytest.raises(InvalidIngestionInputError, match="Cross-project access prohibited"):
        await service.ingest("proj-foreign/doc.pdf", project_id="proj-my")


@pytest.mark.asyncio
async def test_ingest_unsupported_reference_type(mock_storage):
    """Verify unsupported source reference object raises InvalidIngestionInputError."""
    service = StorageIngestionService(storage=mock_storage)
    with pytest.raises(InvalidIngestionInputError, match="Unsupported source reference type"):
        await service.ingest(12345, project_id="proj-1")


# ==============================================================================
# 5. Batch Ingestion Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ingest_batch(mock_storage):
    """Verify batch ingestion processes multiple documents in order."""
    mock_storage.download.side_effect = [b"content-1", b"content-2"]

    service = StorageIngestionService(storage=mock_storage)

    sources = [
        "doc1.pdf",
        "doc2.txt",
    ]

    results = await service.ingest_batch(sources, project_id="proj-batch")

    assert len(results) == 2
    assert results[0].original_filename == "doc1.pdf"
    assert results[0].detected_document_type == DocumentType.PDF
    assert results[0].raw_bytes == b"content-1"

    assert results[1].original_filename == "doc2.txt"
    assert results[1].detected_document_type == DocumentType.TXT
    assert results[1].raw_bytes == b"content-2"


@pytest.mark.asyncio
async def test_ingest_batch_empty(mock_storage):
    """Verify batch ingestion handles empty list gracefully."""
    service = StorageIngestionService(storage=mock_storage)
    results = await service.ingest_batch([], project_id="proj-1")
    assert results == []


# ==============================================================================
# 6. Architectural Boundary & Privacy Guard Tests
# ==============================================================================


def test_ingested_document_repr_masks_bytes():
    """Verify IngestedDocument.__repr__ masks raw byte payload to prevent log leaks."""
    sensitive_bytes = b"SECRET PASSWORDS AND SENSITIVE CONTENT"
    doc = IngestedDocument(
        document_id="doc-123",
        project_id="proj-1",
        source_storage_path="proj-1/doc.pdf",
        original_filename="doc.pdf",
        detected_document_type=DocumentType.PDF,
        content_type="application/pdf",
        raw_bytes=sensitive_bytes,
        size_bytes=len(sensitive_bytes),
    )

    repr_str = repr(doc)
    # The raw string must NOT appear in repr
    assert "SECRET PASSWORDS" not in repr_str
    # Metadata should be visible
    assert "doc-123" in repr_str
    assert "proj-1" in repr_str
    assert f"size_bytes={len(sensitive_bytes)}" in repr_str


def test_ingested_document_to_dict():
    """Verify to_dict excludes raw bytes by default and includes them only when requested."""
    data_bytes = b"some data"
    doc = IngestedDocument(
        document_id="doc-123",
        project_id="proj-1",
        source_storage_path="proj-1/doc.pdf",
        original_filename="doc.pdf",
        detected_document_type=DocumentType.PDF,
        content_type="application/pdf",
        raw_bytes=data_bytes,
    )

    dict_default = doc.to_dict()
    assert "raw_bytes" not in dict_default
    assert dict_default["size_bytes"] == len(data_bytes)
    assert dict_default["detected_document_type"] == "pdf"

    dict_with_bytes = doc.to_dict(include_bytes=True)
    assert dict_with_bytes["raw_bytes"] == data_bytes


def test_ingested_document_no_future_pipeline_fields():
    """Assert IngestedDocument does NOT contain future parsing/extraction/chunking fields."""
    doc = IngestedDocument(
        document_id="doc-123",
        project_id="proj-1",
        source_storage_path="proj-1/doc.pdf",
        original_filename="doc.pdf",
        detected_document_type=DocumentType.PDF,
        content_type="application/pdf",
        raw_bytes=b"bytes",
    )

    forbidden_attributes = [
        "text",
        "extracted_text",
        "cleaned_text",
        "chunks",
        "chunk_count",
        "embedding",
        "embeddings",
        "vector_id",
        "vector_ids",
    ]
    for attr in forbidden_attributes:
        assert not hasattr(doc, attr), f"IngestedDocument should not have future field '{attr}'"


# ==============================================================================
# 7. Singleton & Factory Tests
# ==============================================================================


def test_get_and_reset_ingestion_service():
    """Verify singleton lifecycle for get_ingestion_service and reset_ingestion_service."""
    service1 = get_ingestion_service()
    service2 = get_ingestion_service()
    assert service1 is service2

    reset_ingestion_service()
    service3 = get_ingestion_service()
    assert service3 is not service1

    # Custom storage override should produce new instance without overwriting default
    custom_storage = MagicMock(spec=BaseObjectStorage)
    custom_service = get_ingestion_service(storage=custom_storage)
    assert custom_service is not service3


# ==============================================================================
# 8. Additional Edge Cases & Mixed Batch Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ingest_empty_0_byte_file(mock_storage):
    """Verify ingestion of a valid 0-byte file."""
    mock_storage.download.return_value = b""
    service = StorageIngestionService(storage=mock_storage)

    ingested = await service.ingest("proj-zero/empty.txt", project_id="proj-zero")
    assert ingested.raw_bytes == b""
    assert ingested.size_bytes == 0
    assert ingested.detected_document_type == DocumentType.TXT


@pytest.mark.asyncio
async def test_ingest_batch_mixed_sources(mock_storage):
    """Verify batch ingestion handles heterogeneous source reference types in a single call."""
    mock_storage.download.side_effect = [
        b"%PDF doc1",
        b"PK doc2",
        b"text doc3",
        b"# Markdown doc4",
    ]
    service = StorageIngestionService(storage=mock_storage)

    src_doc = SourceDocument(
        source_id="b/proj-mix/d1.pdf",
        project_id="proj-mix",
        filename="d1.pdf",
        storage_path="proj-mix/d1.pdf",
        storage_bucket="b",
        document_type=DocumentType.PDF,
        content_type="application/pdf",
        extension=".pdf",
    )
    doc_ref = DocumentSourceReference(
        project_id="proj-mix",
        storage_path="proj-mix/d2.docx",
    )
    storage_ref = SourceReference(
        bucket="b",
        path="proj-mix/d3.txt",
    )
    raw_path = "proj-mix/d4.md"

    results = await service.ingest_batch(
        [src_doc, doc_ref, storage_ref, raw_path],
        project_id="proj-mix",
    )

    assert len(results) == 4
    assert [r.detected_document_type for r in results] == [
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.TXT,
        DocumentType.MARKDOWN,
    ]


def test_document_source_reference_to_dict():
    """Verify serialization of DocumentSourceReference."""
    ref = DocumentSourceReference(
        project_id="proj-1",
        storage_path="proj-1/report.pdf",
        original_filename="report.pdf",
        content_type="application/pdf",
        document_type=DocumentType.PDF,
        document_version_id="etag-v1",
        document_id="doc-001",
        source_metadata={"env": "prod"},
    )
    d = ref.to_dict()
    assert d["project_id"] == "proj-1"
    assert d["document_type"] == "pdf"
    assert d["document_version_id"] == "etag-v1"
    assert d["source_metadata"]["env"] == "prod"


def test_ingestion_exceptions_hierarchy():
    """Verify custom ingestion exception inheritance and attributes."""
    orig = ValueError("Low-level network glitch")
    err = IngestionError("Ingestion failed", original_error=orig)
    assert str(err) == "Ingestion failed"
    assert err.original_error is orig
    assert issubclass(DocumentRetrievalError, IngestionError)
    assert issubclass(InvalidIngestionInputError, IngestionError)
    assert issubclass(UnsupportedDocumentTypeError, IngestionError)
    assert issubclass(IngestionConfigurationError, IngestionError)
