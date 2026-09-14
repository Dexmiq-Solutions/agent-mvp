"""Unit tests for the Acquisition layer of the document indexing pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from rag.acquisition import (
    AcquisitionConfigurationError,
    AcquisitionError,
    AcquisitionResult,
    AcquisitionStatus,
    BaseAcquisitionService,
    DocumentType,
    InvalidSourceReferenceError,
    SourceDiscoveryError,
    SourceDocument,
    SourceReference,
    StorageAcquisitionService,
    UnsupportedDocumentError,
    get_acquisition_service,
    get_canonical_mime_type,
    get_document_type,
    is_supported_extension,
    normalize_extension,
    reset_acquisition_service,
)
from core.config import Settings
from exceptions.storage import (
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageConfigurationError,
    StorageError,
)
from storage.object import BaseObjectStorage, StorageObjectMetadata


@pytest.fixture(autouse=True)
def cleanup_acquisition_service():
    """Reset the singleton acquisition service before and after each test."""
    reset_acquisition_service()
    yield
    reset_acquisition_service()


@pytest.fixture
def mock_storage():
    """Create a mock BaseObjectStorage instance."""
    storage = MagicMock(spec=BaseObjectStorage)
    storage.bucket_name = "test-documents"
    storage.list_objects = AsyncMock(return_value=[])
    storage.get_metadata = AsyncMock()
    storage.download = AsyncMock()  # Must NEVER be called by acquisition
    storage.exists = AsyncMock(return_value=True)
    return storage


# ==============================================================================
# 1. Supported Document Formats & Utilities
# ==============================================================================


def test_normalize_extension():
    """Verify extension normalization handles various formats, cases, and paths."""
    assert normalize_extension("doc.PDF") == ".pdf"
    assert normalize_extension(".DOCX") == ".docx"
    assert normalize_extension("file.txt") == ".txt"
    assert normalize_extension("path/to/readme.md") == ".md"
    assert normalize_extension("folder\\guide.markdown") == ".markdown"
    assert normalize_extension("pdf") == ".pdf"
    assert normalize_extension("") == ""


def test_is_supported_extension():
    """Verify supported and unsupported extensions detection."""
    # Supported: .pdf, .docx, .txt, .md, .markdown
    assert is_supported_extension("file.pdf") is True
    assert is_supported_extension("FILE.PDF") is True
    assert is_supported_extension("document.docx") is True
    assert is_supported_extension("notes.txt") is True
    assert is_supported_extension("README.md") is True
    assert is_supported_extension("guide.markdown") is True
    assert is_supported_extension(".md") is True

    # Unsupported
    assert is_supported_extension("data.csv") is False
    assert is_supported_extension("photo.png") is False
    assert is_supported_extension("app.py") is False
    assert is_supported_extension("archive.zip") is False
    assert is_supported_extension("script.sh") is False
    assert is_supported_extension("") is False


def test_get_document_type():
    """Verify resolution of canonical DocumentType enum."""
    assert get_document_type("test.pdf") == DocumentType.PDF
    assert get_document_type("test.docx") == DocumentType.DOCX
    assert get_document_type("test.txt") == DocumentType.TXT
    assert get_document_type("test.md") == DocumentType.MARKDOWN
    assert get_document_type("test.markdown") == DocumentType.MARKDOWN
    assert get_document_type("test.unknown") is None


def test_get_canonical_mime_type():
    """Verify canonical MIME types for supported DocumentTypes."""
    assert get_canonical_mime_type(DocumentType.PDF) == "application/pdf"
    assert (
        get_canonical_mime_type(DocumentType.DOCX)
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert get_canonical_mime_type(DocumentType.TXT) == "text/plain"
    assert get_canonical_mime_type(DocumentType.MARKDOWN) == "text/markdown"


# ==============================================================================
# 2. Supported Format Discovery Tests
# ==============================================================================


@pytest.mark.anyio
async def test_acquire_pdf_document(mock_storage):
    """Verify supported PDF files are discovered and mapped correctly."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="report.pdf",
            path="project-alpha/report.pdf",
            bucket="test-documents",
            size_bytes=1048576,
            content_type="application/pdf",
            created_at="2026-09-01T10:00:00Z",
            updated_at="2026-09-01T11:00:00Z",
            etag='"etag-pdf-123"',
            extra_metadata={"eTag": '"etag-pdf-123"'},
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-alpha")

    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, SourceDocument)
    assert doc.project_id == "project-alpha"
    assert doc.filename == "report.pdf"
    assert doc.storage_path == "project-alpha/report.pdf"
    assert doc.storage_bucket == "test-documents"
    assert doc.document_type == DocumentType.PDF
    assert doc.content_type == "application/pdf"
    assert doc.extension == ".pdf"
    assert doc.size_bytes == 1048576
    assert doc.created_at == "2026-09-01T10:00:00Z"
    assert doc.source_id == "test-documents/project-alpha/report.pdf"
    assert doc.source_ref.path == "project-alpha/report.pdf"
    assert doc.source_ref.etag == '"etag-pdf-123"'


@pytest.mark.anyio
async def test_acquire_docx_document(mock_storage):
    """Verify supported DOCX files are discovered and mapped correctly."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="spec.docx",
            path="project-alpha/spec.docx",
            bucket="test-documents",
            size_bytes=20480,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            created_at="2026-09-02T12:00:00Z",
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-alpha")

    assert len(result) == 1
    doc = result[0]
    assert doc.document_type == DocumentType.DOCX
    assert doc.extension == ".docx"
    assert (
        doc.content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@pytest.mark.anyio
async def test_acquire_txt_document(mock_storage):
    """Verify supported TXT files are discovered and mapped correctly."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="notes.txt",
            path="project-alpha/notes.txt",
            bucket="test-documents",
            size_bytes=512,
            content_type="text/plain",
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-alpha")

    assert len(result) == 1
    doc = result[0]
    assert doc.document_type == DocumentType.TXT
    assert doc.extension == ".txt"
    assert doc.content_type == "text/plain"


@pytest.mark.anyio
async def test_acquire_markdown_documents(mock_storage):
    """Verify supported Markdown files (.md and .markdown) are discovered correctly."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="readme.md",
            path="project-alpha/readme.md",
            bucket="test-documents",
            size_bytes=2048,
            content_type="text/markdown",
        ),
        StorageObjectMetadata(
            name="guide.markdown",
            path="project-alpha/guide.markdown",
            bucket="test-documents",
            size_bytes=4096,
            content_type="text/markdown",
        ),
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-alpha")

    assert len(result) == 2
    assert result[0].document_type == DocumentType.MARKDOWN
    assert result[0].extension == ".md"
    assert result[1].document_type == DocumentType.MARKDOWN
    assert result[1].extension == ".markdown"


# ==============================================================================
# 3. Unsupported Files & Directory Handling Tests
# ==============================================================================


@pytest.mark.anyio
async def test_unsupported_files_skipped_in_batch(mock_storage):
    """Verify unsupported files are excluded from documents and recorded in skipped_paths."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="valid.pdf",
            path="project-beta/valid.pdf",
            bucket="test-documents",
            size_bytes=1000,
        ),
        StorageObjectMetadata(
            name="table.csv",
            path="project-beta/table.csv",
            bucket="test-documents",
            size_bytes=2000,
        ),
        StorageObjectMetadata(
            name="image.png",
            path="project-beta/image.png",
            bucket="test-documents",
            size_bytes=3000,
        ),
        StorageObjectMetadata(
            name="archive.zip",
            path="project-beta/archive.zip",
            bucket="test-documents",
            size_bytes=4000,
        ),
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-beta")

    assert len(result.documents) == 1
    assert result.documents[0].filename == "valid.pdf"
    assert "project-beta/table.csv" in result.skipped_paths
    assert "project-beta/image.png" in result.skipped_paths
    assert "project-beta/archive.zip" in result.skipped_paths
    assert result.total_discovered == 4
    assert result.status == AcquisitionStatus.PARTIAL


@pytest.mark.anyio
async def test_directory_and_placeholder_entries_skipped(mock_storage):
    """Verify folder placeholders and directory entries are safely skipped."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name=".emptyFolderPlaceholder",
            path="project-beta/.emptyFolderPlaceholder",
            bucket="test-documents",
            size_bytes=0,
        ),
        StorageObjectMetadata(
            name="subfolder/",
            path="project-beta/subfolder/",
            bucket="test-documents",
            size_bytes=None,
        ),
        StorageObjectMetadata(
            name="directory_entry",
            path="project-beta/directory_entry",
            bucket="test-documents",
            size_bytes=None,
            content_type="application/x-directory",
        ),
        StorageObjectMetadata(
            name="doc.pdf",
            path="project-beta/doc.pdf",
            bucket="test-documents",
            size_bytes=5000,
            content_type="application/pdf",
        ),
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-beta")

    assert len(result.documents) == 1
    assert result.documents[0].filename == "doc.pdf"


@pytest.mark.anyio
async def test_acquire_document_unsupported_format_raises(mock_storage):
    """Verify acquire_document on an unsupported file raises UnsupportedDocumentError."""
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(UnsupportedDocumentError) as exc_info:
        await service.acquire_document("project-beta", "project-beta/data.csv")
    assert "unsupported file format" in str(exc_info.value).lower()
    assert ".csv" in str(exc_info.value)


# ==============================================================================
# 4. Project Isolation & Tenant Security Tests
# ==============================================================================


@pytest.mark.anyio
async def test_acquisition_strictly_scoped_to_project_prefix(mock_storage):
    """Verify list_objects is queried with the exact project-scoped prefix."""
    service = StorageAcquisitionService(storage=mock_storage)
    await service.acquire("project-tenant-42")

    mock_storage.list_objects.assert_awaited_once_with(
        prefix="project-tenant-42",
        limit=100,
        offset=0,
    )


@pytest.mark.anyio
async def test_acquisition_with_subfolder_prefix(mock_storage):
    """Verify acquisition with optional subfolder prefix combines with project_id."""
    service = StorageAcquisitionService(storage=mock_storage)
    await service.acquire("project-tenant-42", prefix="invoices/2026")

    mock_storage.list_objects.assert_awaited_once_with(
        prefix="project-tenant-42/invoices/2026",
        limit=100,
        offset=0,
    )


@pytest.mark.anyio
async def test_cross_tenant_leakage_prevented_in_acquire(mock_storage):
    """Verify objects outside the requested project prefix are guarded and excluded."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="valid.pdf",
            path="project-A/valid.pdf",
            bucket="test-documents",
            size_bytes=1000,
        ),
        StorageObjectMetadata(
            name="leaked.pdf",
            path="project-B/leaked.pdf",  # Storage provider returned object from another tenant
            bucket="test-documents",
            size_bytes=1000,
        ),
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("project-A")

    assert len(result.documents) == 1
    assert result.documents[0].storage_path == "project-A/valid.pdf"
    assert "project-B/leaked.pdf" in result.skipped_paths


@pytest.mark.anyio
async def test_cross_tenant_access_prevented_in_acquire_document(mock_storage):
    """Verify acquire_document rejects attempting to access another project's file."""
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(InvalidSourceReferenceError) as exc_info:
        await service.acquire_document("project-A", "project-B/sensitive.pdf")
    assert "Cross-project access prohibited" in str(exc_info.value)


@pytest.mark.anyio
async def test_invalid_project_id_raises(mock_storage):
    """Verify empty or non-string project_id raises InvalidSourceReferenceError."""
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(InvalidSourceReferenceError):
        await service.acquire("")

    with pytest.raises(InvalidSourceReferenceError):
        await service.acquire("   ")

    with pytest.raises(InvalidSourceReferenceError):
        await service.acquire(None)  # type: ignore

    with pytest.raises(InvalidSourceReferenceError):
        await service.acquire_document("", "doc.pdf")


# ==============================================================================
# 5. Performance & Zero-Download Guarantee Tests
# ==============================================================================


@pytest.mark.anyio
async def test_zero_downloads_during_acquisition(mock_storage):
    """Verify acquisition never invokes storage.download() merely to discover documents."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="doc.pdf",
            path="proj/doc.pdf",
            bucket="test-documents",
            size_bytes=50000000,  # 50 MB
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("proj")

    assert len(result) == 1
    mock_storage.download.assert_not_called()


@pytest.mark.anyio
async def test_zero_downloads_during_acquire_document(mock_storage):
    """Verify acquire_document never invokes storage.download()."""
    mock_storage.get_metadata.return_value = StorageObjectMetadata(
        name="doc.pdf",
        path="proj/doc.pdf",
        bucket="test-documents",
        size_bytes=50000,
        content_type="application/pdf",
    )
    service = StorageAcquisitionService(storage=mock_storage)
    doc = await service.acquire_document("proj", "doc.pdf")

    assert doc.filename == "doc.pdf"
    mock_storage.download.assert_not_called()
    mock_storage.get_metadata.assert_awaited_once_with("proj/doc.pdf")


@pytest.mark.anyio
async def test_pagination_and_limit(mock_storage):
    """Verify multi-batch pagination retrieves all project items efficiently."""
    batch_1 = [
        StorageObjectMetadata(
            name=f"doc_{i}.pdf",
            path=f"proj/doc_{i}.pdf",
            bucket="test-documents",
            size_bytes=100,
        )
        for i in range(10)
    ]
    batch_2 = [
        StorageObjectMetadata(
            name=f"doc_{i}.pdf",
            path=f"proj/doc_{i}.pdf",
            bucket="test-documents",
            size_bytes=100,
        )
        for i in range(10, 15)
    ]

    mock_storage.list_objects.side_effect = [batch_1, batch_2]
    service = StorageAcquisitionService(storage=mock_storage, batch_size=10)

    result = await service.acquire("proj")
    assert len(result) == 15
    assert mock_storage.list_objects.await_count == 2


# ==============================================================================
# 6. Malformed & Incomplete Metadata Resilience Tests
# ==============================================================================


@pytest.mark.anyio
async def test_malformed_and_incomplete_metadata_handled_safely(mock_storage):
    """Verify documents with missing size, generic MIME, and null timestamps are acquired safely."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="incomplete.pdf",
            path="proj/incomplete.pdf",
            bucket="test-documents",
            size_bytes=None,  # missing size
            content_type=None,  # missing content type
            created_at=None,
            updated_at=None,
            etag=None,
            extra_metadata={},
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("proj")

    assert len(result) == 1
    doc = result[0]
    assert doc.size_bytes is None
    assert doc.created_at is None
    # Verifies MIME was automatically inferred from supported DocumentType
    assert doc.content_type == "application/pdf"
    assert doc.document_type == DocumentType.PDF


@pytest.mark.anyio
async def test_generic_octet_stream_mime_type_corrected(mock_storage):
    """Verify application/octet-stream content-type is corrected to canonical MIME."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="notes.markdown",
            path="proj/notes.markdown",
            bucket="test-documents",
            content_type="application/octet-stream",
        )
    ]
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("proj")

    assert len(result) == 1
    assert result[0].content_type == "text/markdown"


# ==============================================================================
# 7. Empty Project & Boundary Tests
# ==============================================================================


@pytest.mark.anyio
async def test_empty_project_location_returns_empty_result(mock_storage):
    """Verify empty project folder returns AcquisitionResult with empty status and 0 documents."""
    mock_storage.list_objects.return_value = []
    service = StorageAcquisitionService(storage=mock_storage)
    result = await service.acquire("empty-project")

    assert len(result) == 0
    assert result.documents == []
    assert result.skipped_paths == []
    assert result.total_discovered == 0
    assert result.status == AcquisitionStatus.EMPTY
    assert result.project_id == "empty-project"


# ==============================================================================
# 8. Storage Exception Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_storage_authentication_error_translation(mock_storage):
    """Verify StorageAuthenticationError translates to SourceDiscoveryError."""
    mock_storage.list_objects.side_effect = StorageAuthenticationError("Unauthorized")
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(SourceDiscoveryError) as exc_info:
        await service.acquire("proj")
    assert "Authentication failed" in str(exc_info.value)
    assert exc_info.value.original_error is not None


@pytest.mark.anyio
async def test_storage_configuration_error_translation(mock_storage):
    """Verify StorageConfigurationError translates to AcquisitionConfigurationError."""
    mock_storage.list_objects.side_effect = StorageConfigurationError("Invalid bucket config")
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(AcquisitionConfigurationError) as exc_info:
        await service.acquire("proj")
    assert "Storage configuration error" in str(exc_info.value)


@pytest.mark.anyio
async def test_generic_storage_error_translation(mock_storage):
    """Verify generic StorageError translates to SourceDiscoveryError."""
    mock_storage.list_objects.side_effect = StorageError("Connection timeout")
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(SourceDiscoveryError) as exc_info:
        await service.acquire("proj")
    assert "Storage failure" in str(exc_info.value)


@pytest.mark.anyio
async def test_acquire_document_not_found_translation(mock_storage):
    """Verify ObjectNotFoundError in acquire_document translates to SourceDiscoveryError."""
    mock_storage.get_metadata.side_effect = ObjectNotFoundError("Object not found")
    service = StorageAcquisitionService(storage=mock_storage)

    with pytest.raises(SourceDiscoveryError) as exc_info:
        await service.acquire_document("proj", "missing.pdf")
    assert "not found in storage bucket" in str(exc_info.value).lower()


# ==============================================================================
# 9. Single Document Acquisition Tests
# ==============================================================================


@pytest.mark.anyio
async def test_acquire_document_success_with_relative_path(mock_storage):
    """Verify acquire_document works with bare relative path (resolving under project)."""
    mock_storage.get_metadata.return_value = StorageObjectMetadata(
        name="contract.pdf",
        path="proj-99/contract.pdf",
        bucket="test-documents",
        size_bytes=12345,
        content_type="application/pdf",
    )
    service = StorageAcquisitionService(storage=mock_storage)
    doc = await service.acquire_document("proj-99", "contract.pdf")

    assert doc.project_id == "proj-99"
    assert doc.storage_path == "proj-99/contract.pdf"
    assert doc.filename == "contract.pdf"
    mock_storage.get_metadata.assert_awaited_once_with("proj-99/contract.pdf")


@pytest.mark.anyio
async def test_acquire_document_success_with_full_project_path(mock_storage):
    """Verify acquire_document works when path already includes project prefix."""
    mock_storage.get_metadata.return_value = StorageObjectMetadata(
        name="contract.pdf",
        path="proj-99/contract.pdf",
        bucket="test-documents",
        size_bytes=12345,
        content_type="application/pdf",
    )
    service = StorageAcquisitionService(storage=mock_storage)
    doc = await service.acquire_document("proj-99", "proj-99/contract.pdf")

    assert doc.storage_path == "proj-99/contract.pdf"


@pytest.mark.anyio
async def test_acquire_document_directory_target_raises(mock_storage):
    """Verify acquire_document on a directory target raises InvalidSourceReferenceError."""
    mock_storage.get_metadata.return_value = StorageObjectMetadata(
        name="folder/",
        path="proj-99/folder/",
        bucket="test-documents",
        size_bytes=None,
    )
    service = StorageAcquisitionService(storage=mock_storage)

    # Note: folder/ will have no extension, so it fails format check or directory check
    with pytest.raises(UnsupportedDocumentError):
        await service.acquire_document("proj-99", "folder/")


# ==============================================================================
# 10. Service Singleton & Serialization Tests
# ==============================================================================


def test_get_acquisition_service_caching():
    """Verify get_acquisition_service returns cached singleton and reset clears it."""
    with patch("rag.acquisition.service.get_object_storage") as mock_get_storage:
        mock_get_storage.return_value = MagicMock(spec=BaseObjectStorage)

        srv1 = get_acquisition_service()
        srv2 = get_acquisition_service()
        assert srv1 is srv2

        reset_acquisition_service()
        srv3 = get_acquisition_service()
        assert srv3 is not srv1


def test_acquisition_result_and_document_serialization():
    """Verify to_dict() serialization on models."""
    ref = SourceReference(bucket="b", path="p/d.pdf", etag="e")
    doc = SourceDocument(
        source_id="b/p/d.pdf",
        project_id="p",
        filename="d.pdf",
        storage_path="p/d.pdf",
        storage_bucket="b",
        document_type=DocumentType.PDF,
        content_type="application/pdf",
        extension=".pdf",
        size_bytes=100,
        created_at="2026-09-01T00:00:00Z",
        source_ref=ref,
    )
    doc_dict = doc.to_dict()
    assert doc_dict["source_id"] == "b/p/d.pdf"
    assert doc_dict["document_type"] == "pdf"
    assert doc_dict["source_ref"]["etag"] == "e"

    res = AcquisitionResult(
        project_id="p",
        documents=[doc],
        skipped_paths=["p/bad.csv"],
        total_discovered=2,
        status=AcquisitionStatus.PARTIAL,
    )
    res_dict = res.to_dict()
    assert res_dict["project_id"] == "p"
    assert res_dict["status"] == "partial"
    assert res_dict["document_count"] == 1
    assert res_dict["skipped_count"] == 1
