"""Unit tests for Supabase Storage integration."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from storage3.exceptions import StorageApiError

from app.core.config import Settings
from exceptions.storage import (
    BucketNotFoundError,
    ObjectNotFoundError,
    StorageAuthenticationError,
    StorageBucketError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageUploadError,
)
from storage.object import (
    BaseObjectStorage,
    StorageObjectMetadata,
    SupabaseObjectStorage,
    get_async_supabase_client,
    get_object_storage,
    reset_async_supabase_client,
    reset_object_storage,
)


@pytest.fixture(autouse=True)
def cleanup_client():
    """Reset cached singleton storage and client instances before/after each test."""
    reset_async_supabase_client()
    reset_object_storage()
    yield
    reset_async_supabase_client()
    reset_object_storage()



@pytest.fixture
def mock_settings():
    """Sample settings with Supabase configuration."""
    return Settings(
        SUPABASE_URL="https://mock-test-project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="mock-service-role-key",
        SUPABASE_KEY="mock-anon-key",
        SUPABASE_STORAGE_BUCKET="test-documents",
    )


@pytest.fixture
def mock_bucket_proxy():
    """Mock for AsyncBucketProxy."""
    proxy = AsyncMock()
    proxy.download = AsyncMock(return_value=b"test file content")
    proxy.upload = AsyncMock(return_value={"path": "test-documents/doc.pdf", "Key": "doc.pdf"})
    proxy.remove = AsyncMock(return_value=[{"name": "doc.pdf"}])
    proxy.list = AsyncMock(
        return_value=[
            {
                "name": "doc1.pdf",
                "created_at": "2026-09-01T10:00:00Z",
                "updated_at": "2026-09-01T10:00:00Z",
                "metadata": {
                    "size": 1024,
                    "mimetype": "application/pdf",
                    "eTag": '"abc123etag"',
                },
            }
        ]
    )
    proxy.exists = AsyncMock(return_value=True)
    proxy.info = AsyncMock(
        return_value={
            "created_at": "2026-09-01T10:00:00Z",
            "updated_at": "2026-09-01T10:00:00Z",
            "metadata": {
                "size": 2048,
                "mimetype": "application/pdf",
                "eTag": '"info-etag"',
            },
        }
    )
    proxy.create_signed_url = AsyncMock(
        return_value={"signedURL": "https://mock-storage.supabase.co/signed-url-abc"}
    )
    return proxy


@pytest.fixture
def mock_supabase_client(mock_bucket_proxy):
    """Mock for AsyncClient with storage interface."""
    client = MagicMock()
    storage_mock = MagicMock()
    storage_mock.from_ = MagicMock(return_value=mock_bucket_proxy)
    storage_mock.get_bucket = AsyncMock(return_value=MagicMock(id="test-documents", name="test-documents"))
    storage_mock.create_bucket = AsyncMock(return_value={"name": "test-documents"})
    storage_mock.list_buckets = AsyncMock(return_value=[])
    client.storage = storage_mock
    return client


# ==============================================================================
# 1. Configuration & Client Management Tests
# ==============================================================================


@pytest.mark.anyio
async def test_get_async_supabase_client_missing_url():
    """Verify error raised when SUPABASE_URL is not set."""
    empty_settings = Settings(SUPABASE_URL=None, SUPABASE_KEY="some-key")
    with pytest.raises(StorageConfigurationError) as exc:
        await get_async_supabase_client(empty_settings)
    assert "SUPABASE_URL" in str(exc.value)


@pytest.mark.anyio
async def test_get_async_supabase_client_missing_credentials():
    """Verify error raised when no Supabase keys are configured."""
    empty_settings = Settings(
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_KEY=None,
        SUPABASE_SERVICE_ROLE_KEY=None,
    )
    with pytest.raises(StorageConfigurationError) as exc:
        await get_async_supabase_client(empty_settings)
    assert "credentials" in str(exc.value).lower()


@pytest.mark.anyio
async def test_get_async_supabase_client_caching(mock_settings):
    """Verify get_async_supabase_client reuses the singleton client instance."""
    with patch("storage.object.client.create_async_client", new_callable=AsyncMock) as mock_create:
        mock_instance = MagicMock()
        mock_create.return_value = mock_instance

        client_1 = await get_async_supabase_client(mock_settings)
        client_2 = await get_async_supabase_client(mock_settings)

        assert client_1 is client_2
        assert client_1 is mock_instance
        assert mock_create.call_count == 1


@pytest.mark.anyio
async def test_get_async_supabase_client_connection_error(mock_settings):
    """Verify client initialization failure raises StorageConnectionError."""
    with patch("storage.object.client.create_async_client", side_effect=Exception("Network failure")):
        with pytest.raises(StorageConnectionError) as exc:
            await get_async_supabase_client(mock_settings)
        assert "Network failure" in str(exc.value)


# ==============================================================================
# 2. Synchronous Helpers & Local Computation Tests
# ==============================================================================


def test_path_normalization(mock_settings):
    """Verify synchronous path normalization handles various formats."""
    storage = SupabaseObjectStorage(settings=mock_settings)

    assert storage._normalize_path("folder/file.pdf") == "folder/file.pdf"
    assert storage._normalize_path("/folder/file.pdf") == "folder/file.pdf"
    assert storage._normalize_path("folder\\subfolder\\file.pdf") == "folder/subfolder/file.pdf"
    assert storage._normalize_path("  /nested/doc.txt/  ") == "nested/doc.txt"

    with pytest.raises(StorageError):
        storage._normalize_path("")

    with pytest.raises(StorageError):
        storage._normalize_path("   ")


def test_guess_content_type(mock_settings):
    """Verify MIME type inference from filenames."""
    storage = SupabaseObjectStorage(settings=mock_settings)

    assert storage._guess_content_type("document.pdf") == "application/pdf"
    assert storage._guess_content_type("data.json") == "application/json"
    assert storage._guess_content_type("text.txt") == "text/plain"
    assert storage._guess_content_type("unknown.xyz123") == "application/octet-stream"


def test_parse_metadata_dict(mock_settings):
    """Verify conversion of Supabase item dict to StorageObjectMetadata."""
    storage = SupabaseObjectStorage(bucket_name="my-bucket", settings=mock_settings)

    raw_item = {
        "name": "report.pdf",
        "created_at": "2026-09-01T12:00:00Z",
        "updated_at": "2026-09-01T13:00:00Z",
        "metadata": {
            "size": "524288",
            "mimetype": "application/pdf",
            "eTag": '"etag-xyz"',
        },
    }

    meta = storage._parse_metadata_dict(raw_item, parent_prefix="2026/finance")
    assert isinstance(meta, StorageObjectMetadata)
    assert meta.name == "report.pdf"
    assert meta.path == "2026/finance/report.pdf"
    assert meta.bucket == "my-bucket"
    assert meta.size_bytes == 524288
    assert meta.content_type == "application/pdf"
    assert meta.created_at == "2026-09-01T12:00:00Z"
    assert meta.updated_at == "2026-09-01T13:00:00Z"
    assert meta.etag == '"etag-xyz"'


# ==============================================================================
# 3. Asynchronous Object Storage Operations Tests
# ==============================================================================


@pytest.mark.anyio
async def test_download_success(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify successful asynchronous download returns byte content."""
    mock_bucket_proxy.download.return_value = b"Hello, World!"
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    content = await storage.download("documents/sample.txt")
    assert content == b"Hello, World!"
    mock_bucket_proxy.download.assert_awaited_once_with("documents/sample.txt")


@pytest.mark.anyio
async def test_download_not_found(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify 404 response translates to ObjectNotFoundError."""
    mock_bucket_proxy.download.side_effect = StorageApiError(
        message="Object not found", code="404", status=404
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    with pytest.raises(ObjectNotFoundError) as exc:
        await storage.download("missing.pdf")
    assert "missing.pdf" in str(exc.value)


@pytest.mark.anyio
async def test_download_unauthorized(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify 401 response translates to StorageAuthenticationError."""
    mock_bucket_proxy.download.side_effect = StorageApiError(
        message="Unauthorized access", code="401", status=401
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    with pytest.raises(StorageAuthenticationError):
        await storage.download("secret.pdf")


@pytest.mark.anyio
async def test_download_generic_error(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify generic failure translates to StorageDownloadError."""
    mock_bucket_proxy.download.side_effect = Exception("Read timeout")
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    with pytest.raises(StorageDownloadError) as exc:
        await storage.download("doc.pdf")
    assert "Read timeout" in str(exc.value)


@pytest.mark.anyio
async def test_upload_bytes(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify uploading raw bytes."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    meta = await storage.upload("uploads/raw.txt", b"raw text content", content_type="text/plain")
    assert meta.name == "raw.txt"
    assert meta.path == "uploads/raw.txt"
    assert meta.size_bytes == len(b"raw text content")
    assert meta.content_type == "text/plain"

    mock_bucket_proxy.upload.assert_awaited_once_with(
        path="uploads/raw.txt",
        file=b"raw text content",
        file_options={"content-type": "text/plain", "upsert": "false"},
    )


@pytest.mark.anyio
async def test_upload_stream(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify uploading from a BinaryIO stream."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    stream = io.BytesIO(b"streamed content")
    meta = await storage.upload("uploads/stream.bin", stream, upsert=True)
    assert meta.name == "stream.bin"
    assert meta.path == "uploads/stream.bin"
    assert meta.size_bytes == 16

    mock_bucket_proxy.upload.assert_awaited_once_with(
        path="uploads/stream.bin",
        file=b"streamed content",
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )


@pytest.mark.anyio
async def test_upload_local_file(tmp_path: Path, mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify uploading a local file path."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    test_file = tmp_path / "test_doc.md"
    test_file.write_text("# Test Document", encoding="utf-8")

    meta = await storage.upload("remote/test_doc.md", str(test_file))
    assert meta.name == "test_doc.md"
    assert meta.path == "remote/test_doc.md"
    assert meta.size_bytes == len(b"# Test Document")


@pytest.mark.anyio
async def test_upload_bucket_not_found(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify upload failure with non-existing bucket raises BucketNotFoundError."""
    mock_bucket_proxy.upload.side_effect = StorageApiError(
        message="Bucket not found", code="404", status=404
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    with pytest.raises(BucketNotFoundError):
        await storage.upload("doc.pdf", b"content")


@pytest.mark.anyio
async def test_delete_success(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify deleting a single file."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    result = await storage.delete("folder/file.pdf")
    assert result is True
    mock_bucket_proxy.remove.assert_awaited_once_with(["folder/file.pdf"])


@pytest.mark.anyio
async def test_delete_many(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify deleting multiple files."""
    mock_bucket_proxy.remove.return_value = [{"name": "file1.pdf"}, {"name": "file2.pdf"}]
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    deleted = await storage.delete_many(["file1.pdf", "file2.pdf"])
    assert len(deleted) == 2
    mock_bucket_proxy.remove.assert_awaited_once_with(["file1.pdf", "file2.pdf"])


@pytest.mark.anyio
async def test_delete_unauthorized(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify unauthorized delete raises StorageAuthenticationError."""
    mock_bucket_proxy.remove.side_effect = StorageApiError(
        message="Unauthorized", code="403", status=403
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    with pytest.raises(StorageAuthenticationError):
        await storage.delete("doc.pdf")


@pytest.mark.anyio
async def test_list_objects(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify listing objects under a path prefix."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    objects = await storage.list_objects(prefix="reports", limit=50)
    assert len(objects) == 1
    assert objects[0].name == "doc1.pdf"
    assert objects[0].path == "reports/doc1.pdf"
    assert objects[0].size_bytes == 1024
    assert objects[0].content_type == "application/pdf"


@pytest.mark.anyio
async def test_exists(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify exists returns True when found and False on 404."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    mock_bucket_proxy.exists.return_value = True
    assert await storage.exists("existing.pdf") is True

    mock_bucket_proxy.exists.side_effect = StorageApiError(
        message="Not found", code="404", status=404
    )
    assert await storage.exists("missing.pdf") is False


@pytest.mark.anyio
async def test_get_metadata(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify retrieving object metadata via info."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    meta = await storage.get_metadata("nested/folder/report.pdf")
    assert meta.name == "report.pdf"
    assert meta.path == "nested/folder/report.pdf"
    assert meta.size_bytes == 2048
    assert meta.etag == '"info-etag"'


@pytest.mark.anyio
async def test_ensure_bucket_exists_already_present(mock_settings, mock_supabase_client):
    """Verify bucket check returns True when bucket already exists."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    result = await storage.ensure_bucket_exists()
    assert result is True
    mock_supabase_client.storage.get_bucket.assert_awaited_once_with("test-documents")
    mock_supabase_client.storage.create_bucket.assert_not_awaited()


@pytest.mark.anyio
async def test_ensure_bucket_exists_creates_when_missing(mock_settings, mock_supabase_client):
    """Verify bucket is created idempotently when get_bucket raises 404."""
    mock_supabase_client.storage.get_bucket.side_effect = StorageApiError(
        message="Bucket not found", code="404", status=404
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    result = await storage.ensure_bucket_exists()
    assert result is True
    mock_supabase_client.storage.create_bucket.assert_awaited_once_with(
        id="test-documents",
        name="test-documents",
        options={"public": False},
    )


@pytest.mark.anyio
async def test_ensure_bucket_exists_handles_race_condition(mock_settings, mock_supabase_client):
    """Verify bucket creation handles 409 / already exists gracefully."""
    mock_supabase_client.storage.get_bucket.side_effect = StorageApiError(
        message="Bucket not found", code="404", status=404
    )
    mock_supabase_client.storage.create_bucket.side_effect = StorageApiError(
        message="Bucket already exists", code="409", status=409
    )
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    result = await storage.ensure_bucket_exists()
    assert result is True


@pytest.mark.anyio
async def test_create_signed_url(mock_settings, mock_supabase_client, mock_bucket_proxy):
    """Verify generating a pre-signed URL."""
    storage = SupabaseObjectStorage(client=mock_supabase_client, settings=mock_settings)

    url = await storage.create_signed_url("private/contract.pdf", expires_in=1800)
    assert "https://mock-storage.supabase.co/signed-url-abc" in url
    mock_bucket_proxy.create_signed_url.assert_awaited_once_with("private/contract.pdf", expires_in=1800)


# ==============================================================================
# 4. Factory & Interface Compliance Tests
# ==============================================================================


def test_get_object_storage_factory(mock_settings):
    """Verify get_object_storage returns BaseObjectStorage instance with caching."""
    storage1 = get_object_storage(settings=mock_settings)
    storage2 = get_object_storage(settings=mock_settings)

    assert isinstance(storage1, BaseObjectStorage)
    assert storage1 is storage2
    assert storage1.bucket_name == "test-documents"

    # Custom bucket creates distinct instance
    custom_storage = get_object_storage(bucket_name="custom-bucket", settings=mock_settings)
    assert custom_storage.bucket_name == "custom-bucket"
    assert custom_storage is not storage1
