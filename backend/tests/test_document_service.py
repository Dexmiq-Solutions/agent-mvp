"""Unit and integration tests for DocumentService."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from exceptions.document import (
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
    InvalidDocumentDataError,
)
from exceptions.project import ProjectNotFoundError
from models.base import Base
from models.document import DocumentModel, DocumentVersionModel, DocumentVersionStatus
from models.project import ProjectModel
from services.document_service import DocumentService
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_session() -> AsyncIterator[AsyncSession]:
    """Create an in-memory SQLite database and yield an active AsyncSession."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


def make_mock_storage() -> AsyncMock:
    """Create a mock BaseObjectStorage that simulates successful uploads and deletions."""
    storage = AsyncMock(spec=BaseObjectStorage)

    async def fake_upload(path, data, content_type=None, upsert=False):
        filename = path.split("/")[-1]
        size = len(data) if isinstance(data, bytes) else 1024
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=size,
            content_type=content_type or "application/pdf",
            etag="fake-etag-123",
        )

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    return storage


async def seed_project(session: AsyncSession, name: str = "Test Project") -> ProjectModel:
    """Helper to seed a project in the database."""
    proj = ProjectModel(name=name)
    session.add(proj)
    await session.flush()
    return proj


# ==============================================================================
# 1. Document Creation & Initial Version Tests
# ==============================================================================


@pytest.mark.anyio
async def test_create_document_initial_version():
    """Verify document creation generates stable document_id, version 1, and pending status."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id,
            filename="User_Guide.pdf",
            file_data=b"%PDF-1.4 sample content",
            content_type="application/pdf",
            name="Official User Guide",
        )

        assert doc.id is not None
        assert doc.project_id == project.id
        assert doc.name == "Official User Guide"
        assert len(doc.versions) == 1

        v1 = doc.versions[0]
        assert v1.version_number == 1
        assert v1.document_id == doc.id
        assert v1.project_id == project.id
        assert v1.original_filename == "User_Guide.pdf"
        assert v1.storage_bucket == "documents"
        assert v1.storage_path == f"{project.id}/{doc.id}/v1/User_Guide.pdf"
        assert v1.size_bytes == len(b"%PDF-1.4 sample content")
        assert v1.content_type == "application/pdf"
        assert v1.etag == "fake-etag-123"
        # Status MUST be pending, NOT ready
        assert v1.status == DocumentVersionStatus.PENDING.value

        storage.upload.assert_awaited_once()


@pytest.mark.anyio
async def test_create_document_default_name_from_filename():
    """Verify document display name defaults to filename when not provided."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id,
            filename="Specs.docx",
            file_data=b"docx bytes",
        )
        assert doc.name == "Specs.docx"


@pytest.mark.anyio
async def test_create_document_project_not_found():
    """Verify creating a document under non-existent project raises ProjectNotFoundError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service = DocumentService(session=session, storage=storage)

        with pytest.raises(ProjectNotFoundError) as exc_info:
            await service.create_document(
                project_id="missing-project-id",
                filename="file.pdf",
                file_data=b"content",
            )
        assert "missing-project-id" in str(exc_info.value)
        storage.upload.assert_not_awaited()


@pytest.mark.anyio
async def test_create_document_empty_filename():
    """Verify empty filename raises InvalidDocumentDataError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        with pytest.raises(InvalidDocumentDataError):
            await service.create_document(
                project_id=project.id,
                filename="",
                file_data=b"content",
            )


@pytest.mark.anyio
async def test_create_document_empty_content():
    """Verify empty file bytes raises InvalidDocumentDataError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        with pytest.raises(InvalidDocumentDataError):
            await service.create_document(
                project_id=project.id,
                filename="empty.txt",
                file_data=b"",
            )


# ==============================================================================
# 2. Document Versioning Tests
# ==============================================================================


@pytest.mark.anyio
async def test_create_subsequent_document_versions():
    """Verify creating subsequent versions increments version number, keeps stable document_id, and produces distinct version IDs."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        # Initial Document (v1)
        doc = await service.create_document(
            project_id=project.id,
            filename="Report.pdf",
            file_data=b"v1 content",
        )
        doc_id = doc.id
        v1_id = doc.versions[0].id

        # Second Version (v2)
        v2 = await service.create_document_version(
            project_id=project.id,
            document_id=doc_id,
            filename="Report_revised.pdf",
            file_data=b"v2 updated content",
        )

        assert v2.version_number == 2
        assert v2.document_id == doc_id
        assert v2.id != v1_id
        assert v2.storage_path == f"{project.id}/{doc_id}/v2/Report_revised.pdf"
        assert v2.status == DocumentVersionStatus.PENDING.value

        # Third Version (v3)
        v3 = await service.create_document_version(
            project_id=project.id,
            document_id=doc_id,
            filename="Report_final.pdf",
            file_data=b"v3 final content",
        )

        assert v3.version_number == 3
        assert v3.document_id == doc_id
        assert v3.id not in (v1_id, v2.id)
        assert v3.storage_path == f"{project.id}/{doc_id}/v3/Report_final.pdf"
        assert v3.status == DocumentVersionStatus.PENDING.value

        # Verify all versions in document detail
        fetched = await service.get_document(project_id=project.id, document_id=doc_id)
        assert len(fetched.versions) == 3
        assert [v.version_number for v in fetched.versions] == [1, 2, 3]


# ==============================================================================
# 3. Document Listing & Retrieval Tests
# ==============================================================================


@pytest.mark.anyio
async def test_list_documents_project_isolation():
    """Verify listing documents returns only sources belonging to the requested project."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        p1 = await seed_project(session, name="Project 1")
        p2 = await seed_project(session, name="Project 2")
        service = DocumentService(session=session, storage=storage)

        doc1 = await service.create_document(
            project_id=p1.id, filename="p1_doc.txt", file_data=b"content 1"
        )
        doc2 = await service.create_document(
            project_id=p2.id, filename="p2_doc.txt", file_data=b"content 2"
        )

        # List for p1
        p1_docs = await service.list_documents(project_id=p1.id)
        assert len(p1_docs) == 1
        assert p1_docs[0].id == doc1.id
        assert p1_docs[0].name == "p1_doc.txt"

        # List for p2
        p2_docs = await service.list_documents(project_id=p2.id)
        assert len(p2_docs) == 1
        assert p2_docs[0].id == doc2.id
        assert p2_docs[0].name == "p2_doc.txt"


@pytest.mark.anyio
async def test_get_document_success():
    """Verify retrieving document detail returns all versions."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="Doc.pdf", file_data=b"content"
        )
        fetched = await service.get_document(project_id=project.id, document_id=doc.id)

        assert fetched.id == doc.id
        assert len(fetched.versions) == 1
        assert fetched.versions[0].version_number == 1


@pytest.mark.anyio
async def test_get_document_not_found():
    """Verify non-existent document ID raises DocumentNotFoundError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        with pytest.raises(DocumentNotFoundError):
            await service.get_document(project_id=project.id, document_id="missing-id")


# ==============================================================================
# 4. Document Update & Deletion Tests
# ==============================================================================


@pytest.mark.anyio
async def test_update_document_name():
    """Verify updating document name under project boundary."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="Old.pdf", file_data=b"content"
        )
        updated = await service.update_document(
            project_id=project.id, document_id=doc.id, name="Renamed.pdf"
        )

        assert updated.name == "Renamed.pdf"
        fetched = await service.get_document(project_id=project.id, document_id=doc.id)
        assert fetched.name == "Renamed.pdf"


@pytest.mark.anyio
async def test_update_document_empty_name():
    """Verify updating document name to empty raises InvalidDocumentDataError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="Doc.pdf", file_data=b"content"
        )
        with pytest.raises(InvalidDocumentDataError):
            await service.update_document(
                project_id=project.id, document_id=doc.id, name="   "
            )


@pytest.mark.anyio
async def test_delete_document_and_storage_cleanup():
    """Verify deleting a document removes it from DB and triggers storage cleanup for all versions."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="Doc_v1.pdf", file_data=b"v1"
        )
        await service.create_document_version(
            project_id=project.id, document_id=doc.id, filename="Doc_v2.pdf", file_data=b"v2"
        )

        # Delete
        await service.delete_document(project_id=project.id, document_id=doc.id)

        # Verify not found in DB
        with pytest.raises(DocumentNotFoundError):
            await service.get_document(project_id=project.id, document_id=doc.id)

        # Verify storage delete_many was called with both version paths
        storage.delete_many.assert_awaited_once()
        deleted_paths = storage.delete_many.call_args[0][0]
        assert len(deleted_paths) == 2
        assert f"{project.id}/{doc.id}/v1/Doc_v1.pdf" in deleted_paths
        assert f"{project.id}/{doc.id}/v2/Doc_v2.pdf" in deleted_paths


# ==============================================================================
# 5. Document Version Lookup Tests
# ==============================================================================


@pytest.mark.anyio
async def test_get_document_version_success():
    """Verify retrieving specific document version by version number."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="file.pdf", file_data=b"v1 content"
        )
        ver1 = await service.get_document_version(
            project_id=project.id, document_id=doc.id, version_number=1
        )
        assert ver1.version_number == 1
        assert ver1.original_filename == "file.pdf"


@pytest.mark.anyio
async def test_get_document_version_not_found():
    """Verify non-existent version number raises DocumentVersionNotFoundError."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        project = await seed_project(session)
        service = DocumentService(session=session, storage=storage)

        doc = await service.create_document(
            project_id=project.id, filename="file.pdf", file_data=b"v1 content"
        )
        with pytest.raises(DocumentVersionNotFoundError):
            await service.get_document_version(
                project_id=project.id, document_id=doc.id, version_number=99
            )
