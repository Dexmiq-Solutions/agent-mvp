"""Unit and integration tests verifying strict Project Isolation boundaries."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from exceptions.document import (
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
)
from models.base import Base
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
    """Create a mock BaseObjectStorage."""
    storage = AsyncMock(spec=BaseObjectStorage)

    async def fake_upload(path, data, content_type=None, upsert=False):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=len(data) if isinstance(data, bytes) else 512,
            content_type=content_type or "application/pdf",
            etag="etag-isolation",
        )

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    return storage


async def seed_tenant_setup(session: AsyncSession, storage: BaseObjectStorage):
    """Seed two distinct projects (Tenant A and Tenant B) with their respective documents."""
    service = DocumentService(session=session, storage=storage)

    proj_a = ProjectModel(name="Tenant Alpha")
    proj_b = ProjectModel(name="Tenant Beta")
    session.add_all([proj_a, proj_b])
    await session.flush()

    doc_a = await service.create_document(
        project_id=proj_a.id,
        filename="alpha_secret.pdf",
        file_data=b"Alpha confidential content",
        name="Alpha Secret Doc",
    )
    doc_b = await service.create_document(
        project_id=proj_b.id,
        filename="beta_secret.pdf",
        file_data=b"Beta confidential content",
        name="Beta Secret Doc",
    )

    return service, proj_a, proj_b, doc_a, doc_b


# ==============================================================================
# Project Isolation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_isolation_list_sources():
    """Verify Tenant A source listing never contains Tenant B documents, and vice versa."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        # Listing for Project A
        a_sources = await service.list_documents(project_id=proj_a.id)
        a_source_ids = {doc.id for doc in a_sources}
        assert doc_a.id in a_source_ids
        assert doc_b.id not in a_source_ids

        # Listing for Project B
        b_sources = await service.list_documents(project_id=proj_b.id)
        b_source_ids = {doc.id for doc in b_sources}
        assert doc_b.id in b_source_ids
        assert doc_a.id not in b_source_ids


@pytest.mark.anyio
async def test_isolation_get_source():
    """Verify Project A cannot retrieve Project B document, returning 404 (no leak)."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        # Cross-tenant read: Project A attempting to read Document B
        with pytest.raises(DocumentNotFoundError) as exc_info:
            await service.get_document(project_id=proj_a.id, document_id=doc_b.id)
        assert doc_b.id in str(exc_info.value)
        assert proj_a.id in str(exc_info.value)

        # Cross-tenant read: Project B attempting to read Document A
        with pytest.raises(DocumentNotFoundError) as exc_info:
            await service.get_document(project_id=proj_b.id, document_id=doc_a.id)
        assert doc_a.id in str(exc_info.value)
        assert proj_b.id in str(exc_info.value)


@pytest.mark.anyio
async def test_isolation_update_source():
    """Verify Project A cannot update Project B document, and document is not mutated."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        # Attempt to modify Doc B using Project A route
        with pytest.raises(DocumentNotFoundError):
            await service.update_document(
                project_id=proj_a.id,
                document_id=doc_b.id,
                name="Hacked Title",
            )

        # Verify Doc B was not modified
        original_b = await service.get_document(project_id=proj_b.id, document_id=doc_b.id)
        assert original_b.name == "Beta Secret Doc"


@pytest.mark.anyio
async def test_isolation_delete_source():
    """Verify Project A cannot delete Project B document, and document persists intact."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        # Attempt cross-tenant deletion
        with pytest.raises(DocumentNotFoundError):
            await service.delete_document(project_id=proj_a.id, document_id=doc_b.id)

        # Verify Doc B is still present under Project B
        recheck_b = await service.get_document(project_id=proj_b.id, document_id=doc_b.id)
        assert recheck_b.id == doc_b.id
        assert storage.delete_many.call_count == 0


@pytest.mark.anyio
async def test_isolation_create_version():
    """Verify Project A cannot append a new version to Project B document."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        with pytest.raises(DocumentNotFoundError):
            await service.create_document_version(
                project_id=proj_a.id,
                document_id=doc_b.id,
                filename="injected.pdf",
                file_data=b"malicious content",
            )


@pytest.mark.anyio
async def test_isolation_get_version():
    """Verify Project A cannot retrieve version coordinates of Project B document."""
    storage = make_mock_storage()
    async with create_test_session() as session:
        service, proj_a, proj_b, doc_a, doc_b = await seed_tenant_setup(session, storage)

        with pytest.raises(DocumentVersionNotFoundError):
            await service.get_document_version(
                project_id=proj_a.id,
                document_id=doc_b.id,
                version_number=1,
            )
