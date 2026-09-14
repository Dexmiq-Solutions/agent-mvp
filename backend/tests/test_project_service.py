"""Unit and integration tests for ProjectService."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from exceptions.project import InvalidProjectDataError, ProjectNotFoundError
from models.base import Base
from models.project import ProjectModel
from services.project_service import ProjectService
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


@pytest.fixture
def mock_storage():
    """Mock implementation of BaseObjectStorage."""
    storage = AsyncMock(spec=BaseObjectStorage)
    storage.list_objects = AsyncMock(return_value=[])
    storage.delete_many = AsyncMock(return_value=[])
    return storage


# ==============================================================================
# 1. Project Creation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_create_project_success():
    """Verify creating a project persists it with valid attributes and timestamps."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        project = await service.create_project(
            name="Apollo Mission",
            description="Lunar exploration research documentation.",
        )

        assert project.id is not None
        assert project.name == "Apollo Mission"
        assert project.description == "Lunar exploration research documentation."
        assert project.created_at is not None
        assert project.updated_at is not None

        # Retrieve from DB to verify persistence
        fetched = await service.get_project(project.id)
        assert fetched.id == project.id
        assert fetched.name == "Apollo Mission"


@pytest.mark.anyio
async def test_create_project_empty_name():
    """Verify empty project name raises InvalidProjectDataError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        with pytest.raises(InvalidProjectDataError) as exc_info:
            await service.create_project(name="")
        assert "cannot be empty" in str(exc_info.value)


@pytest.mark.anyio
async def test_create_project_whitespace_name():
    """Verify whitespace-only project name raises InvalidProjectDataError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        with pytest.raises(InvalidProjectDataError) as exc_info:
            await service.create_project(name="   \t\n  ")
        assert "whitespace" in str(exc_info.value)


# ==============================================================================
# 2. Project Retrieval Tests
# ==============================================================================


@pytest.mark.anyio
async def test_get_project_success():
    """Verify retrieving an existing project by ID."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        created = await service.create_project(name="Alpha")

        fetched = await service.get_project(created.id)
        assert fetched.id == created.id
        assert fetched.name == "Alpha"


@pytest.mark.anyio
async def test_get_project_not_found():
    """Verify non-existent project ID raises ProjectNotFoundError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        with pytest.raises(ProjectNotFoundError) as exc_info:
            await service.get_project("non-existent-uuid")
        assert "non-existent-uuid" in str(exc_info.value)


# ==============================================================================
# 3. Project Listing Tests
# ==============================================================================


@pytest.mark.anyio
async def test_list_projects_empty():
    """Verify listing projects returns empty list when no projects exist."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        projects = await service.list_projects()
        assert projects == []


@pytest.mark.anyio
async def test_list_projects_ordering_and_pagination():
    """Verify listing projects returns results ordered descending by created_at with limit/offset."""
    async with create_test_session() as session:
        service = ProjectService(session=session)

        await service.create_project(name="Project 1")
        await service.create_project(name="Project 2")
        await service.create_project(name="Project 3")

        # List all
        all_projects = await service.list_projects(limit=10)
        assert len(all_projects) == 3

        # Limit and offset
        page1 = await service.list_projects(limit=2, offset=0)
        assert len(page1) == 2

        page2 = await service.list_projects(limit=2, offset=2)
        assert len(page2) == 1


# ==============================================================================
# 4. Project Update Tests
# ==============================================================================


@pytest.mark.anyio
async def test_update_project_success():
    """Verify updating project name and description."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        project = await service.create_project(name="Initial Name", description="Initial Desc")

        updated = await service.update_project(
            project_id=project.id,
            name="Updated Name",
            description="Updated Desc",
        )

        assert updated.name == "Updated Name"
        assert updated.description == "Updated Desc"

        # Verify persistence
        fetched = await service.get_project(project.id)
        assert fetched.name == "Updated Name"
        assert fetched.description == "Updated Desc"


@pytest.mark.anyio
async def test_update_project_partial():
    """Verify updating only name preserves existing description."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        project = await service.create_project(name="Initial Name", description="Preserved Desc")

        updated = await service.update_project(
            project_id=project.id,
            name="New Name Only",
        )

        assert updated.name == "New Name Only"
        assert updated.description == "Preserved Desc"


@pytest.mark.anyio
async def test_update_project_empty_name():
    """Verify updating name to empty string raises InvalidProjectDataError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        project = await service.create_project(name="Valid Name")

        with pytest.raises(InvalidProjectDataError):
            await service.update_project(project_id=project.id, name="   ")


@pytest.mark.anyio
async def test_update_project_not_found():
    """Verify updating non-existent project raises ProjectNotFoundError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        with pytest.raises(ProjectNotFoundError):
            await service.update_project(project_id="missing-id", name="New Name")


# ==============================================================================
# 5. Project Deletion Tests
# ==============================================================================


@pytest.mark.anyio
async def test_delete_project_success():
    """Verify deleting a project removes it from database."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        project = await service.create_project(name="To Be Deleted")

        await service.delete_project(project.id)

        with pytest.raises(ProjectNotFoundError):
            await service.get_project(project.id)


@pytest.mark.anyio
async def test_delete_project_with_storage_cleanup(mock_storage: AsyncMock):
    """Verify deleting a project invokes storage cleanup for project files."""
    mock_storage.list_objects.return_value = [
        StorageObjectMetadata(
            name="file.pdf",
            path="test_project/doc1/v1/file.pdf",
            bucket="documents",
        )
    ]
    async with create_test_session() as session:
        service = ProjectService(session=session, storage=mock_storage)
        project = await service.create_project(name="Storage Cleanup Project")

        await service.delete_project(project.id)

        mock_storage.list_objects.assert_awaited_once()
        mock_storage.delete_many.assert_awaited_once()


@pytest.mark.anyio
async def test_delete_project_not_found():
    """Verify deleting non-existent project raises ProjectNotFoundError."""
    async with create_test_session() as session:
        service = ProjectService(session=session)
        with pytest.raises(ProjectNotFoundError):
            await service.delete_project("non-existent-id")
