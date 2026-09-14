"""Comprehensive unit and integration tests for Document Processing Lifecycle."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from exceptions.document import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentStorageSourceError,
    DocumentVersionMismatchError,
    DocumentVersionNotFoundError,
    InvalidDocumentStateTransitionError,
    ProjectDocumentMismatchError,
)
from exceptions.project import ProjectNotFoundError
from models.base import Base
from models.document import DocumentModel, DocumentVersionModel, DocumentVersionStatus
from models.project import ProjectModel
from rag.pipeline.base import BaseDocumentProcessingPipeline
from services.document_processing_service import DocumentProcessingService, sanitize_error_message
from services.document_service import DocumentService
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_db() -> AsyncIterator[tuple[AsyncSession, async_sessionmaker[AsyncSession]]]:
    """Create an isolated in-memory SQLite database and yield session and session factory."""
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
        yield session, session_factory

    await engine.dispose()


def make_mock_storage() -> AsyncMock:
    """Create a mock BaseObjectStorage for test assertions."""
    storage = AsyncMock(spec=BaseObjectStorage)

    async def fake_upload(path, data, content_type=None, upsert=False):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=len(data) if isinstance(data, bytes) else 512,
            content_type=content_type or "application/pdf",
            etag="etag-lifecycle-test",
        )

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    storage.get_metadata = AsyncMock(
        return_value=StorageObjectMetadata(
            name="file.pdf",
            path="p/d/v1/file.pdf",
            bucket="documents",
            size_bytes=512,
            content_type="application/pdf",
            etag="etag-lifecycle-test",
        )
    )
    return storage


class MockPipeline(BaseDocumentProcessingPipeline):
    """Pipeline stub capturing invocations and supporting simulated failures."""

    def __init__(self, should_fail: bool = False, fail_message: str = "Pipeline processing stage error"):
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.invocations: list[dict] = []

    async def process_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        storage_bucket: str,
        storage_path: str,
        original_filename: str,
        content_type: str | None = None,
    ):
        self.invocations.append({
            "project_id": project_id,
            "document_id": document_id,
            "document_version_id": document_version_id,
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "original_filename": original_filename,
            "content_type": content_type,
        })
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        return {"status": "success", "document_version_id": document_version_id}


async def seed_hierarchy(session: AsyncSession) -> tuple[ProjectModel, DocumentModel, DocumentVersionModel]:
    """Helper to seed a Project, Document, and DocumentVersion in PENDING state."""
    project = ProjectModel(name="Lifecycle Test Project")
    session.add(project)
    await session.flush()

    doc = DocumentModel(project_id=project.id, name="Lifecycle Test Doc")
    session.add(doc)
    await session.flush()

    version = DocumentVersionModel(
        document_id=doc.id,
        project_id=project.id,
        version_number=1,
        storage_bucket="documents",
        storage_path=f"{project.id}/{doc.id}/v1/sample.pdf",
        original_filename="sample.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        status=DocumentVersionStatus.PENDING.value,
    )
    session.add(version)
    await session.commit()
    return project, doc, version


# ==============================================================================
# 1. State Machine & Lifecycle Transition Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lifecycle_pending_to_indexing_to_ready():
    """Verify full successful transition: PENDING -> INDEXING -> READY."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        pipeline = MockPipeline()
        service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )

        # Execute processing
        updated_version = await service.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
        )

        assert updated_version.status == DocumentVersionStatus.READY.value
        assert updated_version.indexed_at is not None
        assert updated_version.error_message is None

        # Verify persisted in database
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.READY.value
            assert db_ver.indexed_at is not None
            assert db_ver.error_message is None

        # Verify pipeline was called with exact identifiers
        assert len(pipeline.invocations) == 1
        inv = pipeline.invocations[0]
        assert inv["project_id"] == project.id
        assert inv["document_id"] == doc.id
        assert inv["document_version_id"] == version.id
        assert inv["storage_path"] == version.storage_path


@pytest.mark.anyio
async def test_lifecycle_pending_to_indexing_to_failed():
    """Verify pipeline failure transitions: PENDING -> INDEXING -> FAILED with error persistence."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        pipeline = MockPipeline(should_fail=True, fail_message="Embedding model timed out")
        service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )

        with pytest.raises(DocumentProcessingError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        assert "Embedding model timed out" in str(exc_info.value)

        # Verify persisted state is FAILED, not READY
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.FAILED.value
            assert db_ver.indexed_at is None
            assert "Embedding model timed out" in db_ver.error_message


@pytest.mark.anyio
async def test_lifecycle_disallow_ready_to_indexing():
    """Verify that a READY version cannot be transitioned to INDEXING (reprocessing not supported)."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        version.status = DocumentVersionStatus.READY.value
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        with pytest.raises(InvalidDocumentStateTransitionError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        assert exc_info.value.current_status == "ready"
        assert exc_info.value.target_status == "indexing"


@pytest.mark.anyio
async def test_lifecycle_disallow_pending_to_ready_direct():
    """Verify direct invalid transition from PENDING to READY is rejected."""
    service = DocumentProcessingService()
    with pytest.raises(InvalidDocumentStateTransitionError):
        service.validate_state_transition(
            version_id="ver-1",
            current_status=DocumentVersionStatus.PENDING.value,
            target_status=DocumentVersionStatus.READY.value,
        )


@pytest.mark.anyio
async def test_lifecycle_disallow_failed_to_indexing_direct():
    """Verify that transitioning a FAILED version to INDEXING is rejected."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        version.status = DocumentVersionStatus.FAILED.value
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        with pytest.raises(InvalidDocumentStateTransitionError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        assert exc_info.value.current_status == "failed"


# ==============================================================================
# 2. Concurrency & Duplicate Processing Protection Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lifecycle_duplicate_processing_rejected():
    """Verify that requesting processing on an actively INDEXING version raises DocumentAlreadyProcessingError."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        version.status = DocumentVersionStatus.INDEXING.value
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        with pytest.raises(DocumentAlreadyProcessingError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        assert exc_info.value.version_id == version.id


# ==============================================================================
# 3. Version Isolation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lifecycle_version_isolation():
    """Verify that processing Version 2 does not mutate or overwrite Version 1."""
    async with create_test_db() as (session, session_factory):
        project, doc, v1 = await seed_hierarchy(session)

        # Process Version 1 to READY
        pipeline = MockPipeline()
        service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )
        await service.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=v1.id,
        )

        # Create Version 2 for the same document
        v2 = DocumentVersionModel(
            document_id=doc.id,
            project_id=project.id,
            version_number=2,
            storage_bucket="documents",
            storage_path=f"{project.id}/{doc.id}/v2/sample_rev2.pdf",
            original_filename="sample_rev2.pdf",
            content_type="application/pdf",
            size_bytes=2048,
            status=DocumentVersionStatus.PENDING.value,
        )
        session.add(v2)
        await session.commit()

        # Process Version 2 with a failure
        fail_pipeline = MockPipeline(should_fail=True, fail_message="Chunking failed")
        service_fail = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=fail_pipeline,
        )
        with pytest.raises(DocumentProcessingError):
            await service_fail.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=v2.id,
            )

        # Verify Version 1 remains READY and unmutated
        async with session_factory() as check_session:
            v1_check = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == v1.id)
                )
            ).scalar_one()
            v2_check = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == v2.id)
                )
            ).scalar_one()

            assert v1_check.status == DocumentVersionStatus.READY.value
            assert v1_check.indexed_at is not None
            assert v1_check.error_message is None

            assert v2_check.status == DocumentVersionStatus.FAILED.value
            assert v2_check.indexed_at is None
            assert "Chunking failed" in v2_check.error_message


# ==============================================================================
# 4. Project & Tenant Ownership Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lifecycle_project_mismatch_rejected():
    """Verify that attempting to process a document with the wrong project_id raises an isolation error."""
    async with create_test_db() as (session, session_factory):
        p1, doc, version = await seed_hierarchy(session)

        # Create a second project
        p2 = ProjectModel(name="Attacker Project")
        session.add(p2)
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        # Project 2 tries to process Document belonging to Project 1
        with pytest.raises(ProjectDocumentMismatchError):
            await service.process_document_version(
                project_id=p2.id,
                document_id=doc.id,
                document_version_id=version.id,
            )


@pytest.mark.anyio
async def test_lifecycle_document_mismatch_rejected():
    """Verify that specifying a version belonging to Document D1 under Document D2 raises an error."""
    async with create_test_db() as (session, session_factory):
        project, doc1, version1 = await seed_hierarchy(session)

        # Create a second document in the same project
        doc2 = DocumentModel(project_id=project.id, name="Other Doc")
        session.add(doc2)
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        # Try to process version1 claiming it belongs to doc2
        with pytest.raises(DocumentVersionMismatchError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc2.id,
                document_version_id=version1.id,
            )

        assert exc_info.value.expected_parent == doc2.id
        assert exc_info.value.actual_parent == doc1.id


@pytest.mark.anyio
async def test_lifecycle_nonexistent_entities():
    """Verify proper 404/not found domain exceptions for missing project, doc, or version."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        service = DocumentProcessingService(session_maker=session_factory)

        # Missing project
        with pytest.raises(ProjectNotFoundError):
            await service.process_document_version("missing-proj", doc.id, version.id)

        # Missing document
        with pytest.raises(DocumentNotFoundError):
            await service.process_document_version(project.id, "missing-doc", version.id)

        # Missing version
        with pytest.raises(DocumentVersionNotFoundError):
            await service.process_document_version(project.id, doc.id, "missing-ver")


# ==============================================================================
# 5. Storage Source Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lifecycle_missing_storage_bucket_fails():
    """Verify missing storage bucket marks version FAILED and raises DocumentStorageSourceError."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        version.storage_bucket = ""
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        with pytest.raises(DocumentStorageSourceError):
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        # Verify state is FAILED in database
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.FAILED.value
            assert "Storage bucket missing" in db_ver.error_message


@pytest.mark.anyio
async def test_lifecycle_missing_storage_path_fails():
    """Verify missing storage path marks version FAILED and raises DocumentStorageSourceError."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        version.storage_path = ""
        await session.commit()

        service = DocumentProcessingService(session_maker=session_factory)

        with pytest.raises(DocumentStorageSourceError):
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.FAILED.value
            assert "Storage path missing" in db_ver.error_message


# ==============================================================================
# 6. Error Sanitization Tests
# ==============================================================================


def test_sanitize_error_message_masks_secrets():
    """Verify error sanitization strips DB passwords, OpenAI keys, and Supabase tokens."""
    err1 = Exception("Failed connecting to postgresql://postgres:secret123@db.supabase.co:5432/postgres")
    assert "secret123" not in sanitize_error_message(err1)
    assert "postgresql://postgres:***@db.supabase.co" in sanitize_error_message(err1)

    err2 = Exception("Invalid key sk-proj-1234567890abcdefghijklmn provided")
    assert "1234567890abcdefghijklmn" not in sanitize_error_message(err2)
    assert "sk-***" in sanitize_error_message(err2)


# ==============================================================================
# 7. Trigger & Execution Mode Tests
# ==============================================================================


@pytest.mark.anyio
async def test_trigger_processing_synchronous():
    """Verify trigger_processing in synchronous mode returns updated version."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        pipeline = MockPipeline()
        service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )

        result = await service.trigger_processing(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            background=False,
        )
        assert result.status == DocumentVersionStatus.READY.value


@pytest.mark.anyio
async def test_trigger_processing_background():
    """Verify trigger_processing in background mode returns an asyncio.Task that completes."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        pipeline = MockPipeline()
        service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )

        task = await service.trigger_processing(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            background=True,
        )
        assert isinstance(task, asyncio.Task)
        await task

        # Check DB state
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.READY.value


@pytest.mark.anyio
async def test_document_service_auto_process_integration():
    """Verify DocumentService triggers processing when auto_process=True."""
    async with create_test_db() as (session, session_factory):
        project = ProjectModel(name="Auto Process Project")
        session.add(project)
        await session.commit()

        storage = make_mock_storage()
        pipeline = MockPipeline()
        processing_service = DocumentProcessingService(
            session_maker=session_factory,
            pipeline=pipeline,
        )

        doc_service = DocumentService(
            session=session,
            storage=storage,
            processing_service=processing_service,
            auto_process=True,
        )

        # 1. Create document with auto_process=True
        doc = await doc_service.create_document(
            project_id=project.id,
            filename="auto_doc.pdf",
            file_data=b"Sample PDF data",
        )

        # Verify initial version was processed to READY
        async with session_factory() as check_session:
            v1 = (
                await check_session.execute(
                    select(DocumentVersionModel).where(
                        DocumentVersionModel.document_id == doc.id,
                        DocumentVersionModel.version_number == 1,
                    )
                )
            ).scalar_one()
            assert v1.status == DocumentVersionStatus.READY.value
            assert v1.indexed_at is not None

        # 2. Create version 2 with auto_process=True
        v2 = await doc_service.create_document_version(
            project_id=project.id,
            document_id=doc.id,
            filename="auto_doc_v2.pdf",
            file_data=b"Sample PDF revision data",
        )
        assert v2.status == DocumentVersionStatus.READY.value
        assert len(pipeline.invocations) == 2


@pytest.mark.anyio
async def test_document_service_get_version_by_id():
    """Verify DocumentService.get_version_by_id retrieves version with isolation."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_hierarchy(session)
        storage = make_mock_storage()
        doc_service = DocumentService(session=session, storage=storage)

        fetched = await doc_service.get_version_by_id(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
        )
        assert fetched.id == version.id
        assert fetched.version_number == 1

        # Mismatched project
        with pytest.raises(DocumentVersionNotFoundError):
            await doc_service.get_version_by_id(
                project_id="other-project",
                document_id=doc.id,
                document_version_id=version.id,
            )
