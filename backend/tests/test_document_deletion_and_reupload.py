"""Comprehensive verification tests for Document Upload, Deletion, Qdrant Vector Integrity, and Re-upload.

Verifies:
1. Deleting a document removes its associated vector points from the vector store.
2. No stale vector embeddings remain for the deleted document.
3. A deleted document can be re-uploaded normally.
4. The re-uploaded document receives fresh IDs, chunks, and embeddings without reusing stale vectors.
5. Project isolation is strictly maintained throughout upload, deletion, and re-upload.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import io
from typing import Any, Optional
from unittest.mock import AsyncMock
import uuid

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from db.base import Base
from db.session import get_db_session
from exceptions.document import DocumentNotFoundError
from models.chunk import ChunkModel
from models.document import DocumentModel, DocumentVersionModel
from models.project import ProjectModel
from services.document_service import DocumentService
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata
from storage.vector.base import BaseVectorStore
from storage.vector.models import VectorPayload, VectorRecord, VectorSearchResult


class InMemoryVectorStore(BaseVectorStore):
    """In-memory implementation of BaseVectorStore for verifying vector deletion & isolation."""

    def __init__(self, collection_name: str = "test_collection") -> None:
        self._collection_name = collection_name
        self.records: dict[str, VectorRecord] = {}
        self.delete_by_filter_calls: list[dict[str, Any]] = []

    @property
    def collection_name(self) -> str:
        return self._collection_name

    async def ensure_collection_exists(
        self, vector_size: Optional[int] = None, distance: str = "Cosine"
    ) -> bool:
        return True

    async def upsert(self, records: list[VectorRecord]) -> int:
        for r in records:
            self.records[str(r.id)] = r
        return len(records)

    async def delete(self, point_ids: list[str | int]) -> int:
        count = 0
        for pid in point_ids:
            if str(pid) in self.records:
                del self.records[str(pid)]
                count += 1
        return count

    async def delete_by_filter(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        document_version_id: Optional[str] = None,
    ) -> bool:
        self.delete_by_filter_calls.append({
            "project_id": project_id,
            "document_id": document_id,
            "document_version_id": document_version_id,
        })
        to_delete = []
        for pid, record in self.records.items():
            payload = record.payload
            if payload.project_id != project_id:
                continue
            if document_id is not None and payload.document_id != document_id:
                continue
            if document_version_id is not None and payload.document_version_id != document_version_id:
                continue
            to_delete.append(pid)

        for pid in to_delete:
            del self.records[pid]
        return True

    async def search(
        self,
        query_vector: list[float],
        project_id: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        document_version_id: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        results = []
        for record in self.records.values():
            if record.payload.project_id != project_id:
                continue
            if document_id is not None and record.payload.document_id != document_id:
                continue
            results.append(
                VectorSearchResult(
                    id=record.id,
                    score=1.0,
                    payload=record.payload,
                    vector=record.vector,
                )
            )
        return results[:limit]


def make_mock_storage() -> AsyncMock:
    storage = AsyncMock(spec=BaseObjectStorage)
    storage.bucket_name = "documents"

    async def fake_upload(path, data, content_type=None, upsert=False):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=len(data) if isinstance(data, bytes) else 1024,
            content_type=content_type or "application/pdf",
            etag="etag-test",
        )

    async def fake_get_metadata(path):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=1024,
            content_type="application/pdf",
            etag="etag-test",
        )

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.get_metadata = AsyncMock(side_effect=fake_get_metadata)
    storage.download = AsyncMock(return_value=b"%PDF-1.4 file content")
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    storage.list_objects = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        finally:
            await session.close()
            await engine.dispose()


@pytest.mark.anyio
async def test_document_lifecycle_upload_delete_and_reupload_with_vector_integrity():
    """Verify upload, deletion (including Qdrant vector removal), and re-upload flow."""
    vector_store = InMemoryVectorStore()
    storage = make_mock_storage()

    async with create_test_session() as session:
        # Create Project A and Project B for isolation testing
        proj_a = ProjectModel(id=str(uuid.uuid4()), name="Project Alpha")
        proj_b = ProjectModel(id=str(uuid.uuid4()), name="Project Beta")
        session.add_all([proj_a, proj_b])
        await session.flush()

        service = DocumentService(session=session, storage=storage, vector_store=vector_store)

        # 1. Upload Document 1 ("requirements.md") to Project A
        doc1 = await service.create_document(
            project_id=proj_a.id,
            filename="requirements.md",
            file_data=b"# Requirements Doc 1",
        )
        # Simulate indexed vector points for doc1 in vector store
        ver1_id = (await service.get_document(proj_a.id, doc1.id)).versions[0].id
        chunk1_id = str(uuid.uuid4())
        point1_id = str(uuid.uuid4())
        await vector_store.upsert([
            VectorRecord(
                id=point1_id,
                vector=[0.1] * 1024,
                payload=VectorPayload(
                    project_id=proj_a.id,
                    document_id=doc1.id,
                    chunk_id=chunk1_id,
                    document_version_id=ver1_id,
                    metadata={"title": "Doc1 Chunk 1"},
                ),
            )
        ])

        # 2. Upload Document 2 ("architecture.md") to Project A (sibling document)
        doc2 = await service.create_document(
            project_id=proj_a.id,
            filename="architecture.md",
            file_data=b"# Architecture Doc",
        )
        ver2_id = (await service.get_document(proj_a.id, doc2.id)).versions[0].id
        point2_id = str(uuid.uuid4())
        await vector_store.upsert([
            VectorRecord(
                id=point2_id,
                vector=[0.2] * 1024,
                payload=VectorPayload(
                    project_id=proj_a.id,
                    document_id=doc2.id,
                    chunk_id=str(uuid.uuid4()),
                    document_version_id=ver2_id,
                ),
            )
        ])

        # 3. Upload Document 3 ("requirements.md") to Project B (same filename, cross-project tenant)
        doc3 = await service.create_document(
            project_id=proj_b.id,
            filename="requirements.md",
            file_data=b"# Project B Requirements",
        )
        ver3_id = (await service.get_document(proj_b.id, doc3.id)).versions[0].id
        point3_id = str(uuid.uuid4())
        await vector_store.upsert([
            VectorRecord(
                id=point3_id,
                vector=[0.3] * 1024,
                payload=VectorPayload(
                    project_id=proj_b.id,
                    document_id=doc3.id,
                    chunk_id=str(uuid.uuid4()),
                    document_version_id=ver3_id,
                ),
            )
        ])

        # Verify all 3 points currently exist in vector store
        assert len(vector_store.records) == 3
        assert point1_id in vector_store.records
        assert point2_id in vector_store.records
        assert point3_id in vector_store.records

        # 4. DELETE Document 1 from Project A
        await service.delete_document(project_id=proj_a.id, document_id=doc1.id)

        # 5. Verify Qdrant Deletion Integrity:
        # - Vector for doc1 must be removed
        assert point1_id not in vector_store.records
        # - Vector for doc2 in Project A must remain intact (document isolation within project)
        assert point2_id in vector_store.records
        # - Vector for doc3 in Project B must remain intact (project tenant isolation)
        assert point3_id in vector_store.records
        assert len(vector_store.records) == 2

        # Verify delete_by_filter was executed with project_id and document_id
        assert len(vector_store.delete_by_filter_calls) == 1
        assert vector_store.delete_by_filter_calls[0] == {
            "project_id": proj_a.id,
            "document_id": doc1.id,
            "document_version_id": None,
        }

        # Verify Document 1 no longer exists in PostgreSQL
        with pytest.raises(DocumentNotFoundError):
            await service.get_document(project_id=proj_a.id, document_id=doc1.id)

        # 6. RE-UPLOAD: Upload the same document ("requirements.md") to Project A again
        reuploaded_doc = await service.create_document(
            project_id=proj_a.id,
            filename="requirements.md",
            file_data=b"# Re-uploaded Requirements Doc 1",
        )

        # Must receive a fresh document ID
        assert reuploaded_doc.id != doc1.id
        assert reuploaded_doc.name == "requirements.md"

        # Index fresh vector point for re-uploaded document
        reupload_ver_id = (await service.get_document(proj_a.id, reuploaded_doc.id)).versions[0].id
        point_reuploaded_id = str(uuid.uuid4())
        await vector_store.upsert([
            VectorRecord(
                id=point_reuploaded_id,
                vector=[0.15] * 1024,
                payload=VectorPayload(
                    project_id=proj_a.id,
                    document_id=reuploaded_doc.id,
                    chunk_id=str(uuid.uuid4()),
                    document_version_id=reupload_ver_id,
                ),
            )
        ])

        # 7. Verify search in Project A:
        # Search returns only new vector and doc2 vector, NOT old doc1 vector
        search_res = await vector_store.search(
            query_vector=[0.1] * 1024,
            project_id=proj_a.id,
        )
        result_ids = {r.id for r in search_res}
        assert point1_id not in result_ids
        assert point_reuploaded_id in result_ids
        assert point2_id in result_ids


@pytest.mark.anyio
async def test_cross_project_document_deletion_forbidden():
    """Verify that deleting a document from an incorrect project raises 404 and does not delete vectors."""
    vector_store = InMemoryVectorStore()
    storage = make_mock_storage()

    async with create_test_session() as session:
        proj_a = ProjectModel(id=str(uuid.uuid4()), name="Proj A")
        proj_b = ProjectModel(id=str(uuid.uuid4()), name="Proj B")
        session.add_all([proj_a, proj_b])
        await session.flush()

        service = DocumentService(session=session, storage=storage, vector_store=vector_store)

        doc_a = await service.create_document(
            project_id=proj_a.id, filename="doc_a.pdf", file_data=b"doc a"
        )
        point_id = str(uuid.uuid4())
        await vector_store.upsert([
            VectorRecord(
                id=point_id,
                vector=[0.5] * 1024,
                payload=VectorPayload(
                    project_id=proj_a.id,
                    document_id=doc_a.id,
                    chunk_id=str(uuid.uuid4()),
                ),
            )
        ])

        # Attempt to delete doc_a through Project B's boundary
        with pytest.raises(DocumentNotFoundError):
            await service.delete_document(project_id=proj_b.id, document_id=doc_a.id)

        # Verify vector was NOT touched
        assert point_id in vector_store.records
        assert len(vector_store.delete_by_filter_calls) == 0


@pytest.mark.anyio
async def test_api_source_upload_delete_and_reupload_flow(monkeypatch):
    """Full HTTP API end-to-end test verifying upload, delete (with vectors), and re-upload."""
    from core.config import get_settings
    monkeypatch.setattr(get_settings(), "AUTO_PROCESS_DOCUMENTS", False)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    mock_storage = make_mock_storage()
    mock_vector_store = InMemoryVectorStore()

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    from api.dependencies import get_storage, get_vector_store_dependency
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Create Project
        p_resp = await client.post("/projects", json={"name": "Lifecycle Project"})
        project_id = p_resp.json()["id"]

        # 2. Upload source document ("specs.pdf")
        files = {"file": ("specs.pdf", io.BytesIO(b"%PDF-1.4 file content"), "application/pdf")}
        up1_resp = await client.post(f"/projects/{project_id}/sources", files=files)
        assert up1_resp.status_code == 201
        doc1_data = up1_resp.json()
        doc1_id = doc1_data["id"]
        assert doc1_data["name"] == "specs.pdf"

        # Simulate Qdrant points existing for doc1
        point_id = str(uuid.uuid4())
        await mock_vector_store.upsert([
            VectorRecord(
                id=point_id,
                vector=[0.42] * 1024,
                payload=VectorPayload(
                    project_id=project_id,
                    document_id=doc1_id,
                    chunk_id=str(uuid.uuid4()),
                ),
            )
        ])
        assert point_id in mock_vector_store.records

        # 3. List sources to verify presence
        list_resp = await client.get(f"/projects/{project_id}/sources")
        assert list_resp.status_code == 200
        sources = list_resp.json()
        assert len(sources) == 1
        assert sources[0]["id"] == doc1_id

        # 4. DELETE source document
        del_resp = await client.delete(f"/projects/{project_id}/sources/{doc1_id}")
        assert del_resp.status_code == 204

        # 5. Verify Qdrant points were deleted
        assert point_id not in mock_vector_store.records
        assert len(mock_vector_store.delete_by_filter_calls) == 1
        assert mock_vector_store.delete_by_filter_calls[0]["document_id"] == doc1_id

        # 6. Verify source list is now empty
        list_empty_resp = await client.get(f"/projects/{project_id}/sources")
        assert list_empty_resp.status_code == 200
        assert len(list_empty_resp.json()) == 0

        # 7. RE-UPLOAD the same document ("specs.pdf")
        files_reupload = {"file": ("specs.pdf", io.BytesIO(b"%PDF-1.4 file content v2"), "application/pdf")}
        up2_resp = await client.post(f"/projects/{project_id}/sources", files=files_reupload)
        assert up2_resp.status_code == 201
        doc2_data = up2_resp.json()
        doc2_id = doc2_data["id"]

        # Verify a new document ID was generated
        assert doc2_id != doc1_id
        assert doc2_data["name"] == "specs.pdf"

        # 8. Verify source list now contains the re-uploaded document
        list_re_resp = await client.get(f"/projects/{project_id}/sources")
        assert list_re_resp.status_code == 200
        re_sources = list_re_resp.json()
        assert len(re_sources) == 1
        assert re_sources[0]["id"] == doc2_id

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_special_character_filename_upload_processing_delete_and_reupload():
    """Verify full lifecycle: upload with Unicode/special characters, safe key generation,
    processing continuation (acquisition + ingestion), deletion cleanup, and re-upload.
    """
    special_filename = "Discovery Notes v1 — Dexmiq Website Revamp & Product-Centric Repositioning.docx"
    safe_storage_filename = "Discovery_Notes_v1_Dexmiq_Website_Revamp_Product-Centric_Repositioning.docx"
    file_bytes = b"mock docx content"

    storage = AsyncMock(spec=BaseObjectStorage)
    storage.bucket_name = "test-documents"
    uploaded_objects: dict[str, bytes] = {}

    async def fake_upload(path, data, content_type=None, upsert=False):
        uploaded_objects[path] = data
        return StorageObjectMetadata(
            name=path.split("/")[-1],
            path=path,
            bucket="test-documents",
            size_bytes=len(data),
            content_type=content_type or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            etag="etag-special-123",
        )

    async def fake_get_metadata(path):
        if path not in uploaded_objects:
            from exceptions.storage import ObjectNotFoundError
            raise ObjectNotFoundError(f"Not found: {path}")
        return StorageObjectMetadata(
            name=path.split("/")[-1],
            path=path,
            bucket="test-documents",
            size_bytes=len(uploaded_objects[path]),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            etag="etag-special-123",
        )

    async def fake_download(path):
        if path not in uploaded_objects:
            from exceptions.storage import ObjectNotFoundError
            raise ObjectNotFoundError(f"Not found: {path}")
        return uploaded_objects[path]

    async def fake_delete_many(paths):
        deleted = []
        for p in paths:
            if p in uploaded_objects:
                del uploaded_objects[p]
                deleted.append(p)
        return deleted

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.get_metadata = AsyncMock(side_effect=fake_get_metadata)
    storage.download = AsyncMock(side_effect=fake_download)
    storage.delete_many = AsyncMock(side_effect=fake_delete_many)

    from rag.acquisition.service import StorageAcquisitionService
    from rag.ingestion.service import StorageIngestionService
    from rag.ingestion.models import DocumentSourceReference

    acquisition_svc = StorageAcquisitionService(storage=storage)
    ingestion_svc = StorageIngestionService(storage=storage)

    async with create_test_session() as session:
        proj = ProjectModel(name="Special Chars Proj")
        session.add(proj)
        await session.flush()

        service = DocumentService(session=session, storage=storage)

        # 1. Upload initial document with Unicode punctuation and special characters
        doc1 = await service.create_document(
            project_id=proj.id,
            filename=special_filename,
            file_data=file_bytes,
        )

        # User-facing metadata remains intact
        assert doc1.name == special_filename
        v1 = doc1.versions[0]
        assert v1.original_filename == special_filename
        expected_v1_path = f"{proj.id}/{doc1.id}/v1/{safe_storage_filename}"
        assert v1.storage_path == expected_v1_path
        assert expected_v1_path in uploaded_objects

        # 2. Acquisition stage verifies format and acquires source doc
        source_doc = await acquisition_svc.acquire_document(
            project_id=proj.id,
            storage_path=v1.storage_path,
        )
        assert source_doc.document_type.value == "docx"
        assert source_doc.storage_path == expected_v1_path

        # 3. Ingestion stage downloads bytes and preserves original filename
        doc_source_ref = DocumentSourceReference(
            project_id=proj.id,
            storage_path=source_doc.storage_path,
            storage_bucket="test-documents",
            original_filename=v1.original_filename,
            content_type=v1.content_type,
            document_type=source_doc.document_type,
            document_id=doc1.id,
            document_version_id=v1.id,
        )
        ingested = await ingestion_svc.ingest(source=doc_source_ref, project_id=proj.id)
        assert ingested.raw_bytes == file_bytes
        assert ingested.original_filename == special_filename
        assert ingested.source_storage_path == expected_v1_path

        # 4. Deletion cleans up the safe storage key
        await service.delete_document(project_id=proj.id, document_id=doc1.id)
        assert expected_v1_path not in uploaded_objects

        # 5. Re-upload works normally with fresh document ID and fresh version
        doc2 = await service.create_document(
            project_id=proj.id,
            filename=special_filename,
            file_data=b"re-uploaded fresh bytes",
        )
        assert doc2.id != doc1.id
        assert doc2.name == special_filename
        v2 = doc2.versions[0]
        assert v2.version_number == 1
        assert v2.original_filename == special_filename
        expected_v2_path = f"{proj.id}/{doc2.id}/v1/{safe_storage_filename}"
        assert v2.storage_path == expected_v2_path
        assert expected_v2_path in uploaded_objects

