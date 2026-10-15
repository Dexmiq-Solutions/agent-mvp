"""End-to-end HTTP route tests for Project and Document/Source API endpoints."""

from collections.abc import AsyncIterator
import io
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from db.base import Base
from db.session import get_db_session
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata
from storage.object import get_object_storage


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_mock_storage() -> AsyncMock:
    """Create a mock BaseObjectStorage for route tests."""
    storage = AsyncMock(spec=BaseObjectStorage)

    async def fake_upload(path, data, content_type=None, upsert=False):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=len(data) if isinstance(data, bytes) else 1024,
            content_type=content_type or "application/pdf",
            etag="etag-api-test",
        )

    async def fake_get_metadata(path):
        filename = path.split("/")[-1]
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket="documents",
            size_bytes=1024,
            content_type="application/pdf",
            etag="etag-api-test",
        )

    storage.upload = AsyncMock(side_effect=fake_upload)
    storage.get_metadata = AsyncMock(side_effect=fake_get_metadata)
    storage.download = AsyncMock(return_value=b"%PDF-1.4 dummy pdf content for testing")
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    storage.list_objects = AsyncMock(return_value=[])
    return storage


@pytest.fixture
async def api_client(monkeypatch):
    """Configure test database, overrides, and an AsyncClient for FastAPI testing."""
    from core.config import get_settings
    monkeypatch.setattr(get_settings(), "AUTO_PROCESS_DOCUMENTS", False)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    mock_storage = make_mock_storage()

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

    from api.dependencies import get_current_user, get_storage
    from models.user import UserModel

    test_user = UserModel(id="test-user-id", email="test@example.com", hashed_password="dummy_hash")
    async with session_factory() as session:
        session.add(test_user)
        await session.commit()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[get_current_user] = lambda: test_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_storage

    app.dependency_overrides.clear()
    await engine.dispose()


# ==============================================================================
# 1. Project API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_create_and_get_project(api_client):
    """Verify POST /projects and GET /projects/{project_id}."""
    client, _ = api_client

    # 1. Create project
    create_resp = await client.post(
        "/projects",
        json={"name": "Project Apollo", "description": "Apollo research documents."},
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert data["name"] == "Project Apollo"
    assert data["description"] == "Apollo research documents."
    assert "id" in data
    project_id = data["id"]

    # 2. Get project by ID
    get_resp = await client.get(f"/projects/{project_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["id"] == project_id
    assert get_data["name"] == "Project Apollo"


@pytest.mark.anyio
async def test_api_create_project_invalid(api_client):
    """Verify POST /projects with invalid payload returns 422 or 400."""
    client, _ = api_client
    resp = await client.post("/projects", json={"name": "   "})
    assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_api_get_project_not_found(api_client):
    """Verify GET /projects/{project_id} returns 404 for non-existent project."""
    client, _ = api_client
    resp = await client.get("/projects/non-existent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_api_list_projects(api_client):
    """Verify GET /projects returns list of projects."""
    client, _ = api_client

    await client.post("/projects", json={"name": "Project One"})
    await client.post("/projects", json={"name": "Project Two"})

    resp = await client.get("/projects?limit=10")
    assert resp.status_code == 200
    projects = resp.json()
    assert len(projects) == 2


@pytest.mark.anyio
async def test_api_update_project(api_client):
    """Verify PATCH /projects/{project_id} updates project attributes."""
    client, _ = api_client

    create_resp = await client.post("/projects", json={"name": "Original Name"})
    project_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/projects/{project_id}",
        json={"name": "Updated Name", "description": "New description"},
    )
    assert patch_resp.status_code == 200
    patch_data = patch_resp.json()
    assert patch_data["name"] == "Updated Name"
    assert patch_data["description"] == "New description"


@pytest.mark.anyio
async def test_api_delete_project(api_client):
    """Verify DELETE /projects/{project_id} deletes project with 204."""
    client, _ = api_client

    create_resp = await client.post("/projects", json={"name": "Temporary"})
    project_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/projects/{project_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/projects/{project_id}")
    assert get_resp.status_code == 404


# ==============================================================================
# 2. Document / Source API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_upload_source_and_retrieve(api_client):
    """Verify POST /projects/{project_id}/sources and GET /projects/{project_id}/sources/{document_id}."""
    client, mock_storage = api_client

    # 1. Create project
    p_resp = await client.post("/projects", json={"name": "RAG Research"})
    project_id = p_resp.json()["id"]

    # 2. Upload file via multipart/form-data
    file_bytes = b"%PDF-1.4 test document content"
    files = {"file": ("manual.pdf", io.BytesIO(file_bytes), "application/pdf")}
    data = {"name": "System Manual"}

    upload_resp = await client.post(
        f"/projects/{project_id}/sources",
        files=files,
        data=data,
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    assert upload_data["name"] == "System Manual"
    assert upload_data["project_id"] == project_id
    assert upload_data["versions_count"] == 1
    assert upload_data["latest_version"]["version_number"] == 1
    assert upload_data["latest_version"]["status"] == "pending"
    doc_id = upload_data["id"]

    # 3. Retrieve document detail
    get_resp = await client.get(f"/projects/{project_id}/sources/{doc_id}")
    assert get_resp.status_code == 200
    doc_data = get_resp.json()
    assert doc_data["id"] == doc_id
    assert len(doc_data["versions"]) == 1
    assert doc_data["versions"][0]["original_filename"] == "manual.pdf"


@pytest.mark.anyio
async def test_api_upload_source_project_not_found(api_client):
    """Verify uploading to non-existent project returns 404."""
    client, _ = api_client
    files = {"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
    resp = await client.post("/projects/missing-project/sources", files=files)
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_api_list_sources(api_client):
    """Verify GET /projects/{project_id}/sources lists sources."""
    client, _ = api_client
    p_resp = await client.post("/projects", json={"name": "Source Listing"})
    project_id = p_resp.json()["id"]

    # Upload two files
    for name in ["doc1.pdf", "doc2.pdf"]:
        files = {"file": (name, io.BytesIO(b"content"), "application/pdf")}
        await client.post(f"/projects/{project_id}/sources", files=files)

    list_resp = await client.get(f"/projects/{project_id}/sources")
    assert list_resp.status_code == 200
    sources = list_resp.json()
    assert len(sources) == 2


@pytest.mark.anyio
async def test_api_update_source_metadata(api_client):
    """Verify PATCH /projects/{project_id}/sources/{document_id} updates source display name."""
    client, _ = api_client
    p_resp = await client.post("/projects", json={"name": "P1"})
    project_id = p_resp.json()["id"]

    files = {"file": ("initial.pdf", io.BytesIO(b"content"), "application/pdf")}
    up_resp = await client.post(f"/projects/{project_id}/sources", files=files)
    doc_id = up_resp.json()["id"]

    patch_resp = await client.patch(
        f"/projects/{project_id}/sources/{doc_id}",
        json={"name": "Renamed Source Document"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Renamed Source Document"


@pytest.mark.anyio
async def test_api_delete_source(api_client):
    """Verify DELETE /projects/{project_id}/sources/{document_id} removes source with 204."""
    client, mock_storage = api_client
    p_resp = await client.post("/projects", json={"name": "P1"})
    project_id = p_resp.json()["id"]

    files = {"file": ("to_delete.pdf", io.BytesIO(b"content"), "application/pdf")}
    up_resp = await client.post(f"/projects/{project_id}/sources", files=files)
    doc_id = up_resp.json()["id"]

    del_resp = await client.delete(f"/projects/{project_id}/sources/{doc_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/projects/{project_id}/sources/{doc_id}")
    assert get_resp.status_code == 404


@pytest.mark.anyio
async def test_api_create_source_version(api_client):
    """Verify POST /projects/{project_id}/sources/{document_id}/versions adds new physical version."""
    client, _ = api_client
    p_resp = await client.post("/projects", json={"name": "Versioning Proj"})
    project_id = p_resp.json()["id"]

    # Initial version v1
    files_v1 = {"file": ("release_v1.pdf", io.BytesIO(b"v1 content"), "application/pdf")}
    init_resp = await client.post(f"/projects/{project_id}/sources", files=files_v1)
    doc_id = init_resp.json()["id"]

    # Upload version v2
    files_v2 = {"file": ("release_v2.pdf", io.BytesIO(b"v2 content"), "application/pdf")}
    v2_resp = await client.post(
        f"/projects/{project_id}/sources/{doc_id}/versions",
        files=files_v2,
    )
    assert v2_resp.status_code == 201
    v2_data = v2_resp.json()
    assert v2_data["document_id"] == doc_id
    assert v2_data["version_number"] == 2
    assert v2_data["status"] == "pending"

    # Verify document detail reflects 2 versions
    detail_resp = await client.get(f"/projects/{project_id}/sources/{doc_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["versions_count"] == 2
    assert len(detail_data["versions"]) == 2
    assert [v["version_number"] for v in detail_data["versions"]] == [1, 2]


# ==============================================================================
# 3. Project Isolation via API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_project_isolation(api_client):
    """Verify cross-tenant document operations return 404 without leaking information."""
    client, _ = api_client

    # Create Project A and Project B
    resp_a = await client.post("/projects", json={"name": "Project A"})
    proj_a = resp_a.json()["id"]
    resp_b = await client.post("/projects", json={"name": "Project B"})
    proj_b = resp_b.json()["id"]

    # Upload Doc A and Doc B
    files_a = {"file": ("doc_a.pdf", io.BytesIO(b"a content"), "application/pdf")}
    doc_a = (await client.post(f"/projects/{proj_a}/sources", files=files_a)).json()["id"]

    files_b = {"file": ("doc_b.pdf", io.BytesIO(b"b content"), "application/pdf")}
    doc_b = (await client.post(f"/projects/{proj_b}/sources", files=files_b)).json()["id"]

    # 1. Project A listing does NOT contain doc_b
    list_a = (await client.get(f"/projects/{proj_a}/sources")).json()
    assert [d["id"] for d in list_a] == [doc_a]

    # 2. Accessing Doc B via Project A returns 404
    get_cross = await client.get(f"/projects/{proj_a}/sources/{doc_b}")
    assert get_cross.status_code == 404

    # 3. Patching Doc B via Project A returns 404
    patch_cross = await client.patch(
        f"/projects/{proj_a}/sources/{doc_b}", json={"name": "Hacked"}
    )
    assert patch_cross.status_code == 404

    # 4. Deleting Doc B via Project A returns 404
    del_cross = await client.delete(f"/projects/{proj_a}/sources/{doc_b}")
    assert del_cross.status_code == 404

    # 5. Adding version to Doc B via Project A returns 404
    files_hack = {"file": ("hack.pdf", io.BytesIO(b"hack"), "application/pdf")}
    ver_cross = await client.post(
        f"/projects/{proj_a}/sources/{doc_b}/versions", files=files_hack
    )
    assert ver_cross.status_code == 404

    # Verify Doc B is intact in Project B
    check_b = await client.get(f"/projects/{proj_b}/sources/{doc_b}")
    assert check_b.status_code == 200
    assert check_b.json()["id"] == doc_b
