"""Integration tests verifying strict User -> Project ownership boundaries."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from db.base import Base
from db.session import get_db_session
from storage.object import get_object_storage
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_mock_storage() -> AsyncMock:
    """Create a mock BaseObjectStorage for test isolation."""
    storage = AsyncMock(spec=BaseObjectStorage)
    storage.upload = AsyncMock(
        return_value=StorageObjectMetadata(
            name="test.pdf",
            path="test/path.pdf",
            bucket="documents",
            size_bytes=100,
            content_type="application/pdf",
            etag="etag",
        )
    )
    storage.list_objects = AsyncMock(return_value=[])
    storage.delete = AsyncMock(return_value=True)
    storage.delete_many = AsyncMock(side_effect=lambda paths: paths)
    return storage


@pytest.fixture
async def client_setup():
    """Setup test database and client for testing two separate authenticated users."""
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

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_object_storage] = lambda: mock_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Register User A
        await client.post("/auth/signup", json={"email": "usera@example.com", "password": "Password123!"})
        login_a = await client.post("/auth/login", json={"email": "usera@example.com", "password": "Password123!"})
        token_a = login_a.json()["access_token"]

        # Register User B
        await client.post("/auth/signup", json={"email": "userb@example.com", "password": "Password123!"})
        login_b = await client.post("/auth/login", json={"email": "userb@example.com", "password": "Password123!"})
        token_b = login_b.json()["access_token"]

        yield client, token_a, token_b

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_unauthenticated_request_rejected(client_setup):
    """Verify unauthenticated requests to protected endpoints return 401."""
    client, _, _ = client_setup
    resp = await client.get("/projects")
    assert resp.status_code in (401, 403)


@pytest.mark.anyio
async def test_user_project_isolation(client_setup):
    """Verify User A and User B cannot view, read, update, or delete each other's projects."""
    client, token_a, token_b = client_setup
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 1. User A creates Project A
    resp_a = await client.post(
        "/projects",
        json={"name": "Project Alpha", "description": "User A project"},
        headers=headers_a,
    )
    assert resp_a.status_code == 201
    project_a_id = resp_a.json()["id"]

    # 2. User B creates Project B
    resp_b = await client.post(
        "/projects",
        json={"name": "Project Beta", "description": "User B project"},
        headers=headers_b,
    )
    assert resp_b.status_code == 201
    project_b_id = resp_b.json()["id"]

    # 3. User A listing projects only sees Project A
    list_a = await client.get("/projects", headers=headers_a)
    assert list_a.status_code == 200
    ids_a = [p["id"] for p in list_a.json()]
    assert project_a_id in ids_a
    assert project_b_id not in ids_a

    # 4. User B listing projects only sees Project B
    list_b = await client.get("/projects", headers=headers_b)
    assert list_b.status_code == 200
    ids_b = [p["id"] for p in list_b.json()]
    assert project_b_id in ids_b
    assert project_a_id not in ids_b

    # 5. User A attempting to get Project B returns 404
    get_cross = await client.get(f"/projects/{project_b_id}", headers=headers_a)
    assert get_cross.status_code == 404

    # 6. User A attempting to update Project B returns 404
    patch_cross = await client.patch(
        f"/projects/{project_b_id}",
        json={"name": "Hacked Name"},
        headers=headers_a,
    )
    assert patch_cross.status_code == 404

    # 7. User A attempting to delete Project B returns 404
    del_cross = await client.delete(f"/projects/{project_b_id}", headers=headers_a)
    assert del_cross.status_code == 404


@pytest.mark.anyio
async def test_sub_resource_project_ownership_isolation(client_setup):
    """Verify child resources (conversations, sources, retrieval) reject cross-tenant access with 404."""
    client, token_a, token_b = client_setup
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User B creates Project B
    resp_b = await client.post(
        "/projects",
        json={"name": "Project Confidential"},
        headers=headers_b,
    )
    project_b_id = resp_b.json()["id"]

    # 1. User A attempting to list conversations in Project B returns 404
    conv_list = await client.get(
        f"/projects/{project_b_id}/conversations",
        headers=headers_a,
    )
    assert conv_list.status_code == 404

    # 2. User A attempting to create conversation in Project B returns 404
    conv_create = await client.post(
        f"/projects/{project_b_id}/conversations",
        json={"title": "Intruder conversation"},
        headers=headers_a,
    )
    assert conv_create.status_code == 404

    # 3. User A attempting to list sources in Project B returns 404
    src_list = await client.get(
        f"/projects/{project_b_id}/sources",
        headers=headers_a,
    )
    assert src_list.status_code == 404

    # 4. User A attempting to query retrieval in Project B returns 404
    retrieval_query = await client.post(
        f"/projects/{project_b_id}/retrieval",
        json={"query": "Leaked secret knowledge"},
        headers=headers_a,
    )
    assert retrieval_query.status_code == 404
