"""End-to-end HTTP route tests for Project-scoped Conversation and Message API endpoints."""

from collections.abc import AsyncIterator
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from db.base import Base
from db.session import get_db_session


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client():
    """Configure in-memory SQLite database, overrides, and AsyncClient for FastAPI testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_conversation_crud_endpoints(api_client: AsyncClient):
    """Verify POST, GET, PATCH, and DELETE on /projects/{project_id}/conversations."""
    # 1. Create a project
    proj_resp = await api_client.post("/projects", json={"name": "Chat Project"})
    assert proj_resp.status_code == 201
    project_id = proj_resp.json()["id"]

    # 2. Create conversation
    conv_resp = await api_client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Q3 Planning Chat"},
    )
    assert conv_resp.status_code == 201
    conv_data = conv_resp.json()
    assert conv_data["title"] == "Q3 Planning Chat"
    assert conv_data["project_id"] == project_id
    assert conv_data["messages_count"] == 0
    conversation_id = conv_data["id"]

    # 3. List conversations
    list_resp = await api_client.get(f"/projects/{project_id}/conversations")
    assert list_resp.status_code == 200
    conversations = list_resp.json()
    assert len(conversations) == 1
    assert conversations[0]["id"] == conversation_id

    # 4. Get conversation details
    detail_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == conversation_id
    assert detail["messages"] == []

    # 5. Patch conversation title
    patch_resp = await api_client.patch(
        f"/projects/{project_id}/conversations/{conversation_id}",
        json={"title": "Updated Q3 Chat"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Updated Q3 Chat"

    # 6. Delete conversation
    del_resp = await api_client.delete(f"/projects/{project_id}/conversations/{conversation_id}")
    assert del_resp.status_code == 204

    # 7. Verify deletion
    get_del_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}")
    assert get_del_resp.status_code == 404


@pytest.mark.anyio
async def test_message_flow_endpoints(api_client: AsyncClient):
    """Verify POST and GET messages within a conversation."""
    # 1. Create project and conversation
    proj_resp = await api_client.post("/projects", json={"name": "RAG Flow Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Research Question"},
    )
    conversation_id = conv_resp.json()["id"]

    # 2. Post a user message
    msg_user_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages",
        json={
            "role": "user",
            "content": "What are the company's Q2 milestones?",
            "metadata": {"client": "web-ui"},
        },
    )
    assert msg_user_resp.status_code == 201
    user_msg = msg_user_resp.json()
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "What are the company's Q2 milestones?"
    assert user_msg["metadata"] == {"client": "web-ui"}
    assert user_msg["conversation_id"] == conversation_id

    # 3. Post an assistant message
    msg_asst_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages",
        json={
            "role": "assistant",
            "content": "The Q2 milestones include launching the MVP platform.",
            "metadata": {"tokens": 42},
        },
    )
    assert msg_asst_resp.status_code == 201
    asst_msg = msg_asst_resp.json()
    assert asst_msg["role"] == "assistant"

    # 4. Retrieve message history
    history_resp = await api_client.get(
        f"/projects/{project_id}/conversations/{conversation_id}/messages"
    )
    assert history_resp.status_code == 200
    messages = history_resp.json()
    assert len(messages) == 2
    assert messages[0]["id"] == user_msg["id"]
    assert messages[1]["id"] == asst_msg["id"]

    # 5. Retrieve conversation detail includes messages
    conv_detail_resp = await api_client.get(
        f"/projects/{project_id}/conversations/{conversation_id}"
    )
    assert conv_detail_resp.status_code == 200
    conv_detail = conv_detail_resp.json()
    assert conv_detail["messages_count"] == 2
    assert len(conv_detail["messages"]) == 2


@pytest.mark.anyio
async def test_api_project_isolation_enforcement(api_client: AsyncClient):
    """Verify that cross-project conversation and message access returns HTTP 404."""
    # Create Project A and Project B
    resp_a = await api_client.post("/projects", json={"name": "Tenant A"})
    proj_a_id = resp_a.json()["id"]

    resp_b = await api_client.post("/projects", json={"name": "Tenant B"})
    proj_b_id = resp_b.json()["id"]

    # Create conversation in Project A
    conv_a = await api_client.post(
        f"/projects/{proj_a_id}/conversations",
        json={"title": "Tenant A Confidential"},
    )
    conv_a_id = conv_a.json()["id"]

    # Create a message in Project A's conversation
    await api_client.post(
        f"/projects/{proj_a_id}/conversations/{conv_a_id}/messages",
        json={"role": "user", "content": "Confidential data for Tenant A"},
    )

    # 1. Tenant B tries to GET Tenant A's conversation
    cross_get = await api_client.get(f"/projects/{proj_b_id}/conversations/{conv_a_id}")
    assert cross_get.status_code == 404
    assert f"Conversation '{conv_a_id}' not found in project '{proj_b_id}'" in cross_get.json()["detail"]

    # 2. Tenant B tries to PATCH Tenant A's conversation
    cross_patch = await api_client.patch(
        f"/projects/{proj_b_id}/conversations/{conv_a_id}",
        json={"title": "Hacked Title"},
    )
    assert cross_patch.status_code == 404

    # 3. Tenant B tries to POST a message into Tenant A's conversation
    cross_msg_post = await api_client.post(
        f"/projects/{proj_b_id}/conversations/{conv_a_id}/messages",
        json={"role": "user", "content": "Unauthorized message"},
    )
    assert cross_msg_post.status_code == 404

    # 4. Tenant B tries to GET messages from Tenant A's conversation
    cross_msg_get = await api_client.get(
        f"/projects/{proj_b_id}/conversations/{conv_a_id}/messages"
    )
    assert cross_msg_get.status_code == 404

    # 5. Tenant B tries to DELETE Tenant A's conversation
    cross_delete = await api_client.delete(f"/projects/{proj_b_id}/conversations/{conv_a_id}")
    assert cross_delete.status_code == 404


@pytest.mark.anyio
async def test_message_validation_failure_handling(api_client: AsyncClient):
    """Verify 400 Bad Request responses on invalid message role or empty content."""
    proj_resp = await api_client.post("/projects", json={"name": "Input Test Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations")
    conv_id = conv_resp.json()["id"]

    # Invalid role
    role_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages",
        json={"role": "system_admin", "content": "Test content"},
    )
    assert role_resp.status_code == 422 or role_resp.status_code == 400

    # Empty content
    content_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages",
        json={"role": "user", "content": "   "},
    )
    assert content_resp.status_code == 422 or content_resp.status_code == 400
