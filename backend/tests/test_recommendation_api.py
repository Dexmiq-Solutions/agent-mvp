"""Integration tests for the Recommendation Workflow API endpoint.

Verifies:
- POST /projects/{project_id}/conversations/{conversation_id}/recommendations
  - Returns ready_for_phase_2 and persists assistant turn with routing metadata.
  - Returns needs_user and persists assistant turn with clarification metadata.
  - Fails closed with 404 on invalid/unauthorized project.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.recommendation.contracts import RouteDecision
from agents.recommendation.router.agent import IntentRouterAgent
from agents.recommendation.workflow import RecommendationWorkflow
from api.dependencies import get_current_user, get_db_session, get_recommendation_workflow
from app.main import app
from db.base import Base
from models.user import UserModel


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client():
    """Configure in-memory SQLite database, overrides, and AsyncClient for API testing."""
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

    test_user = UserModel(id="rec-test-user", email="rec@example.com", hashed_password="dummy_hash")
    async with session_factory() as session:
        session.add(test_user)
        await session.commit()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_current_user] = lambda: test_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_recommendation_api_ready_for_phase_2(api_client: AsyncClient):
    # 1. Create project
    proj_resp = await api_client.post("/projects", json={"name": "Recommendation Test Project"})
    assert proj_resp.status_code == 201
    project_id = proj_resp.json()["id"]

    # 2. Create conversation
    conv_resp = await api_client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Advisory Thread", "active_workflow": "recommendation"},
    )
    assert conv_resp.status_code == 201
    conversation_id = conv_resp.json()["id"]

    # 3. Configure mock workflow router
    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="explain",
            high_stakes=False,
            high_stakes_reasons=[],
            missing_facts=[],
            confidence="high",
            short_reason="User asks to explain section 6.",
        )
    )
    test_workflow = RecommendationWorkflow(router_agent=mock_router)
    app.dependency_overrides[get_recommendation_workflow] = lambda: test_workflow

    # 4. Invoke recommendation endpoint
    rec_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/recommendations",
        json={"content": "Explain section 6."},
    )
    assert rec_resp.status_code == 200
    data = rec_resp.json()
    assert data["status"] == "ready_for_phase_2"
    assert data["route_decision"]["profile"] == "explain"
    assert "Selected Profile" in data["response_markdown"]
    assert data["assistant_message_id"] is not None

    # 5. Verify conversation message list includes assistant response
    messages_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
    assert messages_resp.status_code == 200
    msgs = messages_resp.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["metadata"]["workflow"] == "recommendation"
    assert msgs[1]["metadata"]["status"] == "ready_for_phase_2"


@pytest.mark.anyio
async def test_recommendation_api_needs_user_clarification(api_client: AsyncClient):
    proj_resp = await api_client.post("/projects", json={"name": "Clarification Test Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Delay Thread", "active_workflow": "recommendation"},
    )
    conversation_id = conv_resp.json()["id"]

    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="communication",
            high_stakes=True,
            high_stakes_reasons=["client_facing"],
            missing_facts=["cause", "new_date"],
            confidence="high",
            short_reason="Missing critical timeline facts",
        )
    )
    test_workflow = RecommendationWorkflow(router_agent=mock_router)
    app.dependency_overrides[get_recommendation_workflow] = lambda: test_workflow

    rec_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/recommendations",
        json={"content": "We will be late with the release."},
    )
    assert rec_resp.status_code == 200
    data = rec_resp.json()
    assert data["status"] == "needs_user"
    assert data["clarification"] is not None
    assert len(data["clarification"]["questions"]) == 2
    assert "clarification on the following" in data["response_markdown"]

    messages_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
    msgs = messages_resp.json()
    assert msgs[1]["metadata"]["status"] == "needs_user"
    assert "pending_clarification" in msgs[1]["metadata"]


@pytest.mark.anyio
async def test_recommendation_api_invalid_project(api_client: AsyncClient):
    rec_resp = await api_client.post(
        "/projects/nonexistent-project-id/conversations/nonexistent-conv-id/recommendations",
        json={"content": "Hello"},
    )
    assert rec_resp.status_code == 404
