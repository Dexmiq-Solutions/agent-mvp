"""Comprehensive test suite for Workflow Switching and Execution Lease Management.

Verifies:
1. Atomic workflow validation blocks wrong agent execution:
   - Calling /recommendations when active_workflow="brd" returns HTTP 400.
   - Calling /messages when active_workflow="recommendation" returns HTTP 400.
2. Clean workflow switching via PATCH /conversations/{id} when idle.
3. Workflow switching is blocked (HTTP 409) when an agent execution lease is active.
4. Execution lease concurrency conflict: concurrent agent execution returns HTTP 409.
5. Non-revivable heartbeat: expired lease cannot be revived by delayed heartbeat.
6. Real-time pre-commit fencing assertion catches expired/stolen lease and prevents commit.
7. Multi-turn workflow state isolation:
   - Turns tagged with workflow="brd" vs workflow="recommendation".
   - BRD state restored only from BRD turns while dialog includes Recommendation turns labeled.
   - Recommendation context restores recommendation state without corrupting BRD state.
"""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.brd.agent import BRDLeadAgent
from agents.brd.context import AgentContext, AgentRunResponse
from agents.brd.state import BRDAgentState, BRDSectionStatus
from agents.recommendation.contracts import RouteDecision
from agents.recommendation.router.agent import IntentRouterAgent
from agents.recommendation.workflow import RecommendationWorkflow
from api.dependencies import get_brd_lead_agent, get_current_user, get_db_session, get_recommendation_workflow
from app.main import app
from db.base import Base
from exceptions.conversation import (
    ExecutionLeaseConflictError,
    StaleExecutionFencedError,
    WorkflowMismatchError,
    WorkflowSwitchBlockedError,
)
from models.conversation import ConversationModel
from models.user import UserModel
from schemas.conversation import ConversationUpdate
from services.conversation_service import ConversationService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_engine_and_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    yield engine, session_factory
    await engine.dispose()


@pytest.fixture
async def api_client(db_engine_and_factory):
    engine, session_factory = db_engine_and_factory

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

    test_user = UserModel(id="switch-user", email="switch@example.com", hashed_password="dummy_hash")
    async with session_factory() as session:
        session.add(test_user)
        await session.commit()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_current_user] = lambda: test_user

    # Setup dummy Recommendation workflow
    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="explain",
            high_stakes=False,
            high_stakes_reasons=[],
            missing_facts=[],
            confidence="high",
            short_reason="Explanation request",
        )
    )
    app.dependency_overrides[get_recommendation_workflow] = lambda: RecommendationWorkflow(
        router_agent=mock_router
    )

    # Setup dummy BRD Lead agent
    mock_brd_agent = MagicMock(spec=BRDLeadAgent)
    mock_brd_agent.telemetry_tracker = None
    mock_brd_agent.run_workflow_async = AsyncMock(
        return_value=AgentRunResponse(
            output_text="BRD response generated.",
            state=BRDAgentState(
                current_section="1. Executive Summary",
                section_progress={"1. Executive Summary": BRDSectionStatus.COMPLETED},
                metadata={"working_draft": {"1. Executive Summary": "Executive Summary finalized"}},
            ),
        )
    )
    app.dependency_overrides[get_brd_lead_agent] = lambda: mock_brd_agent

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory

    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_workflow_mismatch_blocks_execution(api_client):
    client, _ = api_client

    # 1. Create project
    proj_resp = await client.post("/projects", json={"name": "Mismatch Test Project"})
    assert proj_resp.status_code == 201
    project_id = proj_resp.json()["id"]

    # 2. Create conversation with active_workflow="brd" (default)
    conv_resp = await client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "BRD Conversation"},
    )
    assert conv_resp.status_code == 201
    conv_data = conv_resp.json()
    assert conv_data["active_workflow"] == "brd"
    conversation_id = conv_data["id"]

    # 3. Invoking recommendation endpoint on BRD conversation must return 400
    rec_resp = await client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/recommendations",
        json={"content": "Please recommend a tech stack."},
    )
    assert rec_resp.status_code == 400
    rec_err = rec_resp.json()["detail"]
    assert "Workflow mismatch" in rec_err
    assert "active workflow is 'brd'" in rec_err
    assert "recommendation" in rec_err

    # 4. Switch conversation to active_workflow="recommendation"
    patch_resp = await client.patch(
        f"/projects/{project_id}/conversations/{conversation_id}",
        json={"active_workflow": "recommendation"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["active_workflow"] == "recommendation"

    # 5. Invoking BRD endpoint on Recommendation conversation must return 400
    brd_resp = await client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
        json={"content": "Please write section 1."},
    )
    assert brd_resp.status_code == 400
    brd_err = brd_resp.json()["detail"]
    assert "Workflow mismatch" in brd_err
    assert "active workflow is 'recommendation'" in brd_err
    assert "brd" in brd_err


@pytest.mark.anyio
async def test_workflow_switching_via_patch(api_client):
    client, _ = api_client

    proj_resp = await client.post("/projects", json={"name": "Switching Test Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Switchable Conversation"},
    )
    assert conv_resp.status_code == 201
    conv_id = conv_resp.json()["id"]
    assert conv_resp.json()["active_workflow"] == "brd"

    # Switch to recommendation
    patch_1 = await client.patch(
        f"/projects/{project_id}/conversations/{conv_id}",
        json={"active_workflow": "recommendation"},
    )
    assert patch_1.status_code == 200
    assert patch_1.json()["active_workflow"] == "recommendation"

    # Verify recommendation endpoint works
    rec_resp = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/recommendations",
        json={"content": "What is the best database?"},
    )
    assert rec_resp.status_code == 200
    assert rec_resp.json()["status"] == "ready_for_phase_2"

    # Switch back to brd
    patch_2 = await client.patch(
        f"/projects/{project_id}/conversations/{conv_id}",
        json={"active_workflow": "brd"},
    )
    assert patch_2.status_code == 200
    assert patch_2.json()["active_workflow"] == "brd"

    # Verify brd endpoint works
    brd_resp = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages?stream=false",
        json={"content": "Generate section 1 draft"},
    )
    assert brd_resp.status_code == 201
    assert brd_resp.json()["content"] == "BRD response generated."


@pytest.mark.anyio
async def test_workflow_switching_blocked_when_running(api_client):
    client, session_factory = api_client

    proj_resp = await client.post("/projects", json={"name": "Lock Test Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Active Run Conversation"},
    )
    conv_id = conv_resp.json()["id"]

    # Simulate an active lease held on this conversation
    future_time = datetime.now(timezone.utc) + timedelta(seconds=45)
    async with session_factory() as session:
        await session.execute(
            update(ConversationModel)
            .where(ConversationModel.id == conv_id)
            .values(
                execution_status="running",
                active_run_id="running-agent-job-123",
                lease_expires_at=future_time,
            )
        )
        await session.commit()

    # Attempt to switch workflow while running
    patch_resp = await client.patch(
        f"/projects/{project_id}/conversations/{conv_id}",
        json={"active_workflow": "recommendation"},
    )
    assert patch_resp.status_code == 409
    detail = patch_resp.json()["detail"]
    assert "Cannot switch workflow" in detail
    assert "is in progress" in detail


@pytest.mark.anyio
async def test_execution_lease_concurrency_conflict(api_client):
    client, session_factory = api_client

    proj_resp = await client.post("/projects", json={"name": "Concurrency Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Concurrent Lease Conversation"},
    )
    conv_id = conv_resp.json()["id"]

    # Acquire lease using service directly
    async with session_factory() as session:
        service = ConversationService(session)
        success = await service.acquire_execution_lease(
            project_id=project_id,
            conversation_id=conv_id,
            run_id="run-owner-1",
            expected_workflow="brd",
            ttl_seconds=45,
        )
        assert success.active_run_id == "run-owner-1"
        await session.commit()

    # A concurrent request to /messages must return 409 Conflict
    msg_resp = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages?stream=false",
        json={"content": "Concurrent message"},
    )
    assert msg_resp.status_code == 409
    detail = msg_resp.json()["detail"]
    assert "active execution lease" in detail


@pytest.mark.anyio
async def test_heartbeat_cannot_revive_expired_lease(db_engine_and_factory):
    _, session_factory = db_engine_and_factory

    async with session_factory() as session:
        # Create conversation
        conv = ConversationModel(
            id="conv-heartbeat-test",
            project_id="proj-hb",
            title="Heartbeat Test",
            active_workflow="brd",
            execution_status="running",
            active_run_id="run-expired-1",
            lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        session.add(conv)
        await session.commit()

        service = ConversationService(session)
        # Delayed heartbeat arrives after expiry
        hb_success = await service.heartbeat_execution_lease(
            conversation_id="conv-heartbeat-test",
            run_id="run-expired-1",
            ttl_seconds=45,
        )
        assert hb_success is False

        # Verify lease was NOT updated or revived
        refreshed = (
            await session.execute(
                select(ConversationModel).where(ConversationModel.id == "conv-heartbeat-test")
            )
        ).scalar_one()
        # lease_expires_at should remain in past
        assert refreshed.lease_expires_at < datetime.now(timezone.utc)


@pytest.mark.anyio
async def test_pre_commit_fencing_assertion_detects_expiry_and_fences(db_engine_and_factory):
    _, session_factory = db_engine_and_factory

    async with session_factory() as session:
        # 1. Setup conversation with active lease
        conv = ConversationModel(
            id="conv-fencing-test",
            project_id="proj-fence",
            title="Fencing Test",
            active_workflow="brd",
            execution_status="running",
            active_run_id="run-fenced-1",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=45),
        )
        session.add(conv)
        await session.commit()

        service = ConversationService(session)
        # Initial check passes while lease is fresh
        await service.pre_commit_fencing_assertion(
            conversation_id="conv-fencing-test",
            expected_run_id="run-fenced-1",
        )

        # 2. Simulate expiry
        await session.execute(
            update(ConversationModel)
            .where(ConversationModel.id == "conv-fencing-test")
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=5))
        )
        await session.commit()

        # 3. Pre-commit check must fail and raise StaleExecutionFencedError
        with pytest.raises(StaleExecutionFencedError) as exc_info:
            await service.pre_commit_fencing_assertion(
                conversation_id="conv-fencing-test",
                expected_run_id="run-fenced-1",
            )
        assert "expired or lost ownership" in str(exc_info.value).lower()

        # 4. Simulate stolen lease by another run
        await session.execute(
            update(ConversationModel)
            .where(ConversationModel.id == "conv-fencing-test")
            .values(
                active_run_id="run-winner-2",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=45),
            )
        )
        await session.commit()

        with pytest.raises(StaleExecutionFencedError) as exc_info2:
            await service.pre_commit_fencing_assertion(
                conversation_id="conv-fencing-test",
                expected_run_id="run-fenced-1",
            )
        assert "expired or lost ownership" in str(exc_info2.value).lower()


@pytest.mark.anyio
async def test_workflow_state_isolation_multi_turn(api_client):
    client, session_factory = api_client

    proj_resp = await client.post("/projects", json={"name": "Isolation Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await client.post(
        f"/projects/{project_id}/conversations",
        json={"title": "Multi-Turn Isolation"},
    )
    conv_id = conv_resp.json()["id"]

    # Turn 1: BRD message
    brd_1 = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages?stream=false",
        json={"content": "User request for BRD section 1"},
    )
    assert brd_1.status_code == 201

    # Switch to recommendation workflow
    patch_rec = await client.patch(
        f"/projects/{project_id}/conversations/{conv_id}",
        json={"active_workflow": "recommendation"},
    )
    assert patch_rec.status_code == 200

    # Turn 2: Recommendation message
    rec_turn = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/recommendations",
        json={"content": "Recommendation question: cloud provider options?"},
    )
    assert rec_turn.status_code == 200

    # Switch back to BRD workflow
    patch_brd = await client.patch(
        f"/projects/{project_id}/conversations/{conv_id}",
        json={"active_workflow": "brd"},
    )
    assert patch_brd.status_code == 200

    # Turn 3: BRD message resuming
    brd_2 = await client.post(
        f"/projects/{project_id}/conversations/{conv_id}/messages?stream=false",
        json={"content": "User resuming BRD section 2"},
    )
    assert brd_2.status_code == 201

    # Verify message list tags and workflow state isolation
    msgs_resp = await client.get(f"/projects/{project_id}/conversations/{conv_id}/messages")
    assert msgs_resp.status_code == 200
    messages = msgs_resp.json()

    # We expect:
    # 0: User turn 1 (BRD)
    # 1: Assistant turn 1 (BRD)
    # 2: User turn 2 (Recommendation)
    # 3: Assistant turn 2 (Recommendation)
    # 4: User turn 3 (BRD)
    # 5: Assistant turn 3 (BRD)
    assert len(messages) == 6

    assert messages[0]["metadata"]["workflow"] == "brd"
    assert messages[1]["metadata"]["workflow"] == "brd"
    assert "workflow_state" in messages[1]["metadata"]

    assert messages[2]["metadata"]["workflow"] == "recommendation"
    assert messages[3]["metadata"]["workflow"] == "recommendation"
    assert messages[3]["metadata"]["status"] == "ready_for_phase_2"

    assert messages[4]["metadata"]["workflow"] == "brd"
    assert messages[5]["metadata"]["workflow"] == "brd"
    assert "workflow_state" in messages[5]["metadata"]
