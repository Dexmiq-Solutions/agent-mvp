"""Focused test suite for Non-Streaming BRD Workflow Execution (DEF-007).

Validates:
1. Non-streaming user message requests (stream=false) invoke the full BRD workflow
   via `BRDLeadAgent.run_workflow_async` rather than bypassing the agent or returning
   the raw user message echo.
2. The endpoint returns HTTP 201 Created with a normalized `MessageResponse`
   containing role="assistant", non-empty content, and durable `workflow_state` metadata.
3. Database persistence records both the user message turn and the assistant message turn.
4. Payload `stream: false` takes precedence over default query parameters.
5. Context grounding provides project name, project description, and document inventory
   to `run_workflow_async`.
6. Durable state is preserved and restored across multi-turn non-streaming interactions.
7. Cross-mode state interoperability functions seamlessly:
   non-streaming (turn 1) -> streaming (turn 2) -> non-streaming (turn 3).
8. Clarification pauses (ASK_USER) return HTTP 201 with clarification content and
   waiting_for_user=True in metadata, and subsequent turns resume with cleared pause.
9. Rework exhaustion returns HTTP 201 with diagnostic summary text and updated section progress.
10. Unhandled agent execution exceptions return HTTP 500 without persisting corrupt
    assistant records.
11. Non-user messages (e.g. role="assistant") bypass agent execution for seeding.
"""

from collections.abc import AsyncIterator
import json
from typing import Any, Optional
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agents.brd.agent import BRDLeadAgent
from agents.brd.context import AgentContext, AgentRunResponse
from agents.brd.state import BRDAgentState, BRDSectionStatus
from api.dependencies import get_brd_lead_agent
from app.main import app
from db.base import Base
from db.session import get_db_session


class DummyChatModel(BaseChatModel):
    """Deterministic dummy chat model for unit testing."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Dummy response"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "dummy-chat-model"


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


@pytest.mark.anyio
async def test_non_streaming_message_invokes_run_workflow_async_and_returns_assistant_message(api_client: AsyncClient):
    """Verify stream=false invokes run_workflow_async, persists both turns, and returns HTTP 201 MessageResponse."""
    calls: list[dict[str, Any]] = []

    class MockBRDAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, **kwargs):
            calls.append({"request": request, "context": context, "kwargs": kwargs})
            self._state.section_progress["1.0 Executive Summary"] = BRDSectionStatus.COMPLETED
            return AgentRunResponse(
                output_text="## Executive Summary\nThis project establishes a core banking gateway.",
                context=context,
                success=True,
                state=self._state,
            )

    test_agent = MockBRDAgent(model=DummyChatModel())
    test_agent._state = BRDAgentState.initialize_from_template(["1.0 Executive Summary", "2.0 Scope"])
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "Banking Project", "description": "Core Gateway"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "BRD Generation"})
        conversation_id = conv_res.json()["id"]

        # Send non-streaming message
        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Draft the executive summary"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "assistant"
        assert "## Executive Summary" in data["content"]
        assert "workflow_state" in data["metadata"]
        assert data["metadata"]["workflow_state"]["section_progress"]["1.0 Executive Summary"] == BRDSectionStatus.COMPLETED.value
        assert data["metadata"]["project_id"] == project_id
        assert data["metadata"]["conversation_id"] == conversation_id
        assert data["metadata"]["agent_run_id"] is not None

        # Verify run_workflow_async was invoked
        assert len(calls) == 1
        assert calls[0]["request"] == "Draft the executive summary"
        assert calls[0]["context"].project_id == project_id

        # Verify database message history contains both user and assistant messages
        history_res = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert history_res.status_code == 200
        messages = history_res.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Draft the executive summary"
        assert messages[1]["role"] == "assistant"
        assert "## Executive Summary" in messages[1]["content"]
        assert messages[1]["metadata"]["workflow_state"] is not None

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_payload_stream_false_precedence(api_client: AsyncClient):
    """Verify that stream: false in payload takes precedence over query parameter defaults."""
    invoked = False

    class MockBRDAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, **kwargs):
            nonlocal invoked
            invoked = True
            return AgentRunResponse(
                output_text="Non-streaming response via payload flag",
                context=context,
                success=True,
                state=self._state,
            )

    test_agent = MockBRDAgent(model=DummyChatModel())
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "Precedence Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Precedence Chat"})
        conversation_id = conv_res.json()["id"]

        # Note: No query parameter ?stream=false is provided (default is True in router),
        # but payload contains {"stream": false}
        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Run non-streaming", "stream": False},
        )

        assert response.status_code == 201
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert data["role"] == "assistant"
        assert data["content"] == "Non-streaming response via payload flag"
        assert invoked is True

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_context_grounding(api_client: AsyncClient):
    """Verify run_workflow_async receives project context with name, description, and document list."""
    captured_context: Optional[AgentContext] = None

    class ContextAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, **kwargs):
            nonlocal captured_context
            captured_context = context
            return AgentRunResponse(output_text="Context verified", context=context, success=True, state=self._state)

    app.dependency_overrides[get_brd_lead_agent] = lambda: ContextAgent(model=DummyChatModel())

    try:
        proj_res = await api_client.post(
            "/projects",
            json={
                "name": "Fintech Platform",
                "description": "High-throughput settlement infrastructure",
            },
        )
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Context Verification"})
        conversation_id = conv_res.json()["id"]

        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Verify my context grounding"},
        )
        assert response.status_code == 201

        assert captured_context is not None
        assert captured_context.project_id == project_id
        assert captured_context.conversation_id == conversation_id
        assert captured_context.project_name == "Fintech Platform"
        assert captured_context.project_description == "High-throughput settlement infrastructure"
        assert isinstance(captured_context.available_documents, list)
        assert "user_message_id" in captured_context.metadata
        assert "agent_run_id" in captured_context.metadata

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_durable_state_restoration(api_client: AsyncClient):
    """Verify durable state restoration across multi-turn non-streaming requests."""
    restored_states: list[Optional[BRDAgentState]] = []

    class StatePreservingAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, initial_state=None, **kwargs):
            restored_states.append(initial_state)

            current = initial_state or self._state
            # Modify state depending on turn
            if initial_state is None:
                current.section_progress["1.0 Executive Summary"] = BRDSectionStatus.COMPLETED
                current.evidence.append("Turn 1 Evidence")
                text = "Turn 1 complete."
            else:
                current.section_progress["2.0 Business Scope"] = BRDSectionStatus.COMPLETED
                current.evidence.append("Turn 2 Evidence")
                text = "Turn 2 complete."

            return AgentRunResponse(
                output_text=text,
                context=context,
                success=True,
                state=current,
            )

    test_agent = StatePreservingAgent(model=DummyChatModel())
    test_agent._state = BRDAgentState.initialize_from_template(["1.0 Executive Summary", "2.0 Business Scope"])
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "State Restoration Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Multi-Turn Chat"})
        conversation_id = conv_res.json()["id"]

        # Turn 1: Fresh execution without initial state
        resp1 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Execute Turn 1"},
        )
        assert resp1.status_code == 201
        assert resp1.json()["content"] == "Turn 1 complete."
        assert len(restored_states) == 1
        assert restored_states[0] is None

        # Turn 2: Follow-up turn, should rehydrate Turn 1 state
        resp2 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Execute Turn 2"},
        )
        assert resp2.status_code == 201
        assert resp2.json()["content"] == "Turn 2 complete."
        assert len(restored_states) == 2

        # Verify that Turn 2 received rehydrated state from Turn 1
        turn2_initial = restored_states[1]
        assert turn2_initial is not None
        assert turn2_initial.section_progress["1.0 Executive Summary"] == BRDSectionStatus.COMPLETED
        assert "Turn 1 Evidence" in turn2_initial.evidence

        # Check final conversation metadata
        final_meta = resp2.json()["metadata"]["workflow_state"]
        assert final_meta["section_progress"]["1.0 Executive Summary"] == BRDSectionStatus.COMPLETED.value
        assert final_meta["section_progress"]["2.0 Business Scope"] == BRDSectionStatus.COMPLETED.value
        assert "Turn 1 Evidence" in final_meta["evidence"]
        assert "Turn 2 Evidence" in final_meta["evidence"]

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_cross_mode_state_interoperability(api_client: AsyncClient):
    """Verify seamless state continuity across non-streaming and streaming turns.

    Flow: Turn 1 (non-streaming) -> Turn 2 (streaming) -> Turn 3 (non-streaming).
    """
    rehydrated_states: list[Optional[BRDAgentState]] = []

    class HybridAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, initial_state=None, **kwargs):
            rehydrated_states.append(initial_state)
            current = initial_state or self._state
            current.evidence.append(f"Non-stream: {request}")
            return AgentRunResponse(
                output_text=f"Processed non-streaming: {request}",
                context=context,
                success=True,
                state=current,
            )

        async def stream_async(self, request=None, context=None, prior_messages=None, **kwargs):
            # stream_async delegates to stream_workflow_async or yields events
            # For testing cross-mode state, retrieve the latest reconstructed state
            current = self._state
            current.evidence.append(f"Stream: {request}")
            yield {"type": "content", "content": "Streamed token "}
            yield {
                "type": "event",
                "event": "state_update",
                "data": {"state": current.to_dict()},
            }

    test_agent = HybridAgent(model=DummyChatModel())
    test_agent._state = BRDAgentState.initialize_from_template(["1.0 Scope", "2.0 Architecture"])
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "Interoperability Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Hybrid Modes"})
        conversation_id = conv_res.json()["id"]

        # Turn 1: Non-streaming
        resp1 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Turn 1 NonStream"},
        )
        assert resp1.status_code == 201
        assert "Non-stream: Turn 1 NonStream" in resp1.json()["metadata"]["workflow_state"]["evidence"]

        # Turn 2: Streaming
        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=true",
            json={"role": "user", "content": "Turn 2 Stream"},
        ) as s_resp:
            assert s_resp.status_code == 200
            async for _ in s_resp.aiter_lines():
                pass

        # Turn 3: Non-streaming
        resp3 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Turn 3 NonStream"},
        )
        assert resp3.status_code == 201

        # Turn 3 must have received state from Turn 2 which preserved state from Turn 1
        turn3_received_state = rehydrated_states[-1]
        assert turn3_received_state is not None
        assert "Non-stream: Turn 1 NonStream" in turn3_received_state.evidence
        assert "Stream: Turn 2 Stream" in turn3_received_state.evidence

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_clarification_pause_and_resume(api_client: AsyncClient):
    """Verify ASK_USER clarification pause returns question and is_waiting_for_user=True, resuming on next turn."""
    clarification_q = "Which cloud provider (AWS, Azure, GCP) will host the infrastructure?"

    class ClarificationAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, initial_state=None, **kwargs):
            current = initial_state or self._state

            if not current.is_waiting_for_user:
                # Pause for clarification
                current.set_waiting_for_user(clarification_q)
                return AgentRunResponse(
                    output_text=clarification_q,
                    context=context,
                    success=True,
                    state=current,
                )
            else:
                # Resume: user answered the question
                current.clear_waiting_for_user()
                current.evidence.append(f"Cloud provider clarified: {request}")
                return AgentRunResponse(
                    output_text="Thank you. Resumed BRD generation with specified cloud provider.",
                    context=context,
                    success=True,
                    state=current,
                )

    test_agent = ClarificationAgent(model=DummyChatModel())
    test_agent._state = BRDAgentState.initialize_from_template(["1.0 Overview"])
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "Clarification Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Clarification Chat"})
        conversation_id = conv_res.json()["id"]

        # Turn 1: Trigger clarification pause
        resp1 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Generate architecture BRD"},
        )
        assert resp1.status_code == 201
        data1 = resp1.json()
        assert data1["role"] == "assistant"
        assert data1["content"] == clarification_q
        assert data1["metadata"]["workflow_state"]["waiting_for_user"] is True
        assert data1["metadata"]["workflow_state"]["pending_clarification"] == clarification_q

        # Turn 2: User provides clarification
        resp2 = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "AWS with ECS and RDS"},
        )
        assert resp2.status_code == 201
        data2 = resp2.json()
        assert data2["role"] == "assistant"
        assert "Resumed BRD generation" in data2["content"]
        assert data2["metadata"]["workflow_state"]["waiting_for_user"] is False
        assert data2["metadata"]["workflow_state"]["pending_clarification"] is None
        assert "Cloud provider clarified: AWS with ECS and RDS" in data2["metadata"]["workflow_state"]["evidence"]

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_failure_diagnostics_on_rework_exhaustion(api_client: AsyncClient):
    """Verify rework budget exhaustion returns diagnostic text and updates section status cleanly."""
    class ExhaustionAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, initial_state=None, **kwargs):
            current = initial_state or self._state
            current.section_progress["1.0 Executive Summary"] = BRDSectionStatus.NEEDS_REVISION
            diagnostic_msg = (
                "Section '1.0 Executive Summary' failed validation after 3 rework attempts. "
                "Missing authoritative source citations for revenue projections."
            )
            current.metadata["failure_diagnostics"] = diagnostic_msg

            return AgentRunResponse(
                output_text=diagnostic_msg,
                context=context,
                success=False,
                state=current,
            )

    test_agent = ExhaustionAgent(model=DummyChatModel())
    test_agent._state = BRDAgentState.initialize_from_template(["1.0 Executive Summary"])
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_res = await api_client.post("/projects", json={"name": "Diagnostic Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Diagnostic Chat"})
        conversation_id = conv_res.json()["id"]

        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Attempt section rework"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "assistant"
        assert "failed validation after 3 rework attempts" in data["content"]
        assert data["metadata"]["workflow_state"]["section_progress"]["1.0 Executive Summary"] == BRDSectionStatus.NEEDS_REVISION.value
        assert "revenue projections" in data["metadata"]["workflow_state"]["metadata"]["failure_diagnostics"]

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_streaming_unhandled_exception_returns_500_and_does_not_persist_assistant_message(
    api_client: AsyncClient,
):
    """Verify that unhandled exceptions return 500 without persisting corrupt assistant records."""
    class CrashingAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, **kwargs):
            raise RuntimeError("Fatal unhandled LLM service crash")

    app.dependency_overrides[get_brd_lead_agent] = lambda: CrashingAgent(model=DummyChatModel())

    try:
        proj_res = await api_client.post("/projects", json={"name": "Crash Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Crash Chat"})
        conversation_id = conv_res.json()["id"]

        # Sending non-streaming message should raise 500
        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "user", "content": "Crash this execution"},
        )
        assert response.status_code == 500

        # Verify only the user message was persisted; NO assistant message exists
        history_res = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert history_res.status_code == 200
        messages = history_res.json()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Crash this execution"

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_non_user_message_bypasses_agent(api_client: AsyncClient):
    """Verify non-user messages (e.g. role="assistant") bypass agent execution and persist directly."""
    agent_called = False

    class SpyAgent(BRDLeadAgent):
        async def run_workflow_async(self, request=None, context=None, **kwargs):
            nonlocal agent_called
            agent_called = True
            return AgentRunResponse(output_text="Spy", context=context, success=True, state=self._state)

    app.dependency_overrides[get_brd_lead_agent] = lambda: SpyAgent(model=DummyChatModel())

    try:
        proj_res = await api_client.post("/projects", json={"name": "Seeding Project"})
        project_id = proj_res.json()["id"]

        conv_res = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Seeding Chat"})
        conversation_id = conv_res.json()["id"]

        response = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
            json={"role": "assistant", "content": "Seeded assistant summary"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "assistant"
        assert data["content"] == "Seeded assistant summary"
        assert agent_called is False

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)
