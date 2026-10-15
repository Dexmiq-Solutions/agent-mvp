"""Targeted test suite for Database Session and Transaction Lifecycle during BRD Agent execution.

Validates the fix for the production connection degradation failure:
1. Test 1 — Pre-agent session is released: No active transaction or connection held when agent begins.
2. Test 2 — Final persistence uses a fresh session: Assistant message persisted with fresh session.
3. Test 3 — Stale original connection does not break final persistence: Invalidation of original
            session during agent execution does not prevent assistant turn persistence.
4. Test 4 — Streaming path: Full SSE lifecycle emits done event without error event.
5. Test 5 — Non-streaming path: Full HTTP 201 lifecycle for stream=false.
6. Test 6 — Failed transaction cleanup: Database failures do not trigger secondary PendingRollbackError.
"""

from collections.abc import AsyncIterator
import json
from typing import Any, Optional
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agents.brd.agent import BRDLeadAgent
from agents.brd.context import AgentContext, AgentRunResponse
from agents.brd.state import BRDAgentState
from api.dependencies import get_brd_lead_agent, get_conversation_service
from app.main import app
from db.base import Base
from db.session import get_async_session_maker, get_db_session
from services.conversation_service import ConversationService
from services.project_service import ProjectService


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for streaming and non-streaming tests."""

    token_chunks: list[str] = ["The ", "Business ", "Requirements ", "Document."]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "".join(self.token_chunks)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for token in self.token_chunks:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def parse_sse_events(lines: list[str]) -> list[dict[str, Any]]:
    """Parse raw SSE lines into structured event dictionaries."""
    events = []
    current_event = "message"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.replace("event:", "").strip()
        elif line.startswith("data:"):
            data_str = line.replace("data:", "", 1).strip()
            data_json = json.loads(data_str)
            events.append({"event": current_event, "data": data_json})
            current_event = "message"
    return events


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_env():
    """Set up in-memory SQLite engine, session maker, and AsyncClient with tracking."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    created_sessions: list[AsyncSession] = []

    async def tracking_get_db_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            created_sessions.append(session)
            try:
                yield session
                if session.is_active and session.in_transaction():
                    await session.commit()
            except Exception:
                if session.is_active:
                    await session.rollback()
                raise
            finally:
                await session.close()

    app.dependency_overrides[get_db_session] = tracking_get_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield {
            "client": client,
            "engine": engine,
            "session_factory": session_factory,
            "created_sessions": created_sessions,
        }

    app.dependency_overrides.clear()
    await engine.dispose()


# ==============================================================================
# Test 1 — Pre-agent session is released (no active transaction when agent runs)
# ==============================================================================

@pytest.mark.anyio
async def test_pre_agent_session_transaction_is_released(test_env):
    """Verify that before the agent begins, the pre-agent session has committed and holds no active transaction."""
    client = test_env["client"]
    created_sessions = test_env["created_sessions"]

    session_in_transaction_at_agent_start: list[bool] = []

    class InspectingAgent(BRDLeadAgent):
        async def stream_async(self, request, context=None, **kwargs):
            # Inspect the pre-agent session at the moment agent begins
            if created_sessions:
                pre_agent_session = created_sessions[0]
                session_in_transaction_at_agent_start.append(
                    pre_agent_session.in_transaction()
                )
            yield {"type": "content", "content": "Done."}

    app.dependency_overrides[get_brd_lead_agent] = lambda: InspectingAgent(model=MockChatModel())

    try:
        # Create project and conversation
        proj_resp = await client.post("/projects", json={"name": "Lifecycle Test Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await client.post(
            f"/projects/{project_id}/conversations", json={"title": "Session Lifecycle Chat"}
        )
        conversation_id = conv_resp.json()["id"]

        # Post user message to trigger streaming
        async with client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Generate the executive summary"},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():
                pass

        # Invariant check: when agent execution began, pre-agent session had NO active transaction
        assert len(session_in_transaction_at_agent_start) == 1
        assert session_in_transaction_at_agent_start[0] is False, (
            "Pre-agent session must NOT hold an active transaction while the agent executes"
        )

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


# ==============================================================================
# Test 2 — Final persistence uses a fresh session
# ==============================================================================

@pytest.mark.anyio
async def test_final_persistence_uses_fresh_session(test_env):
    """Verify that final assistant message persistence uses a newly created AsyncSession instance."""
    client = test_env["client"]

    sessions_used_by_service: list[AsyncSession] = []
    original_init = ConversationService.__init__

    def tracked_conv_init(self, session, *args, **kwargs):
        sessions_used_by_service.append(session)
        original_init(self, session, *args, **kwargs)

    ConversationService.__init__ = tracked_conv_init

    test_agent = BRDLeadAgent(model=MockChatModel())

    async def mock_stream(*args, **kwargs):
        yield {"type": "content", "content": "BRD drafted successfully."}

    test_agent.stream_async = mock_stream
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_resp = await client.post("/projects", json={"name": "Fresh Session Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await client.post(f"/projects/{project_id}/conversations")
        conversation_id = conv_resp.json()["id"]

        async with client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Run fresh session test"},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():
                pass

        # At least 2 sessions should have been created:
        # Session 1: Request dependency session (for user message + history)
        # Session 2+: Fresh session created by _persist_assistant_turn
        assert len(sessions_used_by_service) >= 2
        request_session = sessions_used_by_service[0]
        persistence_session = sessions_used_by_service[-1]
        assert persistence_session is not request_session, (
            "Final assistant persistence must use a fresh session, not the request-scoped session"
        )

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)
        ConversationService.__init__ = original_init


# ==============================================================================
# Test 3 — Stale original connection does not break final persistence
# ==============================================================================

@pytest.mark.anyio
async def test_stale_original_session_does_not_break_persistence(test_env):
    """Reproduce production failure: closing/invalidating original request session does not prevent final persistence."""
    client = test_env["client"]
    created_sessions = test_env["created_sessions"]

    class ConnectionDroppingAgent(BRDLeadAgent):
        async def stream_async(self, request, context=None, **kwargs):
            yield {"type": "progress", "phase": "1_INITIAL_CONTEXT", "message": "Simulating long run..."}
            # Directly close the original request session during agent execution (simulating dropped connection)
            if created_sessions:
                original_sess = created_sessions[0]
                await original_sess.close()

            yield {"type": "content", "content": "Content generated after connection drop."}

    app.dependency_overrides[get_brd_lead_agent] = lambda: ConnectionDroppingAgent(model=MockChatModel())

    try:
        proj_resp = await client.post("/projects", json={"name": "Connection Drop Test"})
        project_id = proj_resp.json()["id"]

        conv_resp = await client.post(f"/projects/{project_id}/conversations")
        conversation_id = conv_resp.json()["id"]

        lines = []
        async with client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Trigger dropped connection test"},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)

        events = parse_sse_events(lines)

        # Assert no error event was emitted
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0, f"Expected no error events, got: {error_events}"

        # Assert done event was emitted
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["message"]["role"] == "assistant"
        assert "Content generated after connection drop." in done_events[0]["data"]["message"]["content"]

        # Verify database has both messages persisted
        hist_resp = await client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert hist_resp.status_code == 200
        messages = hist_resp.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


# ==============================================================================
# Test 4 — Streaming path emits done event with persisted state and metadata
# ==============================================================================

@pytest.mark.anyio
async def test_streaming_path_persists_workflow_state_and_emits_done(test_env):
    """Verify streaming path persists assistant message and emits SSE done event."""
    client = test_env["client"]

    test_agent = BRDLeadAgent(model=MockChatModel())

    async def mock_stream(*args, **kwargs):
        yield {"type": "progress", "phase": "1_INITIAL_CONTEXT", "message": "Drafting..."}
        yield {"type": "content", "content": "Section 1: Executive Summary."}

    test_agent.stream_async = mock_stream
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_resp = await client.post("/projects", json={"name": "Streaming Path Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await client.post(f"/projects/{project_id}/conversations")
        conversation_id = conv_resp.json()["id"]

        lines = []
        async with client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Draft section 1"},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)

        events = parse_sse_events(lines)
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        done_msg = done_events[0]["data"]["message"]
        assert done_msg["role"] == "assistant"
        assert done_msg["content"] == "Section 1: Executive Summary."
        assert "user_message_id" in done_msg["metadata"]
        assert "duration_seconds" in done_msg["metadata"]

        # Ensure no error event
        assert not any(e.get("event") == "error" for e in events)

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


# ==============================================================================
# Test 5 — Non-streaming path persists with fresh session (stream=false)
# ==============================================================================

@pytest.mark.anyio
async def test_non_streaming_path_persists_with_fresh_session(test_env):
    """Verify non-streaming path (stream=false) persists assistant message via fresh session."""
    client = test_env["client"]

    test_agent = BRDLeadAgent(model=MockChatModel())

    async def mock_run_workflow(*args, **kwargs):
        return AgentRunResponse(
            output_text="Non-streaming completed response.",
            context=kwargs.get("context") or AgentContext(),
            success=True,
            state=None,
        )

    test_agent.run_workflow_async = mock_run_workflow
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        proj_resp = await client.post("/projects", json={"name": "Non-Streaming Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await client.post(f"/projects/{project_id}/conversations")
        conversation_id = conv_resp.json()["id"]

        resp = await client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Run non-streaming test", "stream": False},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["role"] == "assistant"
        assert data["content"] == "Non-streaming completed response."
        assert "user_message_id" in data["metadata"]

        # Verify both turns in database
        hist_resp = await client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert hist_resp.status_code == 200
        messages = hist_resp.json()
        assert len(messages) == 2

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


# ==============================================================================
# Test 6 — Failed transaction cleanup does not produce PendingRollbackError
# ==============================================================================

@pytest.mark.anyio
async def test_failed_transaction_cleanup_no_pending_rollback_error():
    """Verify get_db_session handles failures cleanly without producing PendingRollbackError."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    from unittest.mock import patch

    with patch("db.session.get_async_session_maker", return_value=session_factory):
        gen = get_db_session()
        session = await gen.__anext__()

        # Trigger a query that causes an intentional SQL error inside an active transaction
        with pytest.raises(Exception):
            await session.execute(text("SELECT * FROM non_existent_table_xyz"))

        # Close generator via athrow simulating route exception
        with pytest.raises(ValueError) as exc_info:
            await gen.athrow(ValueError("Simulated route exception after query failure"))

        # The propagated exception must be the original error, NOT PendingRollbackError
        assert "Simulated route exception after query failure" in str(exc_info.value)
        assert "PendingRollbackError" not in str(exc_info.value)

    await engine.dispose()
