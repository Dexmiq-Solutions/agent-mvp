from collections.abc import AsyncIterator
import json
from typing import Any, Optional
from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agents.brd import BRDLeadAgent
from api.dependencies import get_brd_lead_agent
from app.main import app
from db.base import Base
from db.session import get_db_session


class MockStreamingChatModel(BaseChatModel):
    """Deterministic mock chat model that streams tokens and records conversation turns."""

    token_chunks: list[str] = ["The ", "BRD ", "section ", "is drafted."]
    received_messages: list[list[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        content = "".join(self.token_chunks)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        for token in self.token_chunks:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-streaming-model"


def parse_sse_events(lines: list[str]) -> list[dict[str, Any]]:
    """Parse raw SSE line output into structured events."""
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

    # 2. Post a user message with stream=false for raw message CRUD test
    msg_user_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?stream=false",
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
        f"/projects/{proj_a_id}/conversations/{conv_a_id}/messages?stream=false",
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


@pytest.mark.anyio
async def test_user_message_streaming_invokes_agent_and_persists_both_messages(api_client: AsyncClient):
    """Verify that posting a user message invokes the BRD Agent workflow, streams SSE tokens, and persists both turns."""
    mock_model = MockStreamingChatModel(token_chunks=["The ", "BRD ", "Executive ", "Summary."])
    test_agent = BRDLeadAgent(model=mock_model)

    async def mock_stream_workflow(*args, **kwargs):
        yield {"type": "workflow_start", "agent_run_id": "test-run-123"}
        yield {"type": "progress", "phase": "1_INITIAL_CONTEXT", "message": "Starting BRD workflow..."}
        for chunk in ["The ", "BRD ", "Executive ", "Summary."]:
            yield {"type": "content", "content": chunk}

    test_agent.stream_workflow_async = mock_stream_workflow
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        # 1. Create project and conversation
        proj_resp = await api_client.post("/projects", json={"name": "Streaming Test Project"})
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        conv_resp = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "BRD Chat"})
        assert conv_resp.status_code == 201
        conversation_id = conv_resp.json()["id"]

        # 2. Post user message (default stream=True)
        user_prompt = "Generate the Executive Summary for our cloud platform."
        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": user_prompt, "metadata": {"source": "test-ui"}},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]

            lines = []
            async for line in stream_resp.aiter_lines():
                if line:
                    lines.append(line)

        events = parse_sse_events(lines)
        assert len(events) >= 2

        # Verify content chunk events
        content_events = [e for e in events if e.get("event") == "message" and e["data"].get("type") == "content"]
        assert len(content_events) == 4
        streamed_text = "".join(e["data"]["content"] for e in content_events)
        assert streamed_text == "The BRD Executive Summary."

        # Verify completion event
        done_events = [e for e in events if e.get("event") == "done" and e["data"].get("type") == "done"]
        assert len(done_events) == 1
        done_msg = done_events[0]["data"]["message"]
        assert done_msg["role"] == "assistant"
        assert done_msg["content"] == "The BRD Executive Summary."
        assert done_msg["conversation_id"] == conversation_id
        assert "user_message_id" in done_msg["metadata"]
        assert "agent_run_id" in done_msg["metadata"]

        # 3. Verify database persistence: both user message and assistant message must be present
        hist_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert hist_resp.status_code == 200
        messages = hist_resp.json()
        assert len(messages) == 2

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == user_prompt
        assert messages[0]["id"] == done_msg["metadata"]["user_message_id"]

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "The BRD Executive Summary."
        assert messages[1]["id"] == done_msg["id"]
        assert messages[1]["metadata"]["user_message_id"] == messages[0]["id"]

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_multi_turn_conversation_preserves_thread_context(api_client: AsyncClient):
    """Verify multi-turn conversation maintains history across turns through controlled workflow."""
    mock_model = MockStreamingChatModel(token_chunks=["Response ", "turn."])
    test_agent = BRDLeadAgent(model=mock_model)
    received_requests: list[str] = []

    async def tracking_stream_workflow(request=None, context=None, **kwargs):
        prompt = request.input_text if hasattr(request, "input_text") else str(request)
        received_requests.append(prompt)
        yield {"type": "progress", "phase": "1_INITIAL_CONTEXT", "message": f"Processing: {prompt}"}
        yield {"type": "content", "content": f"Answer to {prompt}"}

    test_agent.stream_workflow_async = tracking_stream_workflow
    app.dependency_overrides[get_brd_lead_agent] = lambda: test_agent

    try:
        # Create project and conversation
        proj_resp = await api_client.post("/projects", json={"name": "Multi-Turn Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Multi-Turn Chat"})
        conversation_id = conv_resp.json()["id"]

        # Turn 1
        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Turn 1 question"},
        ) as resp1:
            assert resp1.status_code == 200
            async for _ in resp1.aiter_lines():
                pass

        # Turn 2
        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Turn 2 question continuing previous context"},
        ) as resp2:
            assert resp2.status_code == 200
            async for _ in resp2.aiter_lines():
                pass

        # Verify that both turns entered the controlled workflow
        assert len(received_requests) == 2
        assert received_requests[0] == "Turn 1 question"
        assert received_requests[1] == "Turn 2 question continuing previous context"

        # Verify all 4 messages persisted in database in chronological order
        hist_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert hist_resp.status_code == 200
        messages = hist_resp.json()
        assert len(messages) == 4
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[0]["content"] == "Turn 1 question"
        assert messages[2]["content"] == "Turn 2 question continuing previous context"

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_streaming_error_handling_emits_error_event_and_does_not_persist_assistant_message(api_client: AsyncClient):
    """Verify that agent execution failures produce SSE error events without saving assistant responses."""

    class FailingChatModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            raise RuntimeError("Provider connection reset by peer")

        async def _agenerate(self, messages, **kwargs):
            raise RuntimeError("Provider connection reset by peer")

        async def _astream(self, messages, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Starting..."))
            raise RuntimeError("Provider connection reset by peer")

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self) -> str:
            return "failing-model"

    failing_agent = BRDLeadAgent(model=FailingChatModel())
    app.dependency_overrides[get_brd_lead_agent] = lambda: failing_agent

    try:
        # Create project and conversation
        proj_resp = await api_client.post("/projects", json={"name": "Error Handling Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Error Chat"})
        conversation_id = conv_resp.json()["id"]

        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "This prompt will trigger an agent error."},
        ) as resp:
            assert resp.status_code == 200
            lines = []
            async for line in resp.aiter_lines():
                if line:
                    lines.append(line)

        events = parse_sse_events(lines)
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 1
        assert "Provider connection reset by peer" in error_events[0]["data"]["error"]

        # Ensure NO completion event was emitted
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 0

        # Verify database: only the user message was persisted, NOT a broken assistant message
        hist_resp = await api_client.get(f"/projects/{project_id}/conversations/{conversation_id}/messages")
        assert hist_resp.status_code == 200
        messages = hist_resp.json()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


@pytest.mark.anyio
async def test_agent_receives_correct_project_context_and_stream_flag(api_client: AsyncClient):
    """Verify that BRDLeadAgent receives application-controlled project_id and stream=false bypasses agent."""
    captured_contexts: list[Any] = []

    class ContextCapturingAgent(BRDLeadAgent):
        async def stream_async(self, request, context=None, prior_messages=None):
            captured_contexts.append(context)
            yield {"type": "content", "content": "Context verified."}

    mock_agent = ContextCapturingAgent(model=MockStreamingChatModel())
    app.dependency_overrides[get_brd_lead_agent] = lambda: mock_agent

    try:
        proj_resp = await api_client.post("/projects", json={"name": "Context Isolation Project"})
        project_id = proj_resp.json()["id"]

        conv_resp = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Context Chat"})
        conversation_id = conv_resp.json()["id"]

        # 1. Test streaming call: verify captured context has application-controlled IDs
        async with api_client.stream(
            "POST",
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Check context"},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():
                pass

        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.project_id == project_id
        assert ctx.conversation_id == conversation_id
        assert "user_message_id" in ctx.metadata
        assert "agent_run_id" in ctx.metadata

        # 2. Test stream=false in payload: should persist immediately with 201 without invoking stream_async
        initial_capture_count = len(captured_contexts)
        direct_resp = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "Direct persist message", "stream": False},
        )
        assert direct_resp.status_code == 201
        assert direct_resp.json()["content"] == "Direct persist message"
        # stream_async should NOT have been called
        assert len(captured_contexts) == initial_capture_count

        # 3. Test assistant role message: should persist immediately with 201 without invoking stream_async
        asst_direct_resp = await api_client.post(
            f"/projects/{project_id}/conversations/{conversation_id}/messages",
            json={"role": "assistant", "content": "Manual assistant log"},
        )
        assert asst_direct_resp.status_code == 201
        assert asst_direct_resp.json()["content"] == "Manual assistant log"
        assert len(captured_contexts) == initial_capture_count

    finally:
        app.dependency_overrides.pop(get_brd_lead_agent, None)


