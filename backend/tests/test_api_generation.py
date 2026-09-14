"""End-to-end API integration tests for project-scoped RAG + LLM generation."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.dependencies import get_generation_service, get_rag_service_dependency
from app.main import app
from db.base import Base
from db.session import get_db_session
from exceptions.generation import LLMRateLimitError, LLMTimeoutError
from rag.generation.evaluation.models import EvaluationResult
from rag.generation.evaluation.service import EvaluationService
from rag.generation.formatting.models import FormattedContext, FormattedContextItem
from llm.interface import BaseLLMInterface
from llm.models import LLMResult, LLMUsage
from llm.service import LLMService
from rag.retrieval.models import (
    AssembledContext,
    RetrievalExecutionMetadata,
    RetrievalResult,
    RetrievedChunk,
)
from rag.retrieval.service import RAGService
from services.conversation_service import ConversationService
from services.generation_service import GenerationService


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


def make_mock_retrieval_result(project_id: str, query: str) -> RetrievalResult:
    """Helper producing a valid RetrievalResult."""
    items = (
        FormattedContextItem(
            index=1,
            text=f"Project {project_id} document chunk information.",
            content=f"Project {project_id} document chunk information.",
            chunk_id="chunk-1",
            document_id="doc-1",
            source="guide.md",
        ),
    )
    formatted_ctx = FormattedContext(
        text=f"[Context 1]\nProject {project_id} document chunk information.",
        items=items,
        item_count=1,
        project_id=project_id,
    )
    retrieved_chunks = (
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            project_id=project_id,
            document_version_id="ver-1",
            content=f"Project {project_id} document chunk information.",
            rank=1,
            score=0.92,
            heading="Overview",
            section_path=("Overview",),
            contextual_content=None,
            metadata={"source": "guide.md"},
        ),
    )
    exec_meta = RetrievalExecutionMetadata(
        total_duration_ms=25.0,
        attempts_count=1,
        fallback_triggered=False,
        attempts=(),
        stage_latencies_ms={"dense_search_ms": 15.0},
    )
    return RetrievalResult(
        project_id=project_id,
        original_query=query,
        retrieval_query=query,
        chunks=retrieved_chunks,
        assembled_context=AssembledContext(
            items=(),
            project_id=project_id,
            query=query,
        ),
        formatted_context=formatted_ctx,
        execution_metadata=exec_meta,
    )


@pytest.mark.anyio
async def test_api_generation_end_to_end(api_client: AsyncClient):
    """Verify POST /projects/{project_id}/conversations/{conversation_id}/messages?generate=true.

    Produces both a persisted user message and an accepted assistant message turn.
    """
    # 1. Create a project and conversation
    proj_resp = await api_client.post("/projects", json={"name": "API Gen Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations", json={"title": "Q&A"})
    conversation_id = conv_resp.json()["id"]

    # 2. Setup Mock Services for Generation Dependency
    mock_rag = MagicMock(spec=RAGService)
    mock_rag.retrieve = AsyncMock(
        return_value=make_mock_retrieval_result(project_id, "What are the key features?")
    )

    mock_adapter = MagicMock(spec=BaseLLMInterface)
    mock_adapter.provider_name = "mock_provider"
    mock_adapter.model_name = "gpt-4o"
    mock_adapter.generate = AsyncMock(
        return_value=LLMResult(
            content="The key features include project isolation and RAG-driven synthesis.",
            finish_reason="stop",
            usage=LLMUsage(input_tokens=120, output_tokens=30, total_tokens=150),
        )
    )
    llm_service = LLMService(adapter=mock_adapter)

    mock_evaluator = MagicMock(spec=EvaluationService)
    mock_evaluator.config = MagicMock(max_regeneration_attempts=2)
    mock_evaluator.evaluate_async = AsyncMock(
        return_value=EvaluationResult(grounded=True, safe=True, reason="Grounded in context.")
    )

    def override_get_generation_service(session: AsyncSession = pytest.importorskip("fastapi").Depends(get_db_session)):
        conv_service = ConversationService(session=session)
        return GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

    app.dependency_overrides[get_generation_service] = override_get_generation_service

    # 3. Post a message with ?generate=true
    gen_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?generate=true",
        json={"role": "user", "content": "What are the key features?"},
    )

    assert gen_resp.status_code == 201
    asst_data = gen_resp.json()
    assert asst_data["role"] == "assistant"
    assert "project isolation and RAG-driven synthesis" in asst_data["content"]
    assert asst_data["conversation_id"] == conversation_id
    assert asst_data["metadata"]["evaluation_passed"] is True
    assert asst_data["metadata"]["finish_reason"] == "stop"

    # 4. Verify conversation message history contains both user and assistant turns
    history_resp = await api_client.get(
        f"/projects/{project_id}/conversations/{conversation_id}/messages"
    )
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What are the key features?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == asst_data["content"]


@pytest.mark.anyio
async def test_api_message_without_generate_flag(api_client: AsyncClient):
    """Verify POST /projects/{project_id}/conversations/{conversation_id}/messages without ?generate=true.

    Preserves standard message turn creation without invoking LLM generation.
    """
    proj_resp = await api_client.post("/projects", json={"name": "No Gen Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations")
    conversation_id = conv_resp.json()["id"]

    post_resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "Plain message without generation"},
    )
    assert post_resp.status_code == 201
    msg_data = post_resp.json()
    assert msg_data["role"] == "user"
    assert msg_data["content"] == "Plain message without generation"

    # Only 1 message exists in conversation
    history_resp = await api_client.get(
        f"/projects/{project_id}/conversations/{conversation_id}/messages"
    )
    assert len(history_resp.json()) == 1


@pytest.mark.anyio
async def test_api_generation_project_isolation_enforced(api_client: AsyncClient):
    """Verify that cross-project message generation returns HTTP 404."""
    resp_a = await api_client.post("/projects", json={"name": "Tenant A"})
    proj_a_id = resp_a.json()["id"]

    resp_b = await api_client.post("/projects", json={"name": "Tenant B"})
    proj_b_id = resp_b.json()["id"]

    conv_a = await api_client.post(f"/projects/{proj_a_id}/conversations")
    conv_a_id = conv_a.json()["id"]

    # Tenant B tries to generate message into Tenant A's conversation
    cross_resp = await api_client.post(
        f"/projects/{proj_b_id}/conversations/{conv_a_id}/messages?generate=true",
        json={"role": "user", "content": "Unauthorized generation attempt"},
    )
    assert cross_resp.status_code == 404
    assert f"Conversation '{conv_a_id}' not found in project '{proj_b_id}'" in cross_resp.json()["detail"]


@pytest.mark.anyio
async def test_api_generation_quality_gate_exhaustion(api_client: AsyncClient):
    """Verify HTTP 422 Unprocessable Entity when all regeneration attempts fail."""
    proj_resp = await api_client.post("/projects", json={"name": "Quality Gate Fail Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations")
    conversation_id = conv_resp.json()["id"]

    mock_rag = MagicMock(spec=RAGService)
    mock_rag.retrieve = AsyncMock(
        return_value=make_mock_retrieval_result(project_id, "Ungrounded query")
    )

    mock_adapter = MagicMock(spec=BaseLLMInterface)
    mock_adapter.provider_name = "mock_provider"
    mock_adapter.model_name = "gpt-4o"
    mock_adapter.generate = AsyncMock(
        return_value=LLMResult(content="Unsupported hallucination.", finish_reason="stop")
    )
    llm_service = LLMService(adapter=mock_adapter)

    mock_evaluator = MagicMock(spec=EvaluationService)
    mock_evaluator.config = MagicMock(max_regeneration_attempts=1)
    mock_evaluator.evaluate_async = AsyncMock(
        return_value=EvaluationResult(grounded=False, safe=True, reason="Claims not supported by context.")
    )

    def override_get_generation_service(session: AsyncSession = pytest.importorskip("fastapi").Depends(get_db_session)):
        conv_service = ConversationService(session=session)
        return GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

    app.dependency_overrides[get_generation_service] = override_get_generation_service

    resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?generate=true",
        json={"role": "user", "content": "Ungrounded query"},
    )
    assert resp.status_code == 422
    err_body = resp.json()
    assert "quality gate" in err_body["detail"]
    assert err_body["attempts"] == 2

    # User message was preserved in database
    history_resp = await api_client.get(
        f"/projects/{project_id}/conversations/{conversation_id}/messages"
    )
    history = history_resp.json()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Ungrounded query"


@pytest.mark.anyio
async def test_api_generation_llm_timeout_mapped_to_504(api_client: AsyncClient):
    """Verify HTTP 504 Gateway Timeout when LLM provider exceeds bounded timeout."""
    proj_resp = await api_client.post("/projects", json={"name": "Timeout Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations")
    conversation_id = conv_resp.json()["id"]

    mock_rag = MagicMock(spec=RAGService)
    mock_rag.retrieve = AsyncMock(
        return_value=make_mock_retrieval_result(project_id, "Query triggering timeout")
    )

    mock_adapter = MagicMock(spec=BaseLLMInterface)
    mock_adapter.provider_name = "mock_provider"
    mock_adapter.model_name = "gpt-4o"
    mock_adapter.generate = AsyncMock(
        side_effect=LLMTimeoutError("LLM upstream provider timed out after 30.0s")
    )
    llm_service = LLMService(adapter=mock_adapter)

    def override_get_generation_service(session: AsyncSession = pytest.importorskip("fastapi").Depends(get_db_session)):
        conv_service = ConversationService(session=session)
        return GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
        )

    app.dependency_overrides[get_generation_service] = override_get_generation_service

    resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?generate=true",
        json={"role": "user", "content": "Query triggering timeout"},
    )
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]


@pytest.mark.anyio
async def test_api_generation_llm_rate_limit_mapped_to_429(api_client: AsyncClient):
    """Verify HTTP 429 Too Many Requests when LLM provider returns 429."""
    proj_resp = await api_client.post("/projects", json={"name": "Rate Limit Project"})
    project_id = proj_resp.json()["id"]

    conv_resp = await api_client.post(f"/projects/{project_id}/conversations")
    conversation_id = conv_resp.json()["id"]

    mock_rag = MagicMock(spec=RAGService)
    mock_rag.retrieve = AsyncMock(
        return_value=make_mock_retrieval_result(project_id, "Query hitting rate limit")
    )

    mock_adapter = MagicMock(spec=BaseLLMInterface)
    mock_adapter.provider_name = "mock_provider"
    mock_adapter.model_name = "gpt-4o"
    mock_adapter.generate = AsyncMock(
        side_effect=LLMRateLimitError("Rate limit exceeded for model gpt-4o")
    )
    llm_service = LLMService(adapter=mock_adapter)

    def override_get_generation_service(session: AsyncSession = pytest.importorskip("fastapi").Depends(get_db_session)):
        conv_service = ConversationService(session=session)
        return GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
        )

    app.dependency_overrides[get_generation_service] = override_get_generation_service

    resp = await api_client.post(
        f"/projects/{project_id}/conversations/{conversation_id}/messages?generate=true",
        json={"role": "user", "content": "Query hitting rate limit"},
    )
    assert resp.status_code == 429
    assert "rate limit exceeded" in resp.json()["detail"].lower()
