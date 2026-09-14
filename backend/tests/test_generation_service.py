"""Comprehensive tests for GenerationService and LLM Generation Integration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from exceptions.conversation import (
    ConversationMessageMismatchError,
    ConversationNotFoundError,
    ProjectConversationMismatchError,
)
from exceptions.generation import (
    EvaluationError,
    GenerationError,
    LLMProviderError,
    LLMTimeoutError,
    RegenerationExhaustedError,
)
from exceptions.retrieval import RetrievalError
from models.base import Base
from models.conversation import ConversationModel
from models.message import MessageModel
from models.project import ProjectModel
from rag.generation.evaluation.models import EvaluationResult
from rag.generation.evaluation.service import EvaluationService
from rag.generation.formatting.models import FormattedContext, FormattedContextItem
from rag.generation.formatting.service import ContextFormattingService
from rag.generation.llm.config import LLMConfig
from rag.generation.llm.interface import BaseLLMInterface
from rag.generation.llm.models import LLMResult, LLMUsage
from rag.generation.llm.service import LLMService
from rag.generation.postprocessing.models import ProcessedResponse
from rag.generation.postprocessing.service import PostProcessingService
from rag.generation.prompt.models import ConstructedPrompt
from rag.generation.prompt.service import PromptConstructionService
from rag.retrieval.models import (
    AssembledContext,
    AssembledContextItem,
    RetrievalExecutionMetadata,
    RetrievalResult,
    RetrievedChunk,
)
from rag.retrieval.service import RAGService
from schemas.generation import GenerationResult
from services.conversation_service import ConversationService
from services.generation_service import GenerationService, get_generation_service, reset_generation_service
from services.project_service import ProjectService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_session() -> AsyncIterator[AsyncSession]:
    """Create in-memory SQLite test database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as session:
        yield session

    await engine.dispose()


def make_sample_retrieval_result(
    project_id: str,
    query: str,
    chunks_count: int = 2,
) -> RetrievalResult:
    """Helper creating a valid RetrievalResult."""
    items = tuple(
        FormattedContextItem(
            index=i,
            text=f"Sample context content {i} for project {project_id}.",
            content=f"Sample context content {i} for project {project_id}.",
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            source=f"source_{i}.txt",
        )
        for i in range(1, chunks_count + 1)
    )
    formatted_ctx = FormattedContext(
        text="\n\n".join(f"[Context {item.index}]\n{item.content}" for item in items),
        items=items,
        item_count=len(items),
        project_id=project_id,
    )
    retrieved_chunks = tuple(
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            project_id=project_id,
            document_version_id=f"ver-{i}",
            content=f"Sample context content {i} for project {project_id}.",
            rank=i,
            score=0.95 - (0.05 * i),
            heading=f"Section {i}",
            section_path=(f"Section {i}",),
            contextual_content=None,
            metadata={"source": f"source_{i}.txt"},
        )
        for i in range(1, chunks_count + 1)
    )
    exec_meta = RetrievalExecutionMetadata(
        total_duration_ms=45.5,
        attempts_count=1,
        fallback_triggered=False,
        attempts=(),
        stage_latencies_ms={"dense_search_ms": 20.0, "reranking_ms": 15.0},
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


# ==============================================================================
# 1. Happy Path End-to-End Generation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_end_to_end_generation_flow():
    """Verify User Message -> RAG -> Prompt -> LLM -> Evaluation -> Assistant Message."""
    async with create_test_session() as session:
        # 1. Setup Project and Conversation
        proj = ProjectModel(id="proj-100", name="Test Project")
        conv = ConversationModel(id="conv-100", project_id="proj-100", title="Test Chat")
        session.add_all([proj, conv])
        await session.commit()

        # 2. Mock RAG Service
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-100", "What is the return policy?")
        )

        # 3. Mock LLM Interface & Service
        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.provider_name = "mock_provider"
        mock_llm_adapter.model_name = "gpt-4o"
        mock_llm_adapter.generate = AsyncMock(
            return_value=LLMResult(
                content="The return policy allows returns within 30 days with receipt.",
                finish_reason="stop",
                usage=LLMUsage(input_tokens=150, output_tokens=35, total_tokens=185),
                metadata={"model": "gpt-4o"},
            )
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        # 4. Mock Evaluator (Grounded and Safe)
        mock_evaluator = MagicMock(spec=EvaluationService)
        mock_evaluator.config = MagicMock(max_regeneration_attempts=2)
        mock_evaluator.evaluate_async = AsyncMock(
            return_value=EvaluationResult(
                grounded=True,
                safe=True,
                reason="Response is fully supported by context and satisfies safety policies.",
                score=1.0,
            )
        )

        # 5. Instantiate GenerationService
        conv_service = ConversationService(session=session)
        gen_service = GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

        # 6. Execute Generation
        result = await gen_service.generate_response(
            project_id="proj-100",
            conversation_id="conv-100",
            user_message="What is the return policy?",
        )

        # 7. Assertions on GenerationResult
        assert isinstance(result, GenerationResult)
        assert result.project_id == "proj-100"
        assert result.conversation_id == "conv-100"
        assert "within 30 days" in result.content
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 185
        assert result.evaluation_metadata["grounded"] is True
        assert result.evaluation_metadata["safe"] is True
        assert result.execution_metadata["generation_attempts_count"] == 1
        assert result.execution_metadata["regeneration_triggered"] is False

        # 8. Verify Persisted Database Records
        messages = await conv_service.list_messages(project_id="proj-100", conversation_id="conv-100")
        assert len(messages) == 2
        user_msg, asst_msg = messages[0], messages[1]

        # User message turn
        assert user_msg.role == "user"
        assert user_msg.content == "What is the return policy?"

        # Assistant message turn
        assert asst_msg.role == "assistant"
        assert asst_msg.content == "The return policy allows returns within 30 days with receipt."
        assert asst_msg.message_metadata["user_message_id"] == user_msg.id
        assert asst_msg.message_metadata["evaluation_passed"] is True
        assert asst_msg.message_metadata["finish_reason"] == "stop"
        assert asst_msg.message_metadata["retrieval"]["chunks_count"] == 2

        # 9. Verify Stage Invocations
        mock_rag.retrieve.assert_awaited_once_with(
            project_id="proj-100",
            query="What is the return policy?",
            config=None,
            session=session,
        )
        mock_llm_adapter.generate.assert_awaited_once()
        mock_evaluator.evaluate_async.assert_awaited_once()


# ==============================================================================
# 2. Single Source of Truth & Configuration Reuse
# ==============================================================================


@pytest.mark.anyio
async def test_single_source_of_truth_configuration():
    """Verify GenerationService consumes existing Settings and LLMConfig without duplicate objects."""
    async with create_test_session() as session:
        settings = get_settings()
        gen_service = GenerationService(session=session, settings=settings)

        # LLM config must come from existing LLMConfig
        assert isinstance(gen_service.llm_service.config, LLMConfig)
        assert gen_service.llm_service.config.model == settings.LLM_MODEL
        assert gen_service.llm_service.config.timeout == settings.LLM_TIMEOUT
        assert gen_service.llm_service.config.max_retries == settings.LLM_MAX_RETRIES

        # Verify no parallel config objects were created
        assert not hasattr(gen_service, "generation_config")
        assert not hasattr(gen_service, "chat_config")


@pytest.mark.anyio
async def test_existing_llm_interface_reuse():
    """Verify GenerationService calls the existing BaseLLMInterface abstraction."""
    async with create_test_session() as session:
        mock_adapter = MagicMock(spec=BaseLLMInterface)
        mock_adapter.provider_name = "test_provider"
        mock_adapter.model_name = "test_model"
        mock_adapter.generate = AsyncMock()

        llm_service = LLMService(adapter=mock_adapter)
        gen_service = GenerationService(session=session, llm_service=llm_service)

        assert gen_service.llm_service.adapter is mock_adapter


# ==============================================================================
# 3. Bounded Semantic Regeneration on Quality Gate Failure
# ==============================================================================


@pytest.mark.anyio
async def test_bounded_regeneration_success():
    """Verify that when attempt 1 is rejected by evaluator, attempt 2 succeeds with feedback."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-200", name="Regen Project")
        conv = ConversationModel(id="conv-200", project_id="proj-200", title="Regen Chat")
        session.add_all([proj, conv])
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-200", "What is the warranty period?")
        )

        # Attempt 1: halluncinated claim; Attempt 2: corrected grounded claim
        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.provider_name = "mock_provider"
        mock_llm_adapter.model_name = "gpt-4o"
        mock_llm_adapter.generate = AsyncMock(
            side_effect=[
                LLMResult(
                    content="The warranty period is lifetime coverage.",
                    finish_reason="stop",
                    usage=LLMUsage(input_tokens=100, output_tokens=20, total_tokens=120),
                ),
                LLMResult(
                    content="The warranty period is 1 year as stated in the context.",
                    finish_reason="stop",
                    usage=LLMUsage(input_tokens=150, output_tokens=25, total_tokens=175),
                ),
            ]
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        # Evaluator: rejects attempt 1 (ungrounded), accepts attempt 2
        mock_evaluator = MagicMock(spec=EvaluationService)
        mock_evaluator.config = MagicMock(max_regeneration_attempts=2)
        mock_evaluator.evaluate_async = AsyncMock(
            side_effect=[
                EvaluationResult(
                    grounded=False,
                    safe=True,
                    reason="Context does not support lifetime warranty claim. Context states 1 year.",
                ),
                EvaluationResult(
                    grounded=True,
                    safe=True,
                    reason="Claims are fully grounded in retrieved context.",
                ),
            ]
        )

        gen_service = GenerationService(
            session=session,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

        result = await gen_service.generate_response(
            project_id="proj-200",
            conversation_id="conv-200",
            user_message="What is the warranty period?",
        )

        assert result.content == "The warranty period is 1 year as stated in the context."
        assert result.execution_metadata["generation_attempts_count"] == 2
        assert result.execution_metadata["regeneration_triggered"] is True
        assert mock_llm_adapter.generate.await_count == 2
        assert mock_evaluator.evaluate_async.await_count == 2


@pytest.mark.anyio
async def test_regeneration_exhaustion_raises_error():
    """Verify that when all regeneration attempts are rejected, RegenerationExhaustedError is raised.

    Crucially verifies that:
    1. The user message remains persisted in the database.
    2. NO fake or ungrounded assistant message is persisted.
    """
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-300", name="Exhaust Project")
        conv = ConversationModel(id="conv-300", project_id="proj-300", title="Exhaust Chat")
        session.add_all([proj, conv])
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-300", "Complex unsupported query")
        )

        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.provider_name = "mock_provider"
        mock_llm_adapter.model_name = "gpt-4o"
        mock_llm_adapter.generate = AsyncMock(
            return_value=LLMResult(
                content="Fabricated ungrounded answer.",
                finish_reason="stop",
            )
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        # Evaluator rejects all attempts
        mock_evaluator = MagicMock(spec=EvaluationService)
        mock_evaluator.config = MagicMock(max_regeneration_attempts=1)
        mock_evaluator.evaluate_async = AsyncMock(
            return_value=EvaluationResult(
                grounded=False,
                safe=True,
                reason="Context does not contain evidence for this question.",
            )
        )

        gen_service = GenerationService(
            session=session,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

        with pytest.raises(RegenerationExhaustedError) as exc_info:
            await gen_service.generate_response(
                project_id="proj-300",
                conversation_id="conv-300",
                user_message="Complex unsupported query",
                max_regeneration_attempts=1,  # 1 initial + 1 retry = 2 total
            )

        assert exc_info.value.attempts == 2
        assert "Context does not contain evidence" in exc_info.value.message

        # Assert: User message is safely stored in DB
        conv_service = ConversationService(session=session)
        messages = await conv_service.list_messages(project_id="proj-300", conversation_id="conv-300")
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Complex unsupported query"

        # Assert: No fake assistant message was created!
        asst_messages = [m for m in messages if m.role == "assistant"]
        assert len(asst_messages) == 0


@pytest.mark.anyio
async def test_safety_rejection_triggers_regeneration():
    """Verify that a response rejected for safety triggers regeneration feedback."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-400", name="Safety Project")
        conv = ConversationModel(id="conv-400", project_id="proj-400")
        session.add_all([proj, conv])
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-400", "How to bypass auth?")
        )

        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.provider_name = "mock_provider"
        mock_llm_adapter.model_name = "gpt-4o"
        mock_llm_adapter.generate = AsyncMock(
            side_effect=[
                LLMResult(content="Here is an exploit script.", finish_reason="stop"),
                LLMResult(content="I cannot provide exploit instructions. Please consult security docs.", finish_reason="stop"),
            ]
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        mock_evaluator = MagicMock(spec=EvaluationService)
        mock_evaluator.config = MagicMock(max_regeneration_attempts=1)
        mock_evaluator.evaluate_async = AsyncMock(
            side_effect=[
                EvaluationResult(grounded=True, safe=False, reason="Malicious code exploit rejected."),
                EvaluationResult(grounded=True, safe=True, reason="Safe refusal complies with guidelines."),
            ]
        )

        gen_service = GenerationService(
            session=session,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

        result = await gen_service.generate_response(
            project_id="proj-400",
            conversation_id="conv-400",
            user_message="How to bypass auth?",
        )

        assert "cannot provide exploit instructions" in result.content
        assert result.execution_metadata["generation_attempts_count"] == 2


# ==============================================================================
# 4. Project Boundary & Isolation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_project_isolation_cross_project_rejected():
    """Verify that attempting generation on another project's conversation raises mismatch error."""
    async with create_test_session() as session:
        proj_a = ProjectModel(id="proj-A", name="Tenant A")
        proj_b = ProjectModel(id="proj-B", name="Tenant B")
        conv_a = ConversationModel(id="conv-A", project_id="proj-A", title="Tenant A Confidential")
        session.add_all([proj_a, proj_b, conv_a])
        await session.commit()

        gen_service = GenerationService(session=session)

        # Tenant B attempts to generate on Tenant A's conversation
        with pytest.raises(ProjectConversationMismatchError) as exc_info:
            await gen_service.generate_response(
                project_id="proj-B",
                conversation_id="conv-A",
                user_message="Attempted cross-tenant query",
            )

        assert exc_info.value.conversation_id == "conv-A"
        assert exc_info.value.expected_project_id == "proj-B"
        assert exc_info.value.actual_project_id == "proj-A"


@pytest.mark.anyio
async def test_nonexistent_conversation_raises_not_found():
    """Verify that targeting a non-existent conversation raises ConversationNotFoundError."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-500", name="Project 500")
        session.add(proj)
        await session.commit()

        gen_service = GenerationService(session=session)

        with pytest.raises(ConversationNotFoundError):
            await gen_service.generate_response(
                project_id="proj-500",
                conversation_id="conv-nonexistent",
                user_message="Query",
            )


# ==============================================================================
# 5. Pre-persisted User Message & Transaction Isolation
# ==============================================================================


@pytest.mark.anyio
async def test_pre_persisted_user_message():
    """Verify that passing an existing MessageModel avoids creating duplicate user messages."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-600", name="Pre-persisted Project")
        conv = ConversationModel(id="conv-600", project_id="proj-600")
        session.add_all([proj, conv])
        await session.commit()

        conv_service = ConversationService(session=session)
        user_msg = await conv_service.create_message(
            project_id="proj-600",
            conversation_id="conv-600",
            content="Pre-existing user message",
            role="user",
        )
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-600", "Pre-existing user message")
        )

        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.provider_name = "mock_provider"
        mock_llm_adapter.model_name = "gpt-4o"
        mock_llm_adapter.generate = AsyncMock(
            return_value=LLMResult(content="Answer to pre-existing turn.", finish_reason="stop")
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        mock_evaluator = MagicMock(spec=EvaluationService)
        mock_evaluator.config = MagicMock(max_regeneration_attempts=1)
        mock_evaluator.evaluate_async = AsyncMock(
            return_value=EvaluationResult(grounded=True, safe=True, reason="Grounded")
        )

        gen_service = GenerationService(
            session=session,
            conversation_service=conv_service,
            rag_service=mock_rag,
            llm_service=llm_service,
            evaluation_service=mock_evaluator,
        )

        result = await gen_service.generate_response(
            project_id="proj-600",
            conversation_id="conv-600",
            user_message=user_msg,
        )

        # Ensure total messages is exactly 2: 1 user, 1 assistant
        messages = await conv_service.list_messages(project_id="proj-600", conversation_id="conv-600")
        assert len(messages) == 2
        assert messages[0].id == user_msg.id
        assert messages[1].id == result.assistant_message_id


@pytest.mark.anyio
async def test_pre_persisted_user_message_conversation_mismatch():
    """Verify that passing a MessageModel from a different conversation raises mismatch error."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-700", name="Project 700")
        conv1 = ConversationModel(id="conv-1", project_id="proj-700")
        conv2 = ConversationModel(id="conv-2", project_id="proj-700")
        session.add_all([proj, conv1, conv2])
        await session.commit()

        conv_service = ConversationService(session=session)
        msg_in_conv1 = await conv_service.create_message(
            project_id="proj-700",
            conversation_id="conv-1",
            content="Message in Conv 1",
            role="user",
        )
        await session.commit()

        gen_service = GenerationService(session=session, conversation_service=conv_service)

        with pytest.raises(ConversationMessageMismatchError):
            await gen_service.generate_response(
                project_id="proj-700",
                conversation_id="conv-2",
                user_message=msg_in_conv1,
            )


# ==============================================================================
# 6. Failure Handling (RAG, LLM, Timeout)
# ==============================================================================


@pytest.mark.anyio
async def test_rag_failure_preserves_user_message():
    """Verify that if RAG retrieval fails, user message remains in DB and no assistant message is saved."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-800", name="RAG Fail Project")
        conv = ConversationModel(id="conv-800", project_id="proj-800")
        session.add_all([proj, conv])
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(side_effect=RetrievalError("Qdrant connection timed out"))

        gen_service = GenerationService(
            session=session,
            rag_service=mock_rag,
        )

        with pytest.raises(RetrievalError) as exc_info:
            await gen_service.generate_response(
                project_id="proj-800",
                conversation_id="conv-800",
                user_message="Query triggering RAG failure",
            )

        assert "Qdrant connection timed out" in str(exc_info.value)

        # Verify user message was preserved
        conv_service = ConversationService(session=session)
        messages = await conv_service.list_messages(project_id="proj-800", conversation_id="conv-800")
        assert len(messages) == 1
        assert messages[0].role == "user"


@pytest.mark.anyio
async def test_llm_timeout_preserves_user_message():
    """Verify that if LLM provider times out, user message remains in DB and no assistant message is saved."""
    async with create_test_session() as session:
        proj = ProjectModel(id="proj-900", name="LLM Timeout Project")
        conv = ConversationModel(id="conv-900", project_id="proj-900")
        session.add_all([proj, conv])
        await session.commit()

        mock_rag = MagicMock(spec=RAGService)
        mock_rag.retrieve = AsyncMock(
            return_value=make_sample_retrieval_result("proj-900", "Query triggering timeout")
        )

        mock_llm_adapter = MagicMock(spec=BaseLLMInterface)
        mock_llm_adapter.generate = AsyncMock(
            side_effect=LLMTimeoutError("LLM provider timed out after 30s")
        )
        llm_service = LLMService(adapter=mock_llm_adapter)

        gen_service = GenerationService(
            session=session,
            rag_service=mock_rag,
            llm_service=llm_service,
        )

        with pytest.raises(LLMTimeoutError):
            await gen_service.generate_response(
                project_id="proj-900",
                conversation_id="conv-900",
                user_message="Query triggering timeout",
            )

        conv_service = ConversationService(session=session)
        messages = await conv_service.list_messages(project_id="proj-900", conversation_id="conv-900")
        assert len(messages) == 1
        assert messages[0].role == "user"


# ==============================================================================
# 7. Factory Function Tests
# ==============================================================================


@pytest.mark.anyio
async def test_get_generation_service_factory():
    """Verify get_generation_service factory initializes properly."""
    async with create_test_session() as session:
        service = get_generation_service(session=session)
        assert isinstance(service, GenerationService)
        assert service.session is session
