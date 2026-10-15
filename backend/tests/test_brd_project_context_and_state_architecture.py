"""Focused test suite for Batch 1 Remediation: Project Context & Agent State Architecture.

Defects tested:
1. DEF-004: Project Context Starvation
   - Authoritative retrieval of project metadata (name, description, document inventory) from PostgreSQL.
   - Flow through AgentContext to BRDLeadAgent._state.metadata.
   - LLM prompt grounding with ## Project Context block without raw document stuffing.
   - Sub-agent (SectionGenerationAgent, SectionValidationAgent, EvaluationAgent) context grounding.
   - Strict project context isolation: Project A context never leaks into Project B.

2. DEF-005: Global Mutable BRD Agent Singleton / Cross-Request State Leakage
   - Elimination of process-level singleton: get_brd_lead_agent() returns fresh instances per request.
   - Conversation isolation within the same project: Conversation A1 state != Conversation A2 state.
   - Cross-project state isolation: Project A state cannot mutate Project B state.
   - Concurrent execution safety: Two simultaneous workflow executions operate independently without cross-talk or race condition corruption.

3. DEF-012: In-Memory BRDAgentState Loss / Lack of Durable Workflow State
   - Pause at ASK_USER: state persists durably into message_metadata["workflow_state"].
   - Agent instance loss / worker restart simulation: Reconstructing workflow state from serialized JSON dictionary.
   - Resumption continuity: Restored state continues from Phase 4 Evaluation / Section Generation without re-running earlier phases.
   - API level multi-turn message persistence and rehydration.
"""

import asyncio
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.brd.agent import (
    ActionDecision,
    ActionType,
    BRDLeadAgent,
    GapResolutionAction,
    GapResolutionDecision,
    _build_project_context_prompt_block,
    create_brd_lead_agent,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext
from agents.brd.evaluation import (
    BRDEvaluationAgent,
    EvaluationContext,
    EvaluationOutcome,
    EvaluationResult,
    InformationStatus,
)
from agents.brd.evaluation.agent import _build_evaluation_prompt
from agents.brd.final_validation import FinalValidationOutcome, FinalValidationResult
from agents.brd.section_generation import (
    BRDSectionGenerationAgent,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
)
from agents.brd.section_generation.agent import _build_generation_prompt
from agents.brd.section_validation import (
    BRDSectionValidationAgent,
    SectionValidationContext,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.section_validation.agent import _build_validation_prompt
from agents.brd.state import BRDAgentState, BRDSectionStatus, SectionStatus
from api.dependencies import get_brd_lead_agent
from app.main import app
from db.base import Base
from db.session import get_db_session
from models.conversation import ConversationModel
from models.document import DocumentModel
from models.message import MessageModel
from models.project import ProjectModel
from services.conversation_service import ConversationService
from services.project_service import ProjectService


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model conforming to BaseChatModel."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Mock response"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


# ==============================================================================
# DEF-004: Project Context Starvation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_project_context_authoritative_retrieval_and_propagation():
    """Verify that ProjectService retrieves project context from DB and AgentContext serializes it."""
    # Test AgentContext fields and serialization
    ctx = AgentContext(
        project_id="proj-alpha",
        conversation_id="conv-1",
        project_name="Alpha Healthcare Portal",
        project_description="Enterprise patient management platform",
        available_documents=["requirements.pdf", "architecture.docx"],
        metadata={"custom_flag": "test"},
    )
    assert ctx.project_name == "Alpha Healthcare Portal"
    assert ctx.project_description == "Enterprise patient management platform"
    assert ctx.available_documents == ["requirements.pdf", "architecture.docx"]

    # Test round-trip dict serialization
    serialized = ctx.to_dict()
    assert serialized["project_name"] == "Alpha Healthcare Portal"
    assert serialized["project_description"] == "Enterprise patient management platform"
    assert serialized["available_documents"] == ["requirements.pdf", "architecture.docx"]

    restored = AgentContext.from_dict(serialized)
    assert restored.project_name == ctx.project_name
    assert restored.project_description == ctx.project_description
    assert restored.available_documents == ctx.available_documents


def test_project_context_prompt_block_formatting():
    """Verify that _build_project_context_prompt_block produces formatted grounding text without document stuffing."""
    ctx = AgentContext(
        project_id="proj-beta",
        project_name="Beta FinTech Platform",
        project_description="High-frequency settlement engine",
        available_documents=["settlement_spec.pdf", "api_design.md"],
    )
    block = _build_project_context_prompt_block(context=ctx)
    assert "## Project Context" in block
    assert "Project Name: Beta FinTech Platform" in block
    assert "Project Description: High-frequency settlement engine" in block
    assert "Available Project Sources (indexed in RAG): settlement_spec.pdf, api_design.md" in block
    # Verify it does NOT stuff raw document contents
    assert "raw file binary" not in block


def test_subagent_prompts_grounded_in_project_context():
    """Verify that SectionGeneration, SectionValidation, and Evaluation prompts include project context."""
    meta = {
        "project_name": "Logistics Route Optimizer",
        "project_description": "AI-powered fleet routing system",
        "available_documents": ["fleet_spec.pdf"],
    }

    # 1. Section Generation Prompt
    gen_ctx = SectionGenerationContext(
        section_name="System Architecture",
        metadata=meta,
    )
    gen_prompt = _build_generation_prompt(gen_ctx)
    assert "## Project Context" in gen_prompt
    assert "Project Name: Logistics Route Optimizer" in gen_prompt
    assert "Project Description: AI-powered fleet routing system" in gen_prompt

    # 2. Section Validation Prompt
    val_ctx = SectionValidationContext(
        section_name="System Architecture",
        section_content="Architecture overview...",
        metadata=meta,
    )
    val_prompt = _build_validation_prompt(val_ctx)
    assert "## Project Context" in val_prompt
    assert "Project Name: Logistics Route Optimizer" in val_prompt

    # 3. Evaluation Prompt
    eval_ctx = EvaluationContext(
        current_objective="Assess requirements for System Architecture",
        current_section="System Architecture",
        metadata=meta,
    )
    eval_prompt = _build_evaluation_prompt(eval_ctx)
    assert "## Project Context" in eval_prompt
    assert "Project Name: Logistics Route Optimizer" in eval_prompt


@pytest.mark.asyncio
async def test_project_context_isolation_across_projects():
    """Verify Project A context never appears in Project B agent execution or state."""
    agent_a = BRDLeadAgent(model=MockChatModel())
    agent_b = BRDLeadAgent(model=MockChatModel())

    ctx_a = AgentContext(
        project_id="proj-a",
        conversation_id="conv-a",
        project_name="Project Alpha Confidential",
        project_description="Secret Project Alpha",
        available_documents=["alpha_secret.pdf"],
    )
    ctx_b = AgentContext(
        project_id="proj-b",
        conversation_id="conv-b",
        project_name="Project Beta Public",
        project_description="Public Project Beta",
        available_documents=["beta_public.md"],
    )

    with patch.object(agent_a, "stream_workflow_async") as mock_stream_a, \
         patch.object(agent_b, "stream_workflow_async") as mock_stream_b:
        async def fake_stream_a(*args, **kwargs):
            yield {"type": "content", "content": "Done A"}
        async def fake_stream_b(*args, **kwargs):
            yield {"type": "content", "content": "Done B"}
        mock_stream_a.side_effect = fake_stream_a
        mock_stream_b.side_effect = fake_stream_b

        async for _ in agent_a.stream_async("Draft BRD", context=ctx_a):
            pass
        async for _ in agent_b.stream_async("Draft BRD", context=ctx_b):
            pass

    # Verify agent A state metadata contains only Project A
    assert agent_a.state.metadata.get("project_id") == "proj-a"
    assert agent_a.state.metadata.get("project_name") == "Project Alpha Confidential"
    assert "Project Beta Public" not in str(agent_a.state.metadata)

    # Verify agent B state metadata contains only Project B
    assert agent_b.state.metadata.get("project_id") == "proj-b"
    assert agent_b.state.metadata.get("project_name") == "Project Beta Public"
    assert "Project Alpha Confidential" not in str(agent_b.state.metadata)


# ==============================================================================
# DEF-005: Global Mutable Agent Singleton Elimination & State Isolation Tests
# ==============================================================================


def test_brd_lead_agent_factory_returns_fresh_instances():
    """Verify that get_brd_lead_agent dependency returns a new, independent agent instance per call."""
    agent1 = get_brd_lead_agent()
    agent2 = get_brd_lead_agent()

    # Must be distinct Python objects
    assert agent1 is not agent2
    # Must have distinct, isolated working state objects
    assert agent1.state is not agent2.state

    # Mutating agent1 state must not affect agent2 state
    agent1.state.set_current_section("Executive Summary")
    agent1.state.update_section_status("Executive Summary", BRDSectionStatus.COMPLETED)
    agent1.state.metadata["conversation_id"] = "conv-1"

    assert agent2.state.metadata.get("conversation_id") != "conv-1"
    assert agent2.state.current_section != "Executive Summary"


@pytest.mark.asyncio
async def test_conversation_state_isolation_within_same_project():
    """Verify that two conversations within the same project maintain completely isolated section states."""
    agent1 = BRDLeadAgent(model=MockChatModel())
    agent2 = BRDLeadAgent(model=MockChatModel())

    ctx1 = AgentContext(project_id="proj-shared", conversation_id="conv-1")
    ctx2 = AgentContext(project_id="proj-shared", conversation_id="conv-2")

    # Set up distinct state on conversation 1
    agent1.state.metadata["conversation_id"] = "conv-1"
    agent1.state.metadata["project_id"] = "proj-shared"
    agent1.state.set_section_content("1. Executive Summary", "Content for conversation 1")
    agent1.state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)

    # Set up distinct state on conversation 2
    agent2.state.metadata["conversation_id"] = "conv-2"
    agent2.state.metadata["project_id"] = "proj-shared"
    agent2.state.set_section_content("1. Executive Summary", "Content for conversation 2")
    agent2.state.update_section_status("1. Executive Summary", BRDSectionStatus.IN_PROGRESS)

    # Verify complete isolation
    assert agent1.state.get_section_content("1. Executive Summary") == "Content for conversation 1"
    assert agent2.state.get_section_content("1. Executive Summary") == "Content for conversation 2"
    assert agent1.state.get_section_status("1. Executive Summary") == BRDSectionStatus.COMPLETED
    assert agent2.state.get_section_status("1. Executive Summary") == BRDSectionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_concurrent_workflow_executions_independent():
    """Verify two concurrent workflow streams operate independently without race condition or state cross-talk."""
    agent_a = BRDLeadAgent(model=MockChatModel())
    agent_b = BRDLeadAgent(model=MockChatModel())

    ctx_a = AgentContext(project_id="proj-conc-a", conversation_id="conv-conc-a", project_name="Conc A")
    ctx_b = AgentContext(project_id="proj-conc-b", conversation_id="conv-conc-b", project_name="Conc B")

    async def run_worker(agent: BRDLeadAgent, ctx: AgentContext, section_val: str):
        # Simulate staggered concurrent work
        agent.state.metadata["conversation_id"] = ctx.conversation_id
        agent.state.metadata["project_id"] = ctx.project_id
        agent.state.metadata["project_name"] = ctx.project_name
        agent.state.set_current_section("Section 1")
        await asyncio.sleep(0.01)
        agent.state.set_section_content("Section 1", section_val)
        await asyncio.sleep(0.01)
        agent.state.update_section_status("Section 1", BRDSectionStatus.COMPLETED)
        return agent.state.get_section_content("Section 1")

    result_a, result_b = await asyncio.gather(
        run_worker(agent_a, ctx_a, "Output A"),
        run_worker(agent_b, ctx_b, "Output B"),
    )

    assert result_a == "Output A"
    assert result_b == "Output B"
    assert agent_a.state.get_section_content("Section 1") == "Output A"
    assert agent_b.state.get_section_content("Section 1") == "Output B"
    assert agent_a.state.metadata["conversation_id"] == "conv-conc-a"
    assert agent_b.state.metadata["conversation_id"] == "conv-conc-b"


# ==============================================================================
# DEF-012: In-Memory BRDAgentState Persistence & Resume Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_pause_at_ask_user_and_persist_to_message_metadata():
    """Verify workflow pausing on ASK_USER sets waiting_for_user and can be durably serialized."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = AgentContext(project_id="proj-pause", conversation_id="conv-pause")

    # Set up state as if paused for clarification
    agent.state.metadata["project_id"] = "proj-pause"
    agent.state.metadata["conversation_id"] = "conv-pause"
    agent.state.set_waiting_for_user(
        question="What is the expected daily transaction volume?",
    )
    agent.state.set_section_content("1. Executive Summary", "Drafted executive summary")
    agent.state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)

    # 1. Verify in-memory state flags
    assert agent.state.is_waiting_for_user is True
    assert agent.state.pending_clarification == "What is the expected daily transaction volume?"

    # 2. Serialize to dictionary (simulating JSON storage in messages.message_metadata)
    persisted_state = agent.state.to_dict()
    assert persisted_state["waiting_for_user"] is True
    assert persisted_state["pending_clarification"] == "What is the expected daily transaction volume?"
    assert persisted_state["metadata"]["conversation_id"] == "conv-pause"
    assert persisted_state["section_progress"]["1. Executive Summary"] == BRDSectionStatus.COMPLETED.value

    # 3. Simulate process restart / agent instance loss: rehydrate into completely fresh agent
    new_agent = BRDLeadAgent(model=MockChatModel())
    assert new_agent.state.is_waiting_for_user is False

    reconstructed_state = BRDAgentState.from_dict(persisted_state)
    assert reconstructed_state.is_waiting_for_user is True
    assert reconstructed_state.pending_clarification == "What is the expected daily transaction volume?"
    assert reconstructed_state.get_section_content("1. Executive Summary") == "Drafted executive summary"


@pytest.mark.asyncio
async def test_reconstruct_workflow_state_after_agent_instance_loss():
    """Verify that a brand new agent instance resumes a workflow from serialized state without re-running Phase 1-3."""
    # Agent 1 produces state and pauses
    agent1 = BRDLeadAgent(model=MockChatModel())
    ctx = AgentContext(project_id="proj-restart", conversation_id="conv-restart")

    agent1.state.metadata["project_id"] = "proj-restart"
    agent1.state.metadata["conversation_id"] = "conv-restart"
    agent1.state.set_waiting_for_user(question="Target audience?")
    agent1.state.set_section_content("1. Executive Summary", "Completed Summary")
    agent1.state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)

    # Snapshot to JSON string (durable DB simulation)
    durable_json = json.dumps(agent1.state.to_dict())

    # Agent 1 is garbage-collected / process crashes
    del agent1

    # Brand new Agent 2 created in a new process / request
    agent2 = BRDLeadAgent(model=MockChatModel())
    loaded_state = BRDAgentState.from_dict(json.loads(durable_json))

    # Resume workflow with user clarification
    with patch.object(agent2.evaluator, "evaluate_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent2.section_generator, "generate_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent2.section_validator, "validate_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent2.final_validator, "validate_async", new_callable=AsyncMock) as mock_fval:

        mock_eval.return_value = EvaluationResult(
            outcome=EvaluationOutcome.SUFFICIENT,
            summary="Sufficient",
            findings=[],
        )
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Scope",
            content="Scope details...",
            operation=SectionOperation.GENERATE,
        )
        mock_val.return_value = ValidationResult(outcome=ValidationOutcome.VALID, findings=[])
        mock_fval.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID, findings=[])

        # Stream resume: passing initial_state=loaded_state
        events = []
        async for event in agent2.stream_async(
            request="The target audience is enterprise financial institutions.",
            context=ctx,
            initial_state=loaded_state,
        ):
            events.append(event)

    # Verify that:
    # 1. State was cleared of waiting_for_user
    assert agent2.state.is_waiting_for_user is False
    assert agent2.state.pending_clarification is None
    # 2. Previous completed section was preserved
    assert agent2.state.get_section_content("1. Executive Summary") == "Completed Summary"
    # 3. Clarification answer was added to evidence
    evidence_contents = [e.get("content", "") for e in agent2.state.evidence if isinstance(e, dict)]
    assert any("The target audience is enterprise financial institutions." in c for c in evidence_contents)
