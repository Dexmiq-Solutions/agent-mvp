"""Tests for the Application-Owned BRD Workflow Architecture.

Verifies:
1. System instruction decoupling: Full workflow executes even with blank/empty system instruction.
2. Three-tier responsibility boundary:
   - Tier 1: Lead Agent owns intelligence, action decisions, analytical direct work, evaluation interpretation, section rework strategy, and final validation recovery strategy.
   - Tier 2: Specialized sub-agents evaluate/generate/validate and report findings without workflow/recovery decisions.
   - Tier 3: Application control flow executes sequence, enforces progression gates, enforces max 2 section reworks, enforces assembly gate, enforces max 3 recovery cycles, and logs correlated traces.
3. Observability & Correlated Logging: agent_run_id correlated across APPLICATION, LEAD_AGENT, SPECIALIZED_AGENT, TOOL actors.
4. Streaming progress events and execution parity.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd.agent import (
    ActionDecision,
    ActionType,
    BRDLeadAgent,
    FinalValidationStrategy,
    GapResolutionAction,
    GapResolutionDecision,
    SectionReworkStrategy,
    WorkflowDecision,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext
from agents.brd.evaluation import (
    EvaluationFinding,
    EvaluationOutcome,
    EvaluationResult,
    InformationStatus,
)
from agents.brd.final_validation import (
    FinalValidationCategory,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
    FinalValidationSeverity,
)
from agents.brd.section_generation import (
    SectionGenerationResult,
    SectionOperation,
)
from agents.brd.section_validation import (
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.state import BRDAgentState, SectionStatus
from observability.logging import TraceActor, format_trace_event, log_trace_event


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

class MockChatModel(BaseChatModel):
    """Deterministic mock chat model conforming to BaseChatModel."""

    messages_to_return: list[AIMessage] = []
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default fallback"))]
            )
        msg = self.messages_to_return[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def make_test_context(project_id: str = "proj-test-123", run_id: str = "run-trace-456") -> AgentContext:
    return AgentContext(
        project_id=project_id,
        conversation_id="conv-test-789",
        metadata={"agent_run_id": run_id, "project_id": project_id},
    )


# ---------------------------------------------------------------------------
# 1. System Instruction Decoupling Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_runs_with_empty_system_instruction():
    """Verify that the BRD workflow executes end-to-end even when system instruction is completely empty.

    This proves that the application owns and executes the workflow, rather than relying on
    the LLM system instruction to steer the phase progression.
    """
    agent = BRDLeadAgent(model=MockChatModel(), system_instruction="")
    assert agent.system_instruction == ""

    ctx = make_test_context()

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

        mock_decide.return_value = ActionDecision(
            action_type=ActionType.DIRECT_WORK,
            direct_work_objective="Analyze business objective directly",
            reasoning="Initial objective analysis",
        )
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
            reasoning="All required information is sufficient",
        )
        mock_gen.return_value = SectionGenerationResult(
            section_name="1. Executive Summary",
            content="# 1. Executive Summary\nApproved content.",
            operation=SectionOperation.GENERATE,
        )
        mock_val.return_value = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
            rework_feedback="",
        )
        mock_assembly.return_value = BRDAssemblyResult(
            assembled_document="# Business Requirements Document\n\n## 1. Executive Summary\nContent",
            sections_assembled=list(agent.sections),
            section_count=len(agent.sections),
            assembly_complete=True,
        )
        mock_final_val.return_value = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="All document sections are complete and consistent.",
        )

        state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
        state.current_section = "1. Executive Summary"

        response = await agent.run_workflow_async(
            objective="Develop cloud-native payment gateway",
            context=ctx,
            initial_state=state,
        )

        assert response is not None
        assert response.success is True
        assert mock_decide.call_count >= 1
        assert mock_gen.call_count >= 1
        assert mock_val.call_count >= 1
        assert mock_assembly.call_count >= 1
        assert mock_final_val.call_count >= 1


# ---------------------------------------------------------------------------
# 2. Lead Agent Intelligence & Decision Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lead_agent_decide_action_rag():
    """Verify Lead Agent intelligently decides RAG when information is needed from project docs."""
    mock_model = MockChatModel(
        messages_to_return=[
            AIMessage(
                content='{"action_type": "rag", "rationale": "Need security compliance docs", "rag_queries": ["PCI-DSS compliance architecture"]}'
            )
        ]
    )
    agent = BRDLeadAgent(model=mock_model)
    ctx = make_test_context()

    decision = await agent.decide_action_async(
        objective="Build PCI-DSS compliant checkout",
        context=ctx,
        section_name="Security Requirements",
    )

    assert decision.action_type == ActionType.RAG
    assert "security" in decision.reasoning.lower()
    assert decision.query is not None
    assert "pci" in decision.query.lower()


@pytest.mark.asyncio
async def test_lead_agent_decide_action_delegation():
    """Verify Lead Agent intelligently decides Delegation when complex domain tasks are required."""
    mock_model = MockChatModel(
        messages_to_return=[
            AIMessage(
                content='{"action_type": "delegation", "rationale": "Decompose high-scale microservices", "delegated_tasks": ["Define REST endpoints", "Design schema"]}'
            )
        ]
    )
    agent = BRDLeadAgent(model=mock_model)
    ctx = make_test_context()

    decision = await agent.decide_action_async(
        objective="Design microservices architecture",
        context=ctx,
    )

    assert decision.action_type == ActionType.DELEGATION
    assert len(decision.delegated_tasks) >= 2


@pytest.mark.asyncio
async def test_lead_agent_decide_action_direct_work():
    """Verify Lead Agent intelligently decides Direct Work when synthesizing known facts."""
    mock_model = MockChatModel(
        messages_to_return=[
            AIMessage(
                content='{"action_type": "direct_work", "rationale": "Directly summarize project background", "direct_work_objective": "Summarize background"}'
            )
        ]
    )
    agent = BRDLeadAgent(model=mock_model)
    ctx = make_test_context()

    decision = await agent.decide_action_async(
        objective="Project Background Summary",
        context=ctx,
        section_name="Project Background",
    )

    assert decision.action_type == ActionType.DIRECT_WORK
    assert decision.direct_work_objective == "Summarize background"


@pytest.mark.asyncio
async def test_lead_agent_interpret_evaluation_sufficient():
    """Verify Lead Agent interprets sufficient evaluation as ready to generate."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.SUFFICIENT,
        summary="All scope requirements identified.",
        findings=[
            EvaluationFinding(
                observation="Scope is fully defined",
                status=InformationStatus.PRESENT,
            )
        ],
    )

    decision = await agent.interpret_evaluation_async(
        eval_result=eval_result,
        context=ctx,
    )

    assert decision.action == GapResolutionAction.PROCEED_TO_SECTION_GENERATION
    assert "sufficient" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_lead_agent_interpret_evaluation_gap_targeted_rag():
    """Verify Lead Agent interprets missing information gap and formulates targeted RAG queries."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.INSUFFICIENT,
        summary="GDPR retention details are absent.",
        findings=[
            EvaluationFinding(
                observation="Missing GDPR retention policy details",
                status=InformationStatus.MISSING,
            )
        ],
        missing_information=["GDPR data retention timeframe"],
    )

    decision = await agent.interpret_evaluation_async(
        eval_result=eval_result,
        context=ctx,
    )

    assert decision.action == GapResolutionAction.RAG
    assert decision.query is not None
    assert "gdpr" in decision.query.lower()


@pytest.mark.asyncio
async def test_lead_agent_interpret_section_validation():
    """Verify Lead Agent interprets validation feedback into actionable rework strategy."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    validation_result = ValidationResult(
        section_name="Functional Requirements",
        outcome=ValidationOutcome.NEEDS_REWORK,
        rework_feedback="Missing acceptance criteria for checkout flow",
        findings=[
            ValidationFinding(
                category=ValidationCategory.COMPLETENESS,
                issue="Acceptance criteria absent",
                explanation="Acceptance criteria is required",
                required_change="Add acceptance criteria for checkout flow",
            )
        ],
    )

    strategy = await agent.interpret_section_validation_async(
        validation_result=validation_result,
        section_name="Functional Requirements",
        current_content="Basic checkout requirements...",
        context=ctx,
    )

    assert isinstance(strategy, SectionReworkStrategy)
    assert strategy.section_name == "Functional Requirements"
    assert strategy.requires_rework is True
    assert "acceptance criteria" in strategy.rework_guidance.lower()


@pytest.mark.asyncio
async def test_lead_agent_interpret_final_validation():
    """Verify Lead Agent interprets complete document final validation into recovery strategy."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    first_sec = agent.sections[0]
    final_val_result = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Inconsistencies detected across document",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Database specification inconsistency",
                affected_sections=[first_sec],
            )
        ],
    )

    strategy = await agent.interpret_final_validation_async(
        final_validation_result=final_val_result,
        assembled_content="# BRD\n...",
        context=ctx,
    )

    assert isinstance(strategy, FinalValidationStrategy)
    assert strategy.requires_recovery is True
    assert first_sec in strategy.affected_sections


# ---------------------------------------------------------------------------
# 3. Application Control Flow & Enforcement Gates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_application_progression_gate_blocks_invalid_section():
    """Verify Application Gate: A section CANNOT advance to completed if validation fails."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    state = BRDAgentState.initialize_from_template(sections=["Section 1", "Section 2"])
    state.current_section = "Section 1"

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK, reasoning="Direct analysis")
        mock_eval.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION, reasoning="Proceed")
        mock_gen.return_value = SectionGenerationResult(
            section_name="Section 1",
            content="Draft",
            operation=SectionOperation.GENERATE,
        )
        # Validation consistently returns NEEDS_REWORK
        mock_val.return_value = ValidationResult(
            section_name="Section 1",
            outcome=ValidationOutcome.NEEDS_REWORK,
            rework_feedback="Fails requirements",
        )

        response = await agent.run_workflow_async(
            objective="Test progression gate",
            context=ctx,
            initial_state=state,
        )

        # Section 1 must NOT be marked COMPLETED
        assert state.get_section_status("Section 1") != SectionStatus.COMPLETED
        # Must not advance to Section 2 as completed
        assert state.get_section_status("Section 2") != SectionStatus.COMPLETED
        assert response.success is False


@pytest.mark.asyncio
async def test_application_enforces_max_2_section_reworks():
    """Verify Application Boundary: Application strictly bounds section rework attempts to 2."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    state = BRDAgentState.initialize_from_template(sections=["Critical Section"])
    state.current_section = "Critical Section"

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK, reasoning="Direct")
        mock_eval.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION, reasoning="Ready")
        mock_gen.return_value = SectionGenerationResult(
            section_name="Critical Section",
            content="Draft",
            operation=SectionOperation.GENERATE,
        )
        mock_update.return_value = SectionGenerationResult(
            section_name="Critical Section",
            content="Draft v2",
            operation=SectionOperation.UPDATE,
        )
        mock_val.return_value = ValidationResult(
            section_name="Critical Section",
            outcome=ValidationOutcome.NEEDS_REWORK,
            rework_feedback="Still needs rework",
        )

        await agent.run_workflow_async(
            objective="Test rework limit",
            context=ctx,
            initial_state=state,
            max_section_rework_attempts=2,
        )

        # Max 2 reworks: initial gen called once, updates called at most 2 times
        assert mock_gen.call_count == 1
        assert mock_update.call_count <= 2
        # Validation called for initial + up to 2 reworks = 3 calls max
        assert mock_val.call_count <= 3


@pytest.mark.asyncio
async def test_application_assembly_gate_blocks_incomplete_brd():
    """Verify Application Gate: assemble_brd_async fails safely if sections are incomplete."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    # All 16 sections in fresh state are NOT_STARTED; calling assemble_brd_async must raise ValueError
    with pytest.raises(ValueError, match="Cannot assemble BRD"):
        await agent.assemble_brd_async(context=ctx)


@pytest.mark.asyncio
async def test_application_enforces_max_3_final_validation_recovery_cycles():
    """Verify Application Boundary: Application strictly bounds complete BRD recovery to 3 cycles."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    # Mark all template sections completed with content so initial assembly succeeds
    for sec in agent.sections:
        agent.state.set_section_content(sec, f"## {sec}\nContent")
        agent.state.update_section_status(sec, SectionStatus.COMPLETED)

    first_sec = agent.sections[0]

    with patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "rewrite_brd_async", new_callable=AsyncMock) as mock_rewrite:

        # Always returns NEEDS_REWORK
        mock_val.return_value = FinalValidationResult(
            outcome=FinalValidationOutcome.NEEDS_REWORK,
            summary="Persistent document issue",
            findings=[
                FinalValidationFinding(
                    finding_id="FV-001",
                    category=FinalValidationCategory.COMPLETENESS,
                    severity=FinalValidationSeverity.ERROR,
                    issue="Missing key detail",
                    location="Section 1",
                    affected_sections=[first_sec],
                )
            ],
        )
        from agents.brd.rewriter.agent import BRDRewriterResult
        mock_rewrite.return_value = BRDRewriterResult(
            summary="Attempted targeted edit",
            edits=[],
            unapplied_findings=["FV-001"],
        )

        recovery_result = await agent.recover_final_validation_async(
            context=ctx,
        )

        # Application terminates correction cycle immediately after Pass 2
        assert recovery_result.recovery_cycles == 1
        assert recovery_result.max_cycles_exhausted is True
        assert recovery_result.success is False
        assert mock_rewrite.await_count == 1
        assert mock_val.await_count == 2  # Pass 1 and Pass 2


# ---------------------------------------------------------------------------
# 4. Observability & Correlated Logging Tests
# ---------------------------------------------------------------------------

def test_trace_actor_values_and_formatting():
    """Verify TraceActor enumeration and trace format string."""
    assert TraceActor.APPLICATION == "APPLICATION"
    assert TraceActor.LEAD_AGENT == "LEAD_AGENT"
    assert TraceActor.SPECIALIZED_AGENT == "SPECIALIZED_AGENT"
    assert TraceActor.TOOL == "TOOL"

    msg = format_trace_event(
        agent_run_id="run-test-999",
        actor=TraceActor.APPLICATION,
        event_name="phase_transition",
        phase="evidence_evaluation",
        status="started",
    )
    assert "[ACTOR: APPLICATION]" in msg
    assert "[RUN: run-test-999]" in msg
    assert "[EVENT: phase_transition]" in msg
    assert 'phase="evidence_evaluation"' in msg
    assert 'status="started"' in msg


def test_log_trace_event_emits_to_logger(caplog):
    """Verify log_trace_event outputs with correct format and metadata."""
    caplog.set_level(logging.INFO)
    test_logger = logging.getLogger("test_trace")

    log_trace_event(
        logger=test_logger,
        agent_run_id="run-trace-abc",
        actor=TraceActor.LEAD_AGENT,
        event_name="decide_action",
        decision="rag",
        reason="Missing compliance information",
    )

    assert any(
        "[ACTOR: LEAD_AGENT]" in record.message and "run-trace-abc" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 5. Workflow Streaming Parity Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_workflow_async_emits_progress_and_content():
    """Verify stream_workflow_async yields progress events for phases and content tokens."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
    state.current_section = "1. Executive Summary"

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_asm, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK, reasoning="Analysis")
        mock_eval.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION, reasoning="Ready")
        mock_gen.return_value = SectionGenerationResult(
            section_name="1. Executive Summary",
            content="# 1. Executive Summary\nContent body",
            operation=SectionOperation.GENERATE,
        )
        mock_val.return_value = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
        )
        mock_asm.return_value = BRDAssemblyResult(
            assembled_document="# Final BRD\n\n## 1. Executive Summary\nContent body",
            sections_assembled=["1. Executive Summary"],
            section_count=1,
            assembly_complete=True,
        )
        mock_final_val.return_value = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="All sections verified.",
        )

        events: list[dict[str, Any]] = []
        async for event in agent.stream_workflow_async(
            objective="Develop payment gateway",
            context=ctx,
            initial_state=state,
        ):
            events.append(event)

        # Must have progress events
        progress_events = [e for e in events if e.get("type") == "progress"]
        assert len(progress_events) >= 5

        # Check that key phases are emitted
        phases_emitted = {e.get("phase") for e in progress_events}
        assert "1_INITIAL_CONTEXT" in phases_emitted
        assert "2_ACTION_DECISION" in phases_emitted
        assert "3_ACTION_EXECUTION" in phases_emitted
        assert "4_EVIDENCE_EVALUATION" in phases_emitted
        assert "5_SECTION_ITERATION" in phases_emitted
        assert "6_DOCUMENT_ASSEMBLY" in phases_emitted
        assert "7_FINAL_VALIDATION" in phases_emitted
        assert "9_COMPLETION" in phases_emitted

        # Must have content events for the assembled document
        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) >= 1
        full_content = "".join(e.get("content", "") for e in content_events)
        assert len(full_content) > 0
