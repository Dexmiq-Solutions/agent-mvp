"""Focused tests for BRD Agent Controlled Workflow Entry and ASK_USER Pause/Resume.

Verifies:
1. Workflow Entry:
   - Every normal user-facing BRD message enters the controlled BRD workflow.
   - No workflow-selection flag (run_workflow=True/False) is required in query, body, or metadata.
   - Generic chat fallback (_graph.astream) is unreachable from stream_async.
2. ASK_USER Pause:
   - When Evidence Evaluation / Lead Agent yields GapResolutionAction.ASK_USER, execution
     stops immediately without continuing into section generation, validation, or assembly.
   - State correctly records waiting_for_user=True and pending_clarification.
   - In streaming, the clarification question is yielded to the user.
3. Resume from Clarification / Follow-up:
   - Subsequent user message incorporates the clarification into evidence context.
   - Clears waiting_for_user status.
   - Resumes workflow at Phase 4 Evidence Evaluation without restarting initial actions or skipping required phases.
4. Authoritative Template Availability:
   - The controlled workflow directly receives and preserves the authoritative 16-section template.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd.agent import (
    ActionDecision,
    ActionType,
    BRDLeadAgent,
    GapResolutionAction,
    GapResolutionDecision,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext
from agents.brd.evaluation import EvaluationFinding, EvaluationOutcome, EvaluationResult, InformationStatus
from agents.brd.final_validation import FinalValidationOutcome, FinalValidationResult
from agents.brd.section_generation import SectionGenerationResult, SectionOperation
from agents.brd.section_validation import ValidationOutcome, ValidationResult
from agents.brd.state import BRDAgentState, SectionStatus
from agents.brd.template import extract_brd_sections, load_brd_template


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


def make_test_context(project_id: str = "proj-workflow-test", run_id: str = "run-test-123") -> AgentContext:
    return AgentContext(
        project_id=project_id,
        conversation_id="conv-workflow-test",
        metadata={"agent_run_id": run_id, "project_id": project_id},
    )


@pytest.mark.asyncio
async def test_normal_brd_message_reaches_controlled_workflow():
    """Test 1 — Verify that normal BRD message enters controlled workflow directly."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    with patch.object(agent, "stream_workflow_async") as mock_stream_wf:
        async def fake_stream_wf(*args, **kwargs):
            yield {"type": "progress", "phase": "1_INITIAL_CONTEXT", "message": "Starting BRD"}
            yield {"type": "content", "content": "Controlled BRD response"}

        mock_stream_wf.side_effect = fake_stream_wf

        events = []
        async for event in agent.stream_async("Generate BRD for Payment Gateway", context=ctx):
            events.append(event)

        mock_stream_wf.assert_called_once()
        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) == 1
        assert content_events[0]["content"] == "Controlled BRD response"


@pytest.mark.asyncio
async def test_no_workflow_flag_required_for_controlled_workflow():
    """Test 2 — Verify that the controlled workflow executes without run_workflow=true."""
    agent = BRDLeadAgent(model=MockChatModel())
    # Neither parameter nor metadata contains run_workflow
    ctx = AgentContext(project_id="proj-no-flag", conversation_id="conv-no-flag", metadata={})

    with patch.object(agent, "stream_workflow_async") as mock_stream_wf:
        async def fake_stream_wf(*args, **kwargs):
            yield {"type": "content", "content": "Controlled BRD executed"}

        mock_stream_wf.side_effect = fake_stream_wf

        events = []
        async for event in agent.stream_async("Draft requirements", context=ctx):
            events.append(event)

        mock_stream_wf.assert_called_once()
        assert "run_workflow" not in ctx.metadata


@pytest.mark.asyncio
async def test_generic_chat_path_unreachable_from_stream_async():
    """Test 3 — Verify generic DeepAgents chat fallback (_graph.astream) is unreachable."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()

    with patch.object(agent._graph, "astream") as mock_astream, \
         patch.object(agent, "stream_workflow_async") as mock_stream_wf:
        async def fake_stream_wf(*args, **kwargs):
            yield {"type": "content", "content": "Controlled workflow only"}

        mock_stream_wf.side_effect = fake_stream_wf

        events = []
        async for event in agent.stream_async("Hello there, conversational greeting", context=ctx):
            events.append(event)

        mock_astream.assert_not_called()
        mock_stream_wf.assert_called_once()


@pytest.mark.asyncio
async def test_followup_clarification_message_enters_controlled_workflow():
    """Test 4 — Verify that a follow-up/clarification message continues through the controlled workflow."""
    initial_state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
    initial_state.set_waiting_for_user("Please provide payment gateway details.")
    agent = BRDLeadAgent(model=MockChatModel(), state=initial_state)
    ctx = make_test_context()

    with patch.object(agent, "stream_workflow_async") as mock_stream_wf:
        async def fake_stream_wf(*args, **kwargs):
            yield {"type": "progress", "phase": "4_EVIDENCE_EVALUATION", "message": "Resuming"}
            yield {"type": "content", "content": "Resumed BRD"}

        mock_stream_wf.side_effect = fake_stream_wf

        events = []
        async for event in agent.stream_async("We use Stripe and PayPal.", context=ctx):
            events.append(event)

        mock_stream_wf.assert_called_once()
        assert any(e.get("content") == "Resumed BRD" for e in events)


def test_controlled_workflow_receives_authoritative_template_structure():
    """Test 5 — Verify the controlled workflow receives authoritative template from brd_template.md."""
    agent = BRDLeadAgent(model=MockChatModel())
    sections = agent.sections
    expected_sections = extract_brd_sections(load_brd_template())
    assert len(sections) == len(expected_sections)
    assert sections == expected_sections
    assert agent.state.template_sections == expected_sections


@pytest.mark.asyncio
async def test_ask_user_pauses_run_workflow_async():
    """Verify that ASK_USER halts run_workflow_async before Section Generation/Assembly."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()
    clarification_q = "What payment providers (e.g. Stripe, Adyen) must be integrated?"

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "perform_direct_work_async", new_callable=AsyncMock) as mock_work, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_interpret, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assemble:

        mock_decide.return_value = ActionDecision(
            action_type=ActionType.DIRECT_WORK,
            direct_work_objective="Analyze payment requirements",
            reasoning="Initial direct analysis",
        )
        mock_eval.return_value = EvaluationResult(
            outcome=EvaluationOutcome.INSUFFICIENT,
            summary="Payment provider specifications are unresolved.",
            unresolved_information=["payment providers"],
            findings=[
                EvaluationFinding(
                    observation="Missing specific payment providers",
                    status=InformationStatus.MISSING,
                )
            ],
        )
        mock_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.ASK_USER,
            clarification_question=clarification_q,
            reasoning="User must select target payment processors.",
            identified_gaps=["payment providers"],
        )

        response = await agent.run_workflow_async(
            objective="Develop Payment Gateway BRD",
            context=ctx,
        )

        # 1. Execution stopped: downstream phases were NOT executed
        mock_gen.assert_not_called()
        mock_assemble.assert_not_called()

        # 2. State marks waiting condition
        assert agent.state.is_waiting_for_user is True
        assert agent.state.pending_clarification == clarification_q
        assert agent.state.metadata.get("pending_clarification") == clarification_q

        # 3. User received the clarification question
        assert response.output_text == clarification_q


@pytest.mark.asyncio
async def test_ask_user_pauses_stream_workflow_async():
    """Verify that ASK_USER halts stream_workflow_async and yields clarification."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = make_test_context()
    clarification_q = "Do you require PCI-DSS Level 1 compliance or Level 2?"

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "perform_direct_work_async", new_callable=AsyncMock) as mock_work, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_interpret, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assemble:

        mock_decide.return_value = ActionDecision(
            action_type=ActionType.DIRECT_WORK,
            direct_work_objective="Analyze compliance requirements",
            reasoning="Initial compliance check",
        )
        mock_eval.return_value = EvaluationResult(
            outcome=EvaluationOutcome.INSUFFICIENT,
            summary="Compliance level needs user clarification.",
            unresolved_information=["PCI compliance level"],
            findings=[
                EvaluationFinding(
                    observation="Unresolved PCI compliance level",
                    status=InformationStatus.MISSING,
                )
            ],
        )
        mock_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.ASK_USER,
            clarification_question=clarification_q,
            reasoning="Compliance level requires stakeholder confirmation.",
            identified_gaps=["PCI compliance level"],
        )

        events = []
        async for event in agent.stream_workflow_async(
            request="Develop Payment Gateway BRD",
            context=ctx,
        ):
            events.append(event)

        # 1. Downstream phases were NOT executed
        mock_gen.assert_not_called()
        mock_assemble.assert_not_called()

        # 2. State marks waiting condition
        assert agent.state.is_waiting_for_user is True
        assert agent.state.pending_clarification == clarification_q

        # 3. Clarification question was yielded as content
        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) == 1
        assert content_events[0]["content"] == clarification_q


@pytest.mark.asyncio
async def test_user_clarification_resumes_workflow_to_completion():
    """Verify that providing clarification resumes the workflow at Phase 4 and completes the BRD."""
    initial_state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
    agent = BRDLeadAgent(model=MockChatModel(), state=initial_state)
    ctx = make_test_context()

    # Turn 1: Initial request encounters gap and pauses
    clarification_q = "Please confirm the SLA target (e.g. 99.9% or 99.99%)."
    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "perform_direct_work_async", new_callable=AsyncMock) as mock_work, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_interpret:

        mock_decide.return_value = ActionDecision(
            action_type=ActionType.DIRECT_WORK,
            direct_work_objective="Analyze SLA needs",
            reasoning="Direct analysis",
        )
        mock_eval.return_value = EvaluationResult(
            outcome=EvaluationOutcome.INSUFFICIENT,
            summary="SLA target is not specified.",
            unresolved_information=["SLA target"],
            findings=[
                EvaluationFinding(
                    observation="Missing SLA target",
                    status=InformationStatus.MISSING,
                )
            ],
        )
        mock_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.ASK_USER,
            clarification_question=clarification_q,
            reasoning="SLA target must be confirmed by user.",
            identified_gaps=["SLA target"],
        )

        resp1 = await agent.run_workflow_async(
            objective="Develop High-Availability Payment Gateway BRD",
            context=ctx,
        )
        assert resp1.output_text == clarification_q
        assert agent.state.is_waiting_for_user is True

    # Turn 2: User responds with clarification
    user_answer = "The SLA target is 99.99% with multi-region failover."

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide2, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval2, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_interpret2, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen2, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val2, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly2, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val2:

        # Phase 4 evaluation with clarification is now SUFFICIENT
        mock_eval2.return_value = EvaluationResult(
            outcome=EvaluationOutcome.SUFFICIENT,
            summary="All required evidence is sufficient following user clarification.",
            missing_information=[],
            unresolved_information=[],
            findings=[
                EvaluationFinding(
                    observation="SLA target confirmed",
                    status=InformationStatus.PRESENT,
                )
            ],
        )
        mock_interpret2.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
            reasoning="All required evidence is sufficient following user clarification.",
        )
        mock_gen2.return_value = SectionGenerationResult(
            section_name="1. Executive Summary",
            content="# 1. Executive Summary\nHigh-availability payment gateway with 99.99% SLA.",
            operation=SectionOperation.GENERATE,
        )
        mock_val2.return_value = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
        )
        mock_assembly2.return_value = BRDAssemblyResult(
            assembled_document="# Complete BRD\n\n## 1. Executive Summary\nHigh-availability payment gateway with 99.99% SLA.",
            sections_assembled=["1. Executive Summary"],
            section_count=1,
            assembly_complete=True,
        )
        mock_final_val2.return_value = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="All sections consistent and complete.",
        )

        # Execute resume through agent.execute_async (simulating incoming user message)
        # Note: controlled workflow is the sole execution path; state.is_waiting_for_user resumes seamlessly
        resp2 = await agent.execute_async(request=user_answer, context=ctx)

        # 1. Initial action decision was NOT re-executed
        mock_decide2.assert_not_called()

        # 2. Evidence contains user clarification
        clarification_evidence = [e for e in agent.state.evidence if e.get("source") == "user_clarification"]
        assert len(clarification_evidence) == 1
        assert clarification_evidence[0]["content"] == user_answer

        # 3. Waiting condition is cleared
        assert agent.state.is_waiting_for_user is False
        assert agent.state.pending_clarification is None

        # 4. Phase 4 re-evaluated and execution progressed through assembly and validation
        mock_eval2.assert_called_once()
        mock_gen2.assert_called_once()
        mock_assembly2.assert_called_once()
        mock_final_val2.assert_called_once()

        # 5. Full workflow succeeded
        assert resp2.success is True
        assert resp2.output_text == "BRD generation completed." or "# Complete BRD" in resp2.output_text


@pytest.mark.asyncio
async def test_stream_async_resumes_from_clarification_to_completion():
    """Verify that stream_async directly resumes workflow from clarification and yields final document."""
    initial_state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
    initial_state.set_waiting_for_user("Please confirm the SLA target (e.g. 99.9% or 99.99%).")
    agent = BRDLeadAgent(model=MockChatModel(), state=initial_state)
    ctx = make_test_context()

    user_answer = "The SLA target is 99.99% with multi-region failover."

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide2, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval2, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_interpret2, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen2, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val2, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly2, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val2:

        mock_eval2.return_value = EvaluationResult(
            outcome=EvaluationOutcome.SUFFICIENT,
            summary="All required evidence is sufficient following user clarification.",
            missing_information=[],
            unresolved_information=[],
            findings=[
                EvaluationFinding(
                    observation="SLA target confirmed",
                    status=InformationStatus.PRESENT,
                )
            ],
        )
        mock_interpret2.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
            reasoning="All required evidence is sufficient following user clarification.",
        )
        mock_gen2.return_value = SectionGenerationResult(
            section_name="1. Executive Summary",
            content="# 1. Executive Summary\nHigh-availability payment gateway with 99.99% SLA.",
            operation=SectionOperation.GENERATE,
        )
        mock_val2.return_value = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
        )
        mock_assembly2.return_value = BRDAssemblyResult(
            assembled_document="# Complete BRD\n\n## 1. Executive Summary\nHigh-availability payment gateway with 99.99% SLA.",
            sections_assembled=["1. Executive Summary"],
            section_count=1,
            assembly_complete=True,
        )
        mock_final_val2.return_value = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="All sections consistent and complete.",
        )

        events = []
        async for event in agent.stream_async(request=user_answer, context=ctx):
            events.append(event)

        mock_decide2.assert_not_called()
        mock_eval2.assert_called_once()
        mock_gen2.assert_called_once()
        mock_assembly2.assert_called_once()
        mock_final_val2.assert_called_once()

        assert agent.state.is_waiting_for_user is False
        assert agent.state.pending_clarification is None
        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) >= 1
        assert any("# Complete BRD" in e.get("content", "") or "completed" in e.get("content", "").lower() for e in content_events)
