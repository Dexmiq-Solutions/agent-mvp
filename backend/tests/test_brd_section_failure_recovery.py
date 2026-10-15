"""Focused tests for DEF-011: Differentiated Failure Handling After Section Rework Exhaustion.

Verifies:
1. Evidence Gap After Rework Exhaustion:
   - Pauses workflow with waiting_for_user == True and pending_clarification
   - Preserves completed sections, current section, drafts, and state
   - Assembly is strictly blocked
2. Clarification Question:
   - Specific to unresolved requirement/information gap, not generic
3. Resume Same Section:
   - Preserves original BRD objective
   - Resumes directly on the paused section
   - Bypasses unnecessary earlier phases
   - Clears clarification state
4. Successful Recovery After Clarification:
   - Re-evaluates/updates section with user clarification
   - Section validates, transitions to COMPLETED, and progression continues
5. Writing / Generation Quality Failure:
   - Fails explicitly with structured diagnostics
   - No user clarification prompt generated
   - Completed sections preserved, no assembly
6. Template / Configuration Failure:
   - Fails explicitly with template_configuration diagnostics
   - Never turned into a user clarification request
7. Infrastructure / System Failure:
   - Preserves system error semantics, never masked as evidence gap
8. Objective Preservation:
   - User clarification follow-up text never overwrites original BRD objective
9. Bounded Retry & No Infinite Loop:
   - Prevents infinite clarification/rework loops on persistent failure
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
    SectionFailureCategory,
    SectionReworkStrategy,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext, AgentRunRequest, AgentRunResponse
from agents.brd.section_generation import SectionGenerationResult, SectionOperation
from agents.brd.section_validation import (
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.final_validation import FinalValidationOutcome, FinalValidationResult
from agents.brd.state import BRDAgentState, BRDSectionStatus
from exceptions.retrieval import RetrievalError
from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Mock response"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def make_test_context(project_id: str = "proj-def-011") -> AgentContext:
    """Helper to generate tenant context."""
    return AgentContext(
        project_id=project_id,
        conversation_id="conv-def-011",
        project_name="Test Enterprise Solution",
        project_description="Test project documentation for DEF-011 verification",
        available_documents=["architecture.md", "business_rules.md"],
        metadata={"agent_run_id": "run-def-011"},
    )


# ---------------------------------------------------------------------------
# Test 1 — Evidence Gap After Rework Exhaustion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evidence_gap_after_rework_exhaustion_pauses_for_user_clarification():
    """When bounded rework fails due to missing evidence, workflow asks user for clarification and preserves state."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Executive Summary", "2. Business Rules", "3. Requirements"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "Enterprise Order Management System BRD"
    state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Executive Summary", "# Executive Summary\nOrder Management System overview.")
    state.set_current_section("2. Business Rules")

    evidence_gap_val = ValidationResult(
        section_name="2. Business Rules",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.GROUNDING,
                issue="Missing refund policy specification",
                explanation="Project documents do not contain cancellation refund rules",
                required_change="Specify refund policy after order cancellation",
            )
        ],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
        )
        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]\nPartial rules"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Business Rules", content="Draft v1 rules", operation=SectionOperation.GENERATE
        )
        mock_val.return_value = evidence_gap_val
        mock_update.return_value = SectionGenerationResult(
            section_name="2. Business Rules", content="Draft retry rules", operation=SectionOperation.UPDATE
        )

        response = await agent.run_workflow_async(context=ctx, initial_state=state)

        # 1. State must be waiting for user
        assert response.state.is_waiting_for_user is True
        assert bool(response.state.pending_clarification) is True
        assert "refund" in response.state.pending_clarification.lower()

        # 2. Section state and drafts preserved
        assert response.state.current_section == "2. Business Rules"
        assert response.state.get_section_status("1. Executive Summary") == BRDSectionStatus.COMPLETED
        assert response.state.get_section_content("1. Executive Summary") != ""
        assert response.state.get_section_status("2. Business Rules") == BRDSectionStatus.NEEDS_REVISION

        # 3. Serialization check: state can be persisted and restored
        state_dict = response.state.to_dict()
        restored_state = BRDAgentState.from_dict(state_dict)
        assert restored_state.is_waiting_for_user is True
        assert restored_state.current_section == "2. Business Rules"
        assert restored_state.get_section_status("1. Executive Summary") == BRDSectionStatus.COMPLETED

        # 4. Strict assembly invariant: no assembly called
        mock_assembly.assert_not_called()
        assert response.state.assembled_brd is None


# ---------------------------------------------------------------------------
# Test 2 — Clarification Question Quality
# ---------------------------------------------------------------------------

def test_clarification_question_is_specific_to_unresolved_requirement():
    """Clarification question communicates the exact missing requirement and is not a generic placeholder."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

    val_res = ValidationResult(
        section_name="Business Rules",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.GROUNDING,
                issue="Missing cancellation refund window",
                explanation="Project documents do not specify whether refunds are allowed after 30 days",
                required_change="Specify refund window and cancellation fee",
            )
        ],
    )

    q = agent.construct_section_clarification_question(
        section_name="Business Rules",
        validation_result=val_res,
        missing_items=["Missing cancellation refund window"],
    )

    assert "Business Rules" in q
    assert "cancellation refund" in q.lower() or "refund window" in q.lower()
    # Must not be a vague generic question
    assert q != "Can you provide more information?"
    assert "internal prompt" not in q.lower()


# ---------------------------------------------------------------------------
# Test 3 — Resume Same Section
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_same_section_preserves_objective_and_resumes_at_current_section():
    """Resuming workflow after user answers clarification stays on the same section and keeps original objective."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Executive Summary", "2. Business Rules", "3. Requirements"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    original_objective = "Enterprise Order Management System BRD"
    state.objective = original_objective
    state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Executive Summary", "# Executive Summary\nOverview text.")
    state.set_current_section("2. Business Rules")
    state.set_section_content("2. Business Rules", "# Business Rules\nDraft rules without refund policy.")
    state.set_waiting_for_user("Please specify the refund policy after cancellation.")
    state.metadata["pending_clarification_section"] = "2. Business Rules"

    user_clarification = "Cancellation within 14 days receives 100% refund, within 30 days receives 50% refund."

    valid_res = ValidationResult(
        section_name="2. Business Rules",
        outcome=ValidationOutcome.VALID,
        findings=[],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "evaluate_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Business Rules",
            content="# Business Rules\nUpdated with 14-day 100% refund policy.",
            operation=SectionOperation.UPDATE,
        )
        mock_val.return_value = valid_res
        mock_assembly.return_value = BRDAssemblyResult(
            assembled_document="# Complete Order Management BRD",
            sections_assembled=sections,
            section_count=len(sections),
            assembly_complete=True,
        )
        mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

        # Pass clarification via AgentRunRequest
        req = AgentRunRequest(input_text=user_clarification, context=ctx, state=state)
        response = await agent.run_workflow_async(req)

        # 1. Objective preservation check: must NOT be overwritten by user clarification answer
        assert response.state.objective == original_objective
        assert response.state.objective != user_clarification

        # 2. Earlier phases must be bypassed for section resume
        mock_decide.assert_not_called()
        mock_eval.assert_not_called()

        # 3. Clarification state cleared
        assert response.state.is_waiting_for_user is False
        assert response.state.pending_clarification is None

        # 4. Completed section 1 remained completed and was not re-generated
        assert response.state.get_section_status("1. Executive Summary") == BRDSectionStatus.COMPLETED

        # 5. Section 2 was updated and validated
        assert mock_val.await_count >= 1


# ---------------------------------------------------------------------------
# Test 4 — Successful Recovery After Clarification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_recovery_after_clarification_completes_section_and_progresses():
    """When clarification is provided, section is updated, validates, and marks COMPLETED."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Executive Summary", "2. Business Rules"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "Financial CRM BRD"
    state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Executive Summary", "# Executive Summary\nCRM details.")
    state.set_current_section("2. Business Rules")
    state.set_section_content("2. Business Rules", "Draft rules")
    state.set_waiting_for_user("Please provide refund policy.")
    state.metadata["pending_clarification_section"] = "2. Business Rules"

    user_answer = "Refund is 100% for 14 days."

    valid_res = ValidationResult(section_name="2. Business Rules", outcome=ValidationOutcome.VALID, findings=[])

    with patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Business Rules",
            content="# Business Rules\nFully validated rules with 14 days refund.",
            operation=SectionOperation.UPDATE,
        )
        mock_val.return_value = valid_res
        mock_assembly.return_value = BRDAssemblyResult(
            assembled_document="# Complete Financial CRM BRD",
            sections_assembled=sections,
            section_count=2,
            assembly_complete=True,
        )
        mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

        req = AgentRunRequest(input_text=user_answer, context=ctx, state=state)
        response = await agent.run_workflow_async(req)

        # Section 2 should now be completed
        assert response.state.get_section_status("2. Business Rules") == BRDSectionStatus.COMPLETED
        assert response.state.section_processing_complete is True
        mock_assembly.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 5 — Writing / Quality Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writing_failure_returns_explicit_failure_with_diagnostics():
    """Writing quality failure after bounded rework produces explicit failure with diagnostics and preserves state."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Executive Summary", "2. Architecture"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "Cloud Architecture BRD"
    state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Executive Summary", "# Summary\nValid executive summary.")
    state.set_current_section("2. Architecture")

    writing_invalid = ValidationResult(
        section_name="2. Architecture",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.SPECIFICITY,
                issue="Vague descriptions and informal tone",
                explanation="The section uses slang and does not provide concrete architectural component names",
                required_change="Rewrite in formal engineering tone with specific component identifiers",
            )
        ],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
        )
        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Architecture", content="Draft architecture", operation=SectionOperation.GENERATE
        )
        mock_val.return_value = writing_invalid
        mock_update.return_value = SectionGenerationResult(
            section_name="2. Architecture", content="Still vague retry", operation=SectionOperation.UPDATE
        )

        response = await agent.run_workflow_async(context=ctx, initial_state=state)

        # 1. Must NOT ask user for clarification
        assert response.state.is_waiting_for_user is False
        assert response.state.pending_clarification is None

        # 2. Must return explicit workflow failure
        assert response.success is False
        assert response.diagnostics is not None
        assert response.diagnostics["status"] == "failed"
        assert response.diagnostics["reason"] == "rework_exhausted"
        assert response.diagnostics["failure_category"] == SectionFailureCategory.GENERATION_QUALITY.value
        assert response.diagnostics["section"] == "2. Architecture"
        assert "1. Executive Summary" in response.diagnostics["completed_sections"]
        assert len(response.diagnostics["validation_findings"]) == 1

        # 3. State preserved
        assert response.state.get_section_status("1. Executive Summary") == BRDSectionStatus.COMPLETED
        assert response.state.get_section_status("2. Architecture") == BRDSectionStatus.NEEDS_REVISION
        assert response.state.metadata.get("failure_diagnostics") is not None

        # 4. Strict assembly invariant: no assembly allowed
        mock_assembly.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6 — Template / Configuration Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_template_configuration_failure_produces_explicit_failure_without_user_clarification():
    """Contradictory template constraints are classified as template_configuration and never ask user."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. System Overview"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "System BRD"

    template_invalid = ValidationResult(
        section_name="1. System Overview",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.TEMPLATE_COMPLIANCE,
                issue="Contradictory section requirements",
                explanation="Template mandates both excluding and including external third-party schemas",
                required_change="Resolve internal template contradiction",
            )
        ],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
        )
        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="1. System Overview", content="Overview draft", operation=SectionOperation.GENERATE
        )
        mock_val.return_value = template_invalid
        mock_update.return_value = SectionGenerationResult(
            section_name="1. System Overview", content="Retry draft", operation=SectionOperation.UPDATE
        )

        response = await agent.run_workflow_async(context=ctx, initial_state=state)

        assert response.state.is_waiting_for_user is False
        assert response.success is False
        assert response.diagnostics is not None
        assert response.diagnostics["failure_category"] == SectionFailureCategory.TEMPLATE_CONFIGURATION.value


# ---------------------------------------------------------------------------
# Test 7 — Infrastructure Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_infrastructure_failure_retains_system_error_semantics():
    """Infrastructure errors retain system error semantics and are never converted into evidence gaps."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[])  # No RAG capability
    ctx = make_test_context()

    state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])
    state.objective = "Project Infrastructure Test BRD"

    response = await agent.run_workflow_async(context=ctx, initial_state=state)

    assert response.success is False
    assert "RAG capability unavailable" in response.error
    assert response.state.is_waiting_for_user is False
    assert response.state.pending_clarification is None


# ---------------------------------------------------------------------------
# Test 8 — Objective Preservation Under User Clarification Follow-up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_objective_preservation_when_user_clarification_received():
    """Reproduces DEF-011 root cause: user's answer must never overwrite state.objective."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    original_objective = "Authoritative Multi-Tenant Billing Platform BRD"
    state = BRDAgentState.initialize_from_template(sections=["1. Scope", "2. Pricing Rules"])
    state.objective = original_objective
    state.update_section_status("1. Scope", BRDSectionStatus.COMPLETED)
    state.set_current_section("2. Pricing Rules")
    state.set_waiting_for_user("What are the subscription billing tiers?")
    state.metadata["pending_clarification_section"] = "2. Pricing Rules"

    user_reply = "Tier 1: Starter $10/mo, Tier 2: Enterprise $100/mo."

    with patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Pricing Rules", content="Pricing rules content", operation=SectionOperation.UPDATE
        )
        mock_val.return_value = ValidationResult(section_name="2. Pricing Rules", outcome=ValidationOutcome.VALID, findings=[])
        mock_assembly.return_value = BRDAssemblyResult(
            assembled_document="# Assembled", sections_assembled=["1. Scope", "2. Pricing Rules"], section_count=2, assembly_complete=True
        )
        mock_final_val.return_value = AsyncMock(is_valid=True)

        req = AgentRunRequest(input_text=user_reply, context=ctx, state=state)
        response = await agent.run_workflow_async(req)

        # Objective MUST be the original objective, NOT user_reply
        assert response.state.objective == original_objective
        assert response.state.objective != user_reply


# ---------------------------------------------------------------------------
# Test 9 — No Infinite Loop on Persistent Evidence Gap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarification_loop_is_bounded_and_does_not_loop_infinitely():
    """If user clarification is answered but section still fails validation, workflow terminates explicitly."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Compliance"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "HIPAA Compliance BRD"
    state.set_current_section("1. Compliance")
    # Simulate that 1 clarification cycle was already completed
    state.metadata["section_clarification_counts"] = {"1. Compliance": 1}
    state.set_waiting_for_user("Please provide audit log retention window.")
    state.metadata["pending_clarification_section"] = "1. Compliance"

    persistent_gap = ValidationResult(
        section_name="1. Compliance",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.GROUNDING,
                issue="Missing audit log retention duration",
                explanation="Retention duration still ungrounded",
                required_change="Provide exact retention duration",
            )
        ],
    )

    with patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update:

        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="1. Compliance", content="Updated draft", operation=SectionOperation.UPDATE
        )
        mock_val.return_value = persistent_gap
        mock_update.return_value = SectionGenerationResult(
            section_name="1. Compliance", content="Retry draft", operation=SectionOperation.UPDATE
        )

        user_answer = "Retention duration is 7 years."
        req = AgentRunRequest(input_text=user_answer, context=ctx, state=state)
        response = await agent.run_workflow_async(req)

        # Must NOT enter an infinite clarification loop
        assert response.state.is_waiting_for_user is False
        assert response.success is False
        assert response.diagnostics is not None
        assert response.diagnostics["status"] == "failed"
        assert response.diagnostics["reason"] == "rework_exhausted"
        assert response.diagnostics["failure_category"] == SectionFailureCategory.EVIDENCE_GAP.value


# ---------------------------------------------------------------------------
# Test 10 — Streaming: Evidence Gap Yields Clarification Question
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_workflow_evidence_gap_yields_clarification():
    """In streaming mode, rework exhaustion on evidence gap yields clarification question and pauses."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. Executive Summary", "2. Data Governance"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "Data Governance BRD"
    state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
    state.set_current_section("2. Data Governance")

    evidence_gap_val = ValidationResult(
        section_name="2. Data Governance",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.GROUNDING,
                issue="Missing data retention schedule",
                explanation="Project documents do not specify data retention period",
                required_change="Specify retention period in years",
            )
        ],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
        )
        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="2. Data Governance", content="Draft v1", operation=SectionOperation.GENERATE
        )
        mock_val.return_value = evidence_gap_val
        mock_update.return_value = SectionGenerationResult(
            section_name="2. Data Governance", content="Draft retry", operation=SectionOperation.UPDATE
        )

        events = []
        async for chunk in agent.stream_workflow_async(context=ctx, initial_state=state):
            events.append(chunk)

        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) >= 1
        last_content = content_events[-1]["content"]
        assert "Data Governance" in last_content
        assert "retention" in last_content.lower()

        # State must record waiting_for_user
        assert agent._state.is_waiting_for_user is True
        assert agent._state.current_section == "2. Data Governance"


# ---------------------------------------------------------------------------
# Test 11 — Streaming: Writing Failure Yields Diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_workflow_writing_failure_yields_diagnostics():
    """In streaming mode, rework exhaustion on writing failure yields diagnostic error and halts."""
    agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
    ctx = make_test_context()

    sections = ["1. System Boundaries"]
    state = BRDAgentState.initialize_from_template(sections=sections)
    state.objective = "System Boundaries BRD"

    writing_invalid = ValidationResult(
        section_name="1. System Boundaries",
        outcome=ValidationOutcome.NEEDS_REWORK,
        findings=[
            ValidationFinding(
                category=ValidationCategory.SPECIFICITY,
                issue="Ambiguous scope and boundary definition",
                explanation="Section fails to clearly define what is in-scope vs out-of-scope",
                required_change="Explicitly list in-scope and out-of-scope systems",
            )
        ],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
         patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
         patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update:

        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval_interpret.return_value = GapResolutionDecision(
            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
        )
        mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
        mock_gen.return_value = SectionGenerationResult(
            section_name="1. System Boundaries", content="Draft v1", operation=SectionOperation.GENERATE
        )
        mock_val.return_value = writing_invalid
        mock_update.return_value = SectionGenerationResult(
            section_name="1. System Boundaries", content="Draft retry", operation=SectionOperation.UPDATE
        )

        events = []
        async for chunk in agent.stream_workflow_async(context=ctx, initial_state=state):
            events.append(chunk)

        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) >= 1
        last_content = content_events[-1]["content"]
        assert "not all template sections could be completed" in last_content
        assert "generation quality" in last_content.lower()

        # State must not be waiting for user
        assert agent._state.is_waiting_for_user is False
        assert agent._state.metadata.get("failure_diagnostics") is not None
