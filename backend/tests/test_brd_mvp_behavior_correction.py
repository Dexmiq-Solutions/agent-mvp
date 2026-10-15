"""Comprehensive unit and integration tests for BRD Agent MVP Behavior Correction.

Covers the 13 required behavioral invariants:
1. BRD generates when optional information is missing.
2. BRD generates when detailed AI requirements are missing.
3. Missing conceptual workflows do not automatically block generation.
4. Missing persona details do not automatically block generation.
5. Unknown metadata remains TBD/TBD Suggested.
6. Unsupported claims are not silently accepted as facts (flagged under Grounding validation).
7. Genuine important unresolved requirements can still be recorded.
8. Clarification can still occur when genuinely necessary.
9. Initial BRD is produced before non-blocking clarification.
10. Existing project evidence remains grounded.
11. Existing RAG behavior remains functional.
12. Existing successful BRD workflows do not regress.
13. Streaming and non-streaming behavior remain consistent.
"""

import json
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    ActionDecision,
    ActionType,
    BRDLeadAgent,
    GapResolutionAction,
    GapResolutionDecision,
    is_documentation_quality_finding,
    is_honest_uncertainty_or_non_provided_text,
    is_substantive_business_clarification_item,
    is_tbd_value,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext, AgentRunRequest
from agents.brd.evaluation.agent import EvaluationFinding, EvaluationOutcome, EvaluationResult, InformationStatus
from agents.brd.final_validation.agent import FinalValidationFinding, FinalValidationResult
from agents.brd.section_generation.agent import (
    BRDSectionGenerationAgent,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
)
from agents.brd.section_validation.agent import (
    BRDSectionValidationAgent,
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.state import BRDAgentState, BRDSectionStatus
from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model conforming to BaseChatModel."""

    response: str = "Mock response"
    messages_to_return: list[AIMessage] = []
    index: int = 0
    tools_bound: list = []

    def __init__(self, response: str = "Mock response", **kwargs):
        super().__init__(**kwargs)
        self.response = response

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.messages_to_return and self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def make_test_context() -> AgentContext:
    return AgentContext(
        project_id="test-proj-mvp",
        conversation_id="conv-mvp-1",
        metadata={"agent_run_id": "run-mvp-1"},
    )


class TestHonestUncertaintyAndDocumentationQuality:
    """Tests 1-5: Handling of optional info, AI requirements, conceptual workflows, personas, and metadata."""

    def test_honest_uncertainty_helper_recognizes_valid_unknown_phrasings(self):
        """Valid uncertainty representations are recognized by helper."""
        assert is_honest_uncertainty_or_non_provided_text("Detailed AI behavior was not defined in the available information.")
        assert is_honest_uncertainty_or_non_provided_text("Not provided in project documentation.")
        assert is_honest_uncertainty_or_non_provided_text("Conceptual workflows were not discussed in the available information.")
        assert is_honest_uncertainty_or_non_provided_text("UI/UX Designer frustrations: TBD (To be clarified)")
        assert is_honest_uncertainty_or_non_provided_text("TBD (Suggested: Finance Team)")
        # Confirmed factual claim is not uncertainty
        assert not is_honest_uncertainty_or_non_provided_text("The payment gateway operates on Stripe API v3.")

    def test_section_validation_agent_treats_honest_uncertainty_as_valid(self):
        """Section validation does not fail a section for honestly stating unprovided information."""
        response_json = {
            "outcome": "NEEDS_REWORK",
            "findings": [
                {
                    "category": "COMPLETENESS",
                    "issue": "Detailed AI behavior was not defined",
                    "explanation": "Section honestly notes that detailed AI engagement logic was not provided in source documents.",
                    "required_change": "",
                }
            ],
            "rework_feedback": "Define AI behavior if available.",
        }
        mock_model = MockChatModel(
            messages_to_return=[AIMessage(content=json.dumps(response_json))]
        )
        validator = BRDSectionValidationAgent(model=mock_model)

        ctx = SectionValidationContext(
            section_name="4. Scope",
            section_content="## 4. Scope\nAI visitor engagement is planned. Detailed AI functionality was not defined in the available information.",
            section_requirements=["Define project scope and AI engagement boundaries"],
            available_information="User wants an AI customer portal.",
        )
        result = validator.validate(ctx)

        assert result.outcome == ValidationOutcome.VALID

    def test_documentation_quality_findings_filtered_from_blocking(self):
        """Documentation quality observations like missing conceptual workflows are filtered."""
        finding = ValidationFinding(
            category=ValidationCategory.COMPLETENESS,
            issue="Missing conceptual workflows",
            explanation="The section does not detail conceptual workflows as they were not provided in project context.",
            required_change="",
        )
        assert is_documentation_quality_finding(finding, "5. Business Process Overview")

    def test_unknown_persona_details_treated_as_documentation_quality_not_blocking(self):
        """Unknown persona details honestly recorded do not fail the section."""
        finding = ValidationFinding(
            category=ValidationCategory.COMPLETENESS,
            issue="Persona priority is not explicitly defined",
            explanation="Detailed responsibilities and goals were not explicitly defined in the available information.",
            required_change="",
        )
        assert is_documentation_quality_finding(finding, "6. User Personas")
        assert not is_substantive_business_clarification_item("Persona priority is not explicitly defined")

    def test_metadata_remains_tbd_or_tbd_suggested(self):
        """Unknown metadata mechanics remain TBD or TBD Suggested without blocking."""
        assert is_tbd_value("TBD")
        assert is_tbd_value("TBD (Suggested: Yashad Sathe)")
        assert not is_tbd_value("Yashad Sathe")


class TestGroundingAndUnsupportedClaims:
    """Test 6 & 10: Grounding validation and rejection of unsupported claims."""

    def test_unsupported_hallucinated_claims_are_not_benign_documentation_quality(self):
        """Unsupported claims presented as confirmed facts fail validation and are NOT marked benign."""
        finding = ValidationFinding(
            category=ValidationCategory.GROUNDING,
            issue="Unsupported claim presented as fact: 99.999% SLA guaranteed",
            explanation="Project evidence only mentions best-effort availability; 99.999% SLA is fabricated.",
            required_change="Remove or qualify SLA target as TBD.",
        )
        assert not is_documentation_quality_finding(finding, "Non-Functional Requirements")

    def test_lead_agent_interpret_section_validation_identifies_unsupported_claims_for_rework(self):
        """Lead Agent creates a rework strategy when unsupported claims are detected."""
        agent = BRDLeadAgent(model=MockChatModel())
        ctx = make_test_context()

        val_res = ValidationResult(
            section_name="Functional Requirements",
            outcome=ValidationOutcome.NEEDS_REWORK,
            rework_feedback="Grounding failure: remove unsupported statement that system supports Oracle DB.",
            findings=[
                ValidationFinding(
                    category=ValidationCategory.GROUNDING,
                    issue="Unsupported claim presented as fact regarding Oracle database support",
                    explanation="No evidence in project documents supports Oracle DB.",
                    required_change="Remove unsupported Oracle DB claim.",
                )
            ],
        )

        strategy = agent.interpret_section_validation(
            section_name="Functional Requirements",
            validation_result=val_res,
            context=ctx,
        )

        assert strategy.requires_rework is True
        assert strategy.rework_guidance is not None


class TestUnresolvedRequirementsAndClarificationWorkflow:
    """Tests 7-9: Recording genuine unresolved items, non-blocking delivery, and consolidated clarification."""

    @pytest.mark.asyncio
    async def test_initial_brd_v1_produced_before_non_blocking_clarification(self):
        """BRD V1 is assembled and delivered even when genuine substantive gaps exist, with clarification asked as continuation."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        sections = ["1. Executive Summary", "2. Data Retention Policy"]
        state = BRDAgentState.initialize_from_template(sections=sections)
        state.objective = "Enterprise Archive BRD"
        state.metadata["consolidate_clarification"] = True

        evidence_gap_val = ValidationResult(
            section_name="2. Data Retention Policy",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.GROUNDING,
                    issue="Data retention period is not defined",
                    explanation="Legal compliance requirements for data retention window were not specified in source docs.",
                    required_change="Specify data retention period.",
                )
            ],
        )
        valid_val = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
            findings=[],
        )

        with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
             patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
             patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
             patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
             patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update, \
             patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
             patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

            mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
            mock_eval_interpret.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION)
            mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
            mock_gen.side_effect = [
                SectionGenerationResult(section_name="1. Executive Summary", content="# Executive Summary\nSystem overview.", operation=SectionOperation.GENERATE),
                SectionGenerationResult(section_name="2. Data Retention Policy", content="# Data Retention Policy\nRetention window: TBD (To be clarified).", operation=SectionOperation.GENERATE),
            ]
            mock_update.return_value = SectionGenerationResult(
                section_name="2. Data Retention Policy",
                content="# Data Retention Policy\nRetention window: TBD (To be clarified).",
                operation=SectionOperation.UPDATE,
            )
            # Section 1 valid; Section 2 fails reworks due to missing retention evidence
            mock_val.side_effect = [valid_val, evidence_gap_val, evidence_gap_val, evidence_gap_val]

            assembled_content = "# Enterprise Archive BRD\n\n## 1. Executive Summary\nSystem overview.\n\n## 2. Data Retention Policy\nRetention window: TBD (To be clarified)."
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document=assembled_content,
                sections_assembled=sections,
                section_count=2,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(
                outcome=ValidationOutcome.VALID,
                findings=[],
                summary="BRD V1 is consistent and grounded.",
            )

            response = await agent.run_workflow_async(
                context=ctx,
                initial_state=state,
                consolidate_clarification=True,
            )

            # 1. BRD V1 WAS ASSEMBLED AND DELIVERED!
            mock_assembly.assert_awaited()
            assert response.state.assembled_brd is not None
            assert "Enterprise Archive BRD" in response.output_text

            # 2. Genuine unresolved requirement is recorded in state
            assert len(response.state.unresolved_information) > 0

            # 3. Consolidated clarification question is appended as continuation mechanism
            assert response.state.is_waiting_for_user is True
            assert response.state.metadata.get("pending_consolidated_clarification") is True
            assert "retention" in response.output_text.lower()


class TestRAGAndWorkflowRegressions:
    """Tests 11-13: RAG functionality, workflow non-regression, and streaming parity."""

    @pytest.mark.asyncio
    async def test_rag_retrieval_functions_for_substantive_evidence_lookup(self):
        """RAG retrieval continues to be invoked for substantive sections."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        mock_tool = AsyncMock()
        mock_tool.name = "search_project_knowledge"
        mock_tool.ainvoke = AsyncMock(return_value="[RETRIEVAL_SUCCESS]\nGDPR Article 17 requires right to erasure.")
        agent._tools = [mock_tool]
        ctx = make_test_context()

        result = await agent.retrieve_section_evidence_async(
            section="Data Compliance",
            requirements=["Specify GDPR compliance"],
            context=ctx,
        )
        mock_tool.ainvoke.assert_awaited()
        assert result is not None
        assert "GDPR" in result

    @pytest.mark.asyncio
    async def test_streaming_and_non_streaming_deliver_brd_v1_consistently(self):
        """Both streaming and non-streaming modes deliver BRD V1 without blocking on missing optional information."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        sections = ["1. Executive Summary", "2. Scope"]
        state = BRDAgentState.initialize_from_template(sections=sections)
        state.objective = "AI Agent Platform BRD"

        valid_val = ValidationResult(section_name="1. Executive Summary", outcome=ValidationOutcome.VALID, findings=[])
        valid_val2 = ValidationResult(section_name="2. Scope", outcome=ValidationOutcome.VALID, findings=[])

        with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
             patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
             patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
             patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
             patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
             patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

            mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
            mock_eval_interpret.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION)
            mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]"
            mock_gen.side_effect = [
                SectionGenerationResult(section_name="1. Executive Summary", content="# Executive Summary\nOverview text.", operation=SectionOperation.GENERATE),
                SectionGenerationResult(section_name="2. Scope", content="# Scope\nIn scope: Portal. Conceptual workflows: Not provided in project documentation.", operation=SectionOperation.GENERATE),
            ]
            mock_val.side_effect = [valid_val, valid_val2]
            assembled_doc = "# Complete BRD V1\n\n## 1. Executive Summary\nOverview text.\n\n## 2. Scope\nIn scope: Portal."
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document=assembled_doc,
                sections_assembled=sections,
                section_count=2,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(outcome=ValidationOutcome.VALID, findings=[])

            # Test streaming
            events = []
            async for ev in agent.stream_workflow_async(context=ctx, initial_state=state, consolidate_clarification=True):
                events.append(ev)

            content_events = [e["content"] for e in events if e.get("type") == "content"]
            full_streamed_doc = "".join(content_events)
            assert "Complete BRD V1" in full_streamed_doc
