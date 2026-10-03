"""Comprehensive tests for BRD Agent MVP Behavior Changes:
1. Administrative Metadata / TBD Handling:
   - Prepared By, Reviewed By, Approved By, Tech Lead legitimately default to TBD.
   - Weak unconfirmed clues become TBD (Suggested: Name).
   - Confirmed user statements become explicit values.
   - Metadata TBD does NOT block workflow or trigger evidence gaps.
   - Substantive project requirements remain strictly validated.
2. Conversation Evidence:
   - Current user prompt becomes authoritative grounding evidence.
   - Prior user conversation messages become authoritative grounding evidence.
   - Assistant messages are NEVER treated as authoritative evidence.
   - Explicit user statements ground generated sections.
3. Document Mechanics:
   - New BRD starts at Version 1.0.
   - Existing BRD update increments minor version (1.0 -> 1.1 -> 1.2).
   - Deterministic question IDs and runtime document dates are generated without RAG.
4. RAG Retrieval Policy:
   - Administrative sections (Header, Version History, Quality Gate) skip RAG.
   - Substantive sections execute RAG.
   - Identical queries during section rework are not repeatedly executed.
5. Consolidated Clarification Workflow:
   - Section with substantive business gap preserves draft, records gap, and continues drafting remaining sections.
   - Multiple unresolved items are consolidated into a grouped clarification question at the gate.
   - Answering consolidated clarification updates affected sections, marks them completed, and assembles the BRD.
"""
from datetime import datetime, timezone
import json
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
from agents.brd.context import AgentContext, AgentRunRequest
from agents.brd.evaluation import EvaluationFinding, EvaluationOutcome, EvaluationResult
from agents.brd.final_validation import FinalValidationOutcome, FinalValidationResult
from agents.brd.section_generation import SectionGenerationResult, SectionOperation
from agents.brd.section_validation import (
    BRDSectionValidationAgent,
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.state import BRDAgentState, BRDSectionStatus
from agents.brd.template import (
    classify_requirement_item,
    extract_brd_sections,
    is_administrative_section,
    is_benign_administrative_metadata_finding,
    is_metadata_or_role_item,
    is_substantive_requirement,
    is_tbd_value,
    load_brd_template,
)


from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model."""

    messages_to_return: list[AIMessage] = []
    index: int = 0

    def __init__(self, messages_to_return: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.messages_to_return = list(messages_to_return or [])
        self.index = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Mock LLM response"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def make_test_context(project_id="proj-mvp-test") -> AgentContext:
    return AgentContext(
        project_id=project_id,
        conversation_id="conv-mvp-test",
        user_id="user-123",
        project_name="Dexmiq MVP Platform",
        project_description="Enterprise payment and ordering platform",
        available_documents=["architecture.md", "specs.md"],
        metadata={"agent_run_id": "run-mvp-001"},
    )


# ===========================================================================
# 1. METADATA & TBD HANDLING TESTS
# ===========================================================================

class TestMetadataAndTBDHandling:
    """Tests verifying administrative metadata defaults to TBD without blocking."""

    def test_administrative_section_detection(self):
        """Administrative sections are correctly identified."""
        assert is_administrative_section("1. Dexmiq Standard Document Header") is True
        assert is_administrative_section("Document Header") is True
        assert is_administrative_section("Version History") is True
        assert is_administrative_section("Quality Gate & Completeness Snapshot") is True

        # Substantive sections must NOT be administrative
        assert is_administrative_section("2. Executive Summary") is False
        assert is_administrative_section("3. In-Scope Business Modules & Feature Groups") is False
        assert is_administrative_section("8. Technical Architecture & Constraints") is False

    def test_metadata_and_role_item_classification(self):
        """Administrative metadata and roles are distinguished from substantive requirements."""
        meta_items = [
            "Prepared By",
            "Reviewed By",
            "Approved By",
            "Tech Lead",
            "Stakeholder",
            "Author",
            "Approver",
            "Document ID",
            "Version Number",
            "Date Created",
        ]
        for item in meta_items:
            assert is_metadata_or_role_item(item) is True
            assert is_substantive_requirement(item) is False

        substantive_items = [
            "Order cancellation refund policy within 30 days",
            "PCI-DSS compliance for payment gateway integration",
            "99.9% uptime SLA target under peak load",
            "Idempotency key enforcement on payment requests",
            "Audit log retention window of 7 years",
        ]
        for item in substantive_items:
            assert is_substantive_requirement(item) is True
            assert is_metadata_or_role_item(item) is False

    def test_tbd_placeholder_detection(self):
        """TBD and suggested values are recognized as acceptable placeholders."""
        assert is_tbd_value("TBD") is True
        assert is_tbd_value("TBD (Suggested: Yashad Sathe)") is True
        assert is_tbd_value("Pending") is True
        assert is_tbd_value("To Be Determined") is True
        assert is_tbd_value("Unspecified") is True

        # Confirmed values are not TBD
        assert is_tbd_value("Yashad Sathe") is False
        assert is_tbd_value("Stripe") is False
        assert is_tbd_value("PostgreSQL") is False

    def test_benign_administrative_metadata_finding_filtering(self):
        """Validator correctly filters benign administrative metadata findings while keeping substantive findings."""
        # Benign: Prepared By is TBD in Document Header
        f_admin = ValidationFinding(
            category=ValidationCategory.GROUNDING,
            issue="Prepared By is TBD",
            explanation="Prepared By contains placeholder TBD which is ungrounded in project docs",
            required_change="Replace TBD with confirmed name",
        )
        assert is_benign_administrative_metadata_finding(f_admin, "1. Dexmiq Standard Document Header") is True

        # Benign: Tech Lead is TBD (Suggested: Yashad Sathe) in Stakeholders
        f_suggested = ValidationFinding(
            category=ValidationCategory.GROUNDING,
            issue="Tech Lead is unconfirmed placeholder",
            explanation="Tech Lead is marked as TBD (Suggested: Yashad Sathe)",
            required_change="Confirm Tech Lead identity",
        )
        assert is_benign_administrative_metadata_finding(f_suggested, "5. Stakeholders & Personas") is True

        # Substantive: Missing refund policy in Business Rules
        f_substantive = ValidationFinding(
            category=ValidationCategory.GROUNDING,
            issue="Missing refund policy",
            explanation="Refund policy is ungrounded in project documentation",
            required_change="Specify 30-day refund policy",
        )
        assert is_benign_administrative_metadata_finding(f_substantive, "6. Business Rules") is False

        # Substantive: Template compliance table violation
        f_table = ValidationFinding(
            category=ValidationCategory.TEMPLATE_COMPLIANCE,
            issue="Missing Stakeholder Matrix Table",
            explanation="Template requires a Markdown table",
            required_change="Format into Markdown table",
        )
        assert is_benign_administrative_metadata_finding(f_table, "5. Stakeholders & Personas") is False

    def test_section_validation_agent_accepts_tbd_administrative_metadata(self):
        """BRDSectionValidationAgent normalizes outcome to VALID when only administrative TBD findings exist."""
        response_json = {
            "outcome": "NEEDS_REWORK",
            "summary": "Prepared By and Reviewed By are TBD.",
            "findings": [
                {
                    "category": "Grounding",
                    "issue": "Prepared By is TBD",
                    "explanation": "Field contains unconfirmed TBD placeholder.",
                    "required_change": "Confirm author name.",
                },
                {
                    "category": "Grounding",
                    "issue": "Reviewed By is TBD",
                    "explanation": "Field contains unconfirmed TBD placeholder.",
                    "required_change": "Confirm reviewer name.",
                },
            ],
            "rework_feedback": "Confirm personnel names.",
        }
        mock_model = MockChatModel(
            messages_to_return=[AIMessage(content=json.dumps(response_json))]
        )
        validator = BRDSectionValidationAgent(model=mock_model)

        ctx = SectionValidationContext(
            section_name="1. Dexmiq Standard Document Header",
            section_content="# Document Header\n- **Prepared By:** TBD\n- **Reviewed By:** TBD",
            section_requirements=["Document Title", "Prepared By", "Reviewed By"],
        )
        result = validator.validate(ctx)

        # Must override outcome to VALID because all findings are benign administrative TBDs
        assert result.outcome == ValidationOutcome.VALID
        assert result.findings == []
        assert result.rework_feedback is None


# ===========================================================================
# 2. CONVERSATION EVIDENCE TESTS
# ===========================================================================

class TestConversationEvidence:
    """Tests verifying user conversation evidence is ingested and strictly authoritative."""

    def test_current_user_prompt_ingested_as_authoritative_evidence(self):
        """Current user input text is ingested with source='user_conversation', role='user', is_authoritative=True."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        user_prompt = "The Tech Lead is Yashad Sathe and we are using Stripe for payments."

        agent._ingest_user_conversation_evidence(input_prompt=user_prompt)

        evidence = agent.state.evidence
        assert len(evidence) >= 1
        user_ev = next(e for e in evidence if e.get("source") == "user_conversation")
        assert user_ev["role"] == "user"
        assert user_ev["is_authoritative"] is True
        assert "Yashad Sathe" in user_ev["content"]
        assert "Stripe" in user_ev["content"]

    def test_prior_user_conversation_ingested_and_assistant_messages_filtered(self):
        """Prior human messages become authoritative evidence, assistant messages are NEVER authoritative."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

        prior_messages = [
            HumanMessage(content="We need multi-currency support for EUR, USD, and GBP."),
            AIMessage(content="I suggest we also add support for JPY and CAD."),  # Assistant guess
            {"role": "user", "content": "The SLA target is 99.95% availability."},
            {"role": "assistant", "content": "I assume the database is MongoDB."},  # Assistant guess
        ]

        agent._ingest_user_conversation_evidence(input_prompt=None, prior_messages=prior_messages)

        evidence = agent.state.evidence
        user_items = [e for e in evidence if e.get("source") == "user_conversation"]

        # Only the 2 user messages should be in authoritative evidence
        assert len(user_items) == 2
        all_content = " ".join(e["content"] for e in user_items)
        assert "multi-currency" in all_content
        assert "99.95%" in all_content

        # Assistant guesses must NEVER be in evidence
        assert "JPY" not in all_content
        assert "MongoDB" not in all_content

    def test_duplicate_user_evidence_is_deduplicated(self):
        """Identical user conversation evidence statements are deduplicated in state."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

        agent._ingest_user_conversation_evidence(
            input_prompt="The Tech Lead is Yashad Sathe.",
            prior_messages=[HumanMessage(content="The Tech Lead is Yashad Sathe.")],
        )

        user_items = [e for e in agent.state.evidence if e.get("source") == "user_conversation"]
        assert len(user_items) == 1


# ===========================================================================
# 3. DOCUMENT MECHANICS TESTS
# ===========================================================================

class TestDocumentMechanics:
    """Tests verifying deterministic document mechanics generation."""

    def test_new_brd_version_defaults_to_1_0(self):
        """New BRD begins at Version 1.0."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        version = agent.resolve_document_version()
        assert version == "1.0"
        assert agent.state.metadata.get("document_version") == "1.0"

    def test_existing_brd_update_increments_minor_version(self):
        """Updating an existing BRD increments the minor version (1.0 -> 1.1 -> 1.2)."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

        # Simulate existing BRD with Version 1.0
        agent.state.metadata["document_version"] = "1.0"
        v1 = agent.resolve_document_version()
        assert v1 == "1.1"

        # Update again: 1.1 -> 1.2
        v2 = agent.resolve_document_version()
        assert v2 == "1.2"

    def test_generation_context_injects_document_mechanics(self):
        """Section generation context receives deterministic mechanics (version, date, question id prefix)."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        with patch.object(agent.section_generator, "generate", new_callable=MagicMock) as mock_gen:
            mock_gen.return_value = SectionGenerationResult(
                section_name="1. Dexmiq Standard Document Header",
                content="# Header\nVersion: 1.0",
                operation=SectionOperation.GENERATE,
            )
            agent.generate_section(
                section="1. Dexmiq Standard Document Header",
                context=ctx,
            )

            mock_gen.assert_called_once()
            called_ctx = mock_gen.call_args[0][0]
            assert called_ctx.metadata.get("document_version") == "1.0"
            assert "document_date" in called_ctx.metadata
            assert called_ctx.metadata.get("question_id_prefix") == "Q-BRD-"


# ===========================================================================
# 4. RAG RETRIEVAL POLICY TESTS
# ===========================================================================

class TestRAGRetrievalPolicy:
    """Tests verifying RAG is skipped on administrative sections and deduplicated during rework."""

    @pytest.mark.asyncio
    async def test_administrative_sections_skip_rag_retrieval(self):
        """Administrative sections skip RAG calls without invoking the RAG tool."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        mock_tool = AsyncMock()
        mock_tool.name = "search_project_knowledge"
        agent._tools = [mock_tool]
        ctx = make_test_context()

        # Header
        res_header = await agent.retrieve_section_evidence_async(
            section="1. Dexmiq Standard Document Header",
            context=ctx,
        )
        assert res_header is None
        mock_tool.ainvoke.assert_not_called()

        # Version History
        res_ver = await agent.retrieve_section_evidence_async(
            section="Version History",
            context=ctx,
        )
        assert res_ver is None
        mock_tool.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_substantive_sections_execute_rag_retrieval(self):
        """Substantive sections invoke RAG and record authoritative evidence."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        mock_tool = AsyncMock()
        mock_tool.name = "search_project_knowledge"
        mock_tool.ainvoke = AsyncMock(return_value="[RETRIEVAL_SUCCESS]\nRelevant project facts.")
        agent._tools = [mock_tool]
        ctx = make_test_context()

        res = await agent.retrieve_section_evidence_async(
            section="6. Business Rules",
            query="Order cancellation rules",
            context=ctx,
        )
        assert res is not None
        assert "[RETRIEVAL_SUCCESS]" in res
        mock_tool.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_rag_query_during_rework_is_not_repeated(self):
        """Executing identical retrieval query repeatedly during rework returns cached result."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        mock_tool = AsyncMock()
        mock_tool.name = "search_project_knowledge"
        mock_tool.ainvoke = AsyncMock(return_value="[RETRIEVAL_SUCCESS]\nRelevant project facts.")
        agent._tools = [mock_tool]
        ctx = make_test_context()

        # First call executes
        res1 = await agent.retrieve_section_evidence_async(
            section="6. Business Rules",
            query="Exact query string",
            context=ctx,
        )
        assert mock_tool.ainvoke.call_count == 1

        # Second identical call skips RAG tool and uses cached result
        res2 = await agent.retrieve_section_evidence_async(
            section="6. Business Rules",
            query="Exact query string",
            context=ctx,
        )
        assert mock_tool.ainvoke.call_count == 1
        assert res1 == res2


# ===========================================================================
# 5. CONSOLIDATED CLARIFICATION WORKFLOW TESTS
# ===========================================================================

class TestConsolidatedClarificationWorkflow:
    """Tests verifying the continuous drafting pass and consolidated clarification behavior."""

    @pytest.mark.asyncio
    async def test_continuous_drafting_defers_clarification_to_consolidation_gate(self):
        """When consolidate_clarification=True, business gap does not stop workflow immediately; continues drafting and asks consolidated question at end."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        sections = ["1. Executive Summary", "2. Payment Rules", "3. Requirements"]
        state = BRDAgentState.initialize_from_template(sections=sections)
        state.objective = "Payment Gateway BRD"
        state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
        state.set_section_content("1. Executive Summary", "# Summary\nPayment gateway overview.")
        state.set_current_section("2. Payment Rules")

        gap_val = ValidationResult(
            section_name="2. Payment Rules",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.GROUNDING,
                    issue="Missing refund policy specification",
                    explanation="Project documents do not contain cancellation refund rules",
                    required_change="Specify refund policy window in days",
                )
            ],
        )
        valid_val = ValidationResult(
            section_name="3. Requirements",
            outcome=ValidationOutcome.VALID,
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

            # Gen returns draft for section 2, then section 3
            mock_gen.side_effect = [
                SectionGenerationResult(section_name="2. Payment Rules", content="Draft rules v1", operation=SectionOperation.GENERATE),
                SectionGenerationResult(section_name="3. Requirements", content="Draft requirements v1", operation=SectionOperation.GENERATE),
            ]
            # Val returns gap on section 2 (initial + 2 rework attempts = 3 calls), then valid on section 3
            mock_val.side_effect = [gap_val, gap_val, gap_val, valid_val]
            mock_update.return_value = SectionGenerationResult(
                section_name="2. Payment Rules", content="Draft rules v2", operation=SectionOperation.UPDATE
            )

            # Run with consolidate_clarification=True
            response = await agent.run_workflow_async(
                context=ctx,
                initial_state=state,
                consolidate_clarification=True,
            )

            # 1. Workflow completed drafting pass for remaining sections
            assert mock_gen.call_count == 2  # Both Section 2 and Section 3 were drafted!

            # 2. Pauses at consolidation gate
            assert response.state.is_waiting_for_user is True
            assert response.state.metadata.get("pending_consolidated_clarification") is True

            # 3. Consolidated clarification question formulated
            question = response.output_text
            assert "Before I finalize the BRD, I need clarification on" in question
            assert "refund policy" in question.lower()

            # 4. Strict assembly invariant: no assembly before clarification answered
            mock_assembly.assert_not_called()
            assert response.state.assembled_brd is None

    @pytest.mark.asyncio
    async def test_user_clarification_response_resumes_updates_affected_sections_and_assembles(self):
        """When user provides answers to consolidated clarification, affected sections are updated and document assembles."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        sections = ["1. Executive Summary", "2. Payment Rules"]
        state = BRDAgentState.initialize_from_template(sections=sections)
        state.objective = "Payment Gateway BRD"
        state.set_section_content("1. Executive Summary", "# Summary\nOverview.")
        state.update_section_status("1. Executive Summary", BRDSectionStatus.COMPLETED)
        state.set_section_content("2. Payment Rules", "# Payment Rules\nDraft content.")
        state.update_section_status("2. Payment Rules", BRDSectionStatus.NEEDS_REVISION)

        # Simulate workflow paused at consolidation gate with unresolved gap on Payment Rules
        state.set_waiting_for_user("Please clarify refund policy window.")
        state.metadata["pending_consolidated_clarification"] = True
        state.metadata["unresolved_section_gaps"] = {"2. Payment Rules": ["Missing refund policy specification"]}
        state.add_unresolved("Missing refund policy specification")

        with patch.object(agent, "update_section_async", new_callable=AsyncMock) as mock_update, \
             patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
             patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
             patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

            mock_update.return_value = SectionGenerationResult(
                section_name="2. Payment Rules",
                content="# Payment Rules\nUpdated with 30-day refund window.",
                operation=SectionOperation.UPDATE,
            )
            mock_val.return_value = ValidationResult(section_name="2. Payment Rules", outcome=ValidationOutcome.VALID)
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document="# Complete Assembled Payment BRD",
                sections_assembled=["1. Executive Summary", "2. Payment Rules"],
                section_count=2,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

            # User answers the clarification
            user_answer = "The refund window is strictly 30 days from transaction date."
            req = AgentRunRequest(input_text=user_answer, context=ctx, state=state)
            response = await agent.run_workflow_async(req, consolidate_clarification=True)

            # 1. State cleared waiting for user
            assert response.state.is_waiting_for_user is False
            assert len(response.state.unresolved_information) == 0

            # 2. Affected section updated and marked COMPLETED
            mock_update.assert_awaited_once()
            assert response.state.get_section_status("2. Payment Rules") == BRDSectionStatus.COMPLETED

            # 3. Document assembled and validated
            mock_assembly.assert_awaited_once()
            assert response.success is True
            assert response.state.assembled_brd == "# Complete Assembled Payment BRD"
