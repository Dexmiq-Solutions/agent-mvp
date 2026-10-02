"""Comprehensive tests for Batch 2 Remediation: DEF-008, DEF-009, DEF-010, DEF-013.

Covers:
- DEF-008: Section-Specific RAG Retrieval
  1. Section-specific retrieval query construction using section name and requirements.
  2. Project isolation enforcement during section retrieval.
  3. Different sections yield different retrieval queries.
  4. Workflow invokes section retrieval prior to drafting each section.

- DEF-009: Separation of Project Evidence from Model-Generated Work
  5. RAG-retrieved evidence is stored in state.evidence.
  6. Direct work is stored in state.agent_work, not state.evidence.
  7. Direct work objects passed to add_evidence are diverted to agent_work.
  8. Generated section content is not stored in state.evidence.
  9. Evidence evaluation receives only authoritative project evidence.
  10. Section validation grounding evaluates only against authoritative project evidence.

- DEF-010: Section Rework Retrieves Additional Evidence
  11. Validation findings distinguish writing problems (Type A) from evidence problems (Type B).
  12. Evidence-related validation failures trigger targeted RAG retrieval.
  13. Newly retrieved evidence reaches the subsequent update_section_async attempt.
  14. Writing-only validation failures do not trigger RAG retrieval.
  15. Bounded rework loop prevents infinite regeneration.

- DEF-013: Explicit Handling of RAG Initialization & Availability Failures
  16. get_brd_lead_agent raises RetrievalError when RAG service initialization fails.
  17. Workflow halts explicitly when RAG capability is missing instead of generating ungrounded content.
  18. RAG-unavailable infrastructure failure is clearly distinguishable from "no relevant evidence found".
"""

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
    GapResolutionAction,
    GapResolutionDecision,
    SectionReworkStrategy,
    WorkflowDecision,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import ActionSource, AgentContext
from agents.brd.evaluation import (
    EvaluationFinding,
    EvaluationOutcome,
    EvaluationResult,
    InformationStatus,
)
from agents.brd.final_validation import (
    FinalValidationCategory,
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
from exceptions.retrieval import RetrievalError
from services.rag_service import RAGService, RetrievalResult, RetrievedChunk
from tools.rag import search_project_knowledge


# ---------------------------------------------------------------------------
# Mocks & Test Fixtures
# ---------------------------------------------------------------------------

class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing."""

    messages_to_return: list[AIMessage] = []
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default mock fallback"))]
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


def _create_mock_retrieval_result(
    project_id: str,
    query: str,
    chunks: list[dict] | None = None,
) -> MagicMock:
    """Helper creating a mock RetrievalResult satisfying the RAG tool contract."""
    mock_result = MagicMock(spec=RetrievalResult)
    mock_result.project_id = project_id
    mock_result.original_query = query
    mock_result.retrieval_query = query

    if chunks:
        mock_chunks = []
        formatted_parts = []
        for i, c in enumerate(chunks, start=1):
            chunk_obj = RetrievedChunk(
                chunk_id=c.get("chunk_id", f"chunk-{i}"),
                document_id=c.get("document_id", f"doc-{i}"),
                project_id=project_id,
                content=c.get("content", f"Content {i}"),
                score=c.get("score", 0.9),
                rank=i,
                metadata=c.get("metadata", {}),
            )
            mock_chunks.append(chunk_obj)
            formatted_parts.append(
                f"[Source: {chunk_obj.document_id} | Score: {chunk_obj.score:.2f}]\n{chunk_obj.content}"
            )
        mock_result.chunks = tuple(mock_chunks)
        mock_result.is_empty = False
        mock_result.formatted_text = f"[RETRIEVAL_SUCCESS]\n\n" + "\n\n".join(formatted_parts)
    else:
        mock_result.chunks = ()
        mock_result.is_empty = True
        mock_result.formatted_text = "[NO_EVIDENCE] No relevant documents found."

    return mock_result


def make_test_context(project_id: str = "proj-test-123", run_id: str = "run-trace-456") -> AgentContext:
    return AgentContext(
        project_id=project_id,
        conversation_id="conv-test-789",
        metadata={"agent_run_id": run_id, "project_id": project_id},
    )


# ---------------------------------------------------------------------------
# DEF-008: Section-Specific RAG Retrieval Tests
# ---------------------------------------------------------------------------

class TestSectionSpecificRetrieval:
    """Tests for DEF-008 verifying that retrieval is section-oriented."""

    def test_construct_section_retrieval_query_uses_section_name_and_requirements(self):
        """Query construction must incorporate section title and specific template requirements."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

        sec1 = "1. Executive Summary"
        reqs1 = ["High-level business context", "Core value proposition", "Target audience"]
        query1 = agent.construct_section_retrieval_query(sec1, reqs1)

        assert "1. Executive Summary" in query1
        assert "High-level business context" in query1
        assert "Target audience" in query1

        sec4 = "4. Functional Requirements"
        reqs4 = ["User authentication workflows", "Role-based access control", "Audit logs"]
        query4 = agent.construct_section_retrieval_query(sec4, reqs4)

        assert "4. Functional Requirements" in query4
        assert "Role-based access control" in query4
        assert query1 != query4

    def test_different_sections_produce_different_queries(self):
        """Different BRD sections must produce distinctly targeted queries."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])

        query_scope = agent.construct_section_retrieval_query("2. Project Scope", ["In-scope deliverables", "Out-of-scope items"])
        query_tech = agent.construct_section_retrieval_query("6. Technical Architecture", ["API specs", "Database schema", "Latency requirements"])

        assert "Project Scope" in query_scope
        assert "Technical Architecture" in query_tech
        assert query_scope != query_tech

    @pytest.mark.asyncio
    async def test_retrieve_section_evidence_enforces_project_isolation(self):
        """Section retrieval must strictly enforce project isolation via AgentContext.project_id."""
        mock_service = MagicMock(spec=RAGService)
        mock_res = _create_mock_retrieval_result(
            project_id="isolated-project-999",
            query="3. Business Objectives",
            chunks=[{"chunk_id": "c1", "content": "ROI goals at 15%", "score": 0.95}],
        )
        mock_service.retrieve = AsyncMock(return_value=mock_res)

        agent = BRDLeadAgent(model=MockChatModel(), rag_service=mock_service)
        ctx = make_test_context(project_id="isolated-project-999")

        result = await agent.retrieve_section_evidence_async(
            section="3. Business Objectives",
            requirements=["ROI goals", "Quarterly milestones"],
            context=ctx,
        )

        assert result is not None
        assert len(agent._state.evidence) == 1
        assert agent._state.evidence[0]["section"] == "3. Business Objectives"
        assert agent._state.evidence[0]["source"] == ActionSource.RAG.value

        # Verify project_id passed to RAGService
        mock_service.retrieve.assert_awaited_once()
        call_kwargs = mock_service.retrieve.call_args.kwargs
        assert call_kwargs["project_id"] == "isolated-project-999"
        assert "3. Business Objectives" in call_kwargs["query"]

    @pytest.mark.asyncio
    async def test_workflow_executes_section_specific_retrieval_before_each_section(self):
        """Workflow Phase 5 must trigger retrieve_section_evidence_async for each section."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        # Set up a 2-section state
        state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary", "2. Scope"])

        with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
             patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval_interpret, \
             patch.object(agent, "retrieve_section_evidence_async", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(agent, "generate_section_async", new_callable=AsyncMock) as mock_gen, \
             patch.object(agent, "validate_section_async", new_callable=AsyncMock) as mock_val, \
             patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_assembly, \
             patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_final_val:

            mock_decide.return_value = ActionDecision(
                action_type=ActionType.DIRECT_WORK,
                direct_work_objective="Analyze objective",
            )
            mock_eval_interpret.return_value = GapResolutionDecision(
                action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
                reasoning="Information sufficient",
            )
            mock_retrieve.side_effect = [
                "[RETRIEVAL_SUCCESS]\nSection 1 evidence",
                "[RETRIEVAL_SUCCESS]\nSection 2 evidence",
            ]
            mock_gen.side_effect = [
                SectionGenerationResult(section_name="1. Executive Summary", content="Exec summary content", operation=SectionOperation.GENERATE),
                SectionGenerationResult(section_name="2. Scope", content="Scope content", operation=SectionOperation.GENERATE),
            ]
            mock_val.side_effect = [
                ValidationResult(section_name="1. Executive Summary", outcome=ValidationOutcome.VALID),
                ValidationResult(section_name="2. Scope", outcome=ValidationOutcome.VALID),
            ]
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document="# Complete BRD",
                sections_assembled=["1. Executive Summary", "2. Scope"],
                section_count=2,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

            response = await agent.run_workflow_async(context=ctx, initial_state=state)

            assert response.success is True
            assert mock_retrieve.await_count == 2
            calls = mock_retrieve.await_args_list
            assert calls[0].kwargs["section"] == "1. Executive Summary"
            assert calls[1].kwargs["section"] == "2. Scope"


# ---------------------------------------------------------------------------
# DEF-009: Separation of Evidence from Model-Generated Work Tests
# ---------------------------------------------------------------------------

class TestEvidenceVsAgentWorkBoundary:
    """Tests for DEF-009 ensuring agent work does not contaminate project evidence."""

    def test_rag_evidence_preserved_in_state_evidence(self):
        """Retrieved project evidence is stored strictly in state.evidence."""
        state = BRDAgentState()
        rag_item = {
            "source": ActionSource.RAG.value,
            "content": "Project uses PostgreSQL and FastAPI",
            "document_id": "doc-001",
        }
        state.add_evidence(rag_item)

        assert len(state.evidence) == 1
        assert len(state.agent_work) == 0
        assert state.get_project_evidence() == [rag_item]

    def test_direct_work_stored_in_agent_work_not_evidence(self):
        """Direct work analysis is stored in state.agent_work and never in state.evidence."""
        state = BRDAgentState()
        agent = BRDLeadAgent(model=MockChatModel())
        agent._state = state

        work_item = {
            "source": ActionSource.DIRECT_WORK.value,
            "objective": "Synthesize business goals",
            "result": "The agent analyzed the goals and believes...",
        }
        state.add_agent_work(work_item)

        assert len(state.evidence) == 0
        assert len(state.agent_work) == 1
        assert state.agent_work[0]["objective"] == "Synthesize business goals"
        assert state.get_project_evidence() == []
        assert len(state.get_agent_work()) == 1

    def test_add_evidence_diverts_direct_work_to_agent_work(self):
        """Calling add_evidence with direct_work must automatically route it to agent_work."""
        state = BRDAgentState()
        state.add_evidence({"source": "direct_work", "content": "Agent reasoning text"})

        assert len(state.evidence) == 0
        assert len(state.agent_work) == 1
        assert state.agent_work[0]["content"] == "Agent reasoning text"

    @pytest.mark.asyncio
    async def test_generated_section_content_not_stored_in_evidence(self):
        """Generated section content goes to state.sections and does NOT enter state.evidence."""
        agent = BRDLeadAgent(model=MockChatModel())
        agent._state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])

        with patch.object(agent.section_generator, "generate_async", new_callable=AsyncMock) as mock_sub_gen:
            mock_sub_gen.return_value = SectionGenerationResult(
                section_name="1. Executive Summary",
                content="# 1. Executive Summary\nDrafted by model.",
                operation=SectionOperation.GENERATE,
            )

            res = await agent.generate_section_async("1. Executive Summary")

            assert res.content == "# 1. Executive Summary\nDrafted by model."
            # Evidence must remain empty
            assert len(agent._state.evidence) == 0
            assert len(agent._state.get_project_evidence()) == 0

    @pytest.mark.asyncio
    async def test_section_validation_evaluates_only_against_project_evidence(self):
        """validate_section_async must pass only project evidence as available_information, not agent work."""
        agent = BRDLeadAgent(model=MockChatModel())
        agent._state = BRDAgentState(
            evidence=[{"source": ActionSource.RAG.value, "content": "Verified project requirement: OAuth2"}],
            agent_work=[{"source": "direct_work", "content": "Model thought: Maybe use SAML too"}],
        )

        with patch.object(agent.section_validator, "validate_async", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = ValidationResult(
                section_name="4. Security",
                outcome=ValidationOutcome.VALID,
            )

            await agent.validate_section_async("4. Security", "Section content text")

            mock_val.assert_awaited_once()
            val_ctx = mock_val.call_args[0][0]
            avail_info = val_ctx.available_information

            # Must contain verified RAG evidence
            assert any("OAuth2" in str(x) for x in avail_info)
            # Must NOT contain model agent work
            assert not any("SAML" in str(x) for x in avail_info)


# ---------------------------------------------------------------------------
# DEF-010: Section Rework Retrieves Additional Evidence Tests
# ---------------------------------------------------------------------------

class TestSectionReworkEvidenceRetrieval:
    """Tests for DEF-010 ensuring rework can retrieve missing project knowledge."""

    def test_rework_strategy_distinguishes_writing_vs_evidence_problem(self):
        """interpret_section_validation flags requires_retrieval=True only for evidence/knowledge gaps."""
        agent = BRDLeadAgent(model=MockChatModel())

        # Case A: Writing problem (e.g., formatting / template compliance)
        val_writing = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.TEMPLATE_COMPLIANCE,
                    issue="Missing markdown headers required by template",
                    explanation="Template requires level 2 headers",
                    required_change="Add required # headers",
                )
            ],
            rework_feedback="Add required # headers",
        )
        strategy_a = agent.interpret_section_validation(
            validation_result=val_writing,
            section_name="1. Executive Summary",
            rework_attempt=1,
        )
        assert strategy_a.requires_retrieval is False
        assert strategy_a.retrieval_query is None

        # Case B: Evidence problem (Grounding / Completeness / Missing information)
        val_evidence = ValidationResult(
            section_name="4. Functional Requirements",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.GROUNDING,
                    issue="Missing details on payment gateway webhook failure handling",
                    explanation="Payment webhook failure handling not found in project knowledge",
                    required_change="Retrieve payment gateway specification",
                )
            ],
            rework_feedback="Retrieve payment gateway specification",
        )
        strategy_b = agent.interpret_section_validation(
            validation_result=val_evidence,
            section_name="4. Functional Requirements",
            rework_attempt=1,
        )
        assert strategy_b.requires_retrieval is True
        assert strategy_b.retrieval_query is not None
        assert "payment gateway webhook" in strategy_b.retrieval_query

    @pytest.mark.asyncio
    async def test_evidence_rework_triggers_rag_retrieval_and_updates_section(self):
        """When rework requires retrieval, the workflow calls RAG and supplies new evidence to update_section."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])

        val_invalid_evidence = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.COMPLETENESS,
                    issue="Missing stakeholder sign-off criteria",
                    explanation="Stakeholder sign-off criteria must be included",
                    required_change="Add stakeholder sign-off criteria",
                )
            ],
        )
        val_valid = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
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
            mock_eval_interpret.return_value = GapResolutionDecision(
                action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
            )
            # Initial section retrieval returns initial evidence
            mock_retrieve.side_effect = [
                "[RETRIEVAL_SUCCESS]\nInitial evidence",
                # Rework retrieval returns newly retrieved stakeholder info
                "[RETRIEVAL_SUCCESS]\nSign-off criteria: VP Product and Lead Architect",
            ]
            mock_gen.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Draft v1", operation=SectionOperation.GENERATE
            )
            mock_val.side_effect = [val_invalid_evidence, val_valid]
            mock_update.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Draft v2 with sign-off criteria", operation=SectionOperation.UPDATE
            )
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document="# Complete BRD",
                sections_assembled=["1. Executive Summary"],
                section_count=1,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

            response = await agent.run_workflow_async(context=ctx, initial_state=state)

            assert response.success is True
            # retrieve_section_evidence_async was called TWICE: once initially, once on rework
            assert mock_retrieve.await_count == 2
            mock_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_writing_only_rework_does_not_trigger_rag(self):
        """Writing-only validation failures do not trigger unnecessary RAG retrieval."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])

        val_invalid_writing = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.TEMPLATE_COMPLIANCE,
                    issue="Section is too verbose; condense bullet points",
                    explanation="Template requests concise bullet points",
                    required_change="Condense bullet points",
                )
            ],
        )
        val_valid = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.VALID,
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
            mock_eval_interpret.return_value = GapResolutionDecision(
                action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION
            )
            # Only initial retrieval
            mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]\nInitial evidence"
            mock_gen.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Verbose draft v1", operation=SectionOperation.GENERATE
            )
            mock_val.side_effect = [val_invalid_writing, val_valid]
            mock_update.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Condensed draft v2", operation=SectionOperation.UPDATE
            )
            mock_assembly.return_value = BRDAssemblyResult(
                assembled_document="# Complete BRD",
                sections_assembled=["1. Executive Summary"],
                section_count=1,
                assembly_complete=True,
            )
            mock_final_val.return_value = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

            response = await agent.run_workflow_async(context=ctx, initial_state=state)

            assert response.success is True
            # retrieve_section_evidence_async was called exactly ONCE (initial only, not on rework)
            assert mock_retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_rework_loop_is_bounded_and_does_not_loop_infinitely(self):
        """The section rework loop enforces max 2 attempts and halts when limit is reached."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[search_project_knowledge])
        ctx = make_test_context()

        state = BRDAgentState.initialize_from_template(sections=["1. Executive Summary"])

        persistent_invalid = ValidationResult(
            section_name="1. Executive Summary",
            outcome=ValidationOutcome.NEEDS_REWORK,
            findings=[
                ValidationFinding(
                    category=ValidationCategory.GROUNDING,
                    issue="Missing critical unresolvable data",
                    explanation="Data is not present in project documents",
                    required_change="Provide source data",
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
            mock_retrieve.return_value = "[RETRIEVAL_SUCCESS]\nEvidence"
            mock_gen.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Draft v1", operation=SectionOperation.GENERATE
            )
            mock_val.return_value = persistent_invalid
            mock_update.return_value = SectionGenerationResult(
                section_name="1. Executive Summary", content="Draft retry", operation=SectionOperation.UPDATE
            )

            response = await agent.run_workflow_async(context=ctx, initial_state=state)

            assert response.success is False
            assert "not all template sections could be completed" in response.output_text
            # Exactly 2 update attempts allowed by MAX_SECTION_REWORK_ATTEMPTS
            assert mock_update.await_count == 2


# ---------------------------------------------------------------------------
# DEF-013: Explicit RAG Initialization & Availability Failures Tests
# ---------------------------------------------------------------------------

class TestRAGInitializationAndFailureSemantics:
    """Tests for DEF-013 ensuring RAG initialization/availability errors are explicit."""

    def test_get_brd_lead_agent_raises_retrieval_error_on_rag_init_failure(self):
        """get_brd_lead_agent must raise RetrievalError when RAG service cannot be initialized."""
        from api.dependencies import get_brd_lead_agent

        with patch("api.dependencies.get_rag_service_dependency", side_effect=Exception("Qdrant connection refused")):
            with pytest.raises(RetrievalError) as exc_info:
                get_brd_lead_agent()

            assert "RAG service initialization failed" in str(exc_info.value)
            assert "Qdrant connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_workflow_halts_explicitly_when_rag_unavailable(self):
        """Workflow must halt with an explicit error if RAG capability is missing."""
        agent = BRDLeadAgent(model=MockChatModel(), tools=[])
        assert agent.has_rag_capability is False

        ctx = make_test_context()
        response = await agent.run_workflow_async(context=ctx)

        assert response.success is False
        assert "RAG capability" in (response.error or response.output_text)

    @pytest.mark.asyncio
    async def test_distinction_between_rag_unavailable_and_no_evidence_found(self):
        """RAG unavailable halts with an infrastructure error; no evidence found evaluates as an evidence gap."""
        # 1. RAG Unavailable:
        agent_no_rag = BRDLeadAgent(model=MockChatModel(), tools=[])
        ctx = make_test_context()
        response_no_rag = await agent_no_rag.run_workflow_async(context=ctx)
        assert response_no_rag.success is False
        assert "RAG capability" in (response_no_rag.error or response_no_rag.output_text)

        # 2. RAG Available but returns no chunks:
        mock_service = MagicMock(spec=RAGService)
        mock_service.retrieve = AsyncMock(return_value=_create_mock_retrieval_result(
            project_id="proj-test-123",
            query="Overview",
            chunks=None,
        ))

        agent_rag = BRDLeadAgent(model=MockChatModel(), rag_service=mock_service)
        result = await agent_rag.retrieve_section_evidence_async("1. Executive Summary", ["Overview"], ctx)
        assert result is None  # [NO_EVIDENCE] returns None gracefully without error
        assert len(agent_rag._state.evidence) == 0
