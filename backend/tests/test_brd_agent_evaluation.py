"""Unit and integration tests for BRD Evaluation Sub-Agent capability.

Validates that:
1. The Evaluation Sub-Agent can be constructed and loads its System Instruction from markdown.
2. The System Instruction is separate from the Lead Agent's instruction.
3. Focused EvaluationContext receives current_objective, current_section, section_requirements, action_result.
4. Section requirements are dynamically extracted from authoritative template.
5. Structured EvaluationResult is returned (outcome, summary, findings, missing, unresolved, contradictions).
6. Evaluator behavior handles representative cases:
   - Sufficient result
   - Missing information
   - Irrelevant result
   - Vague / insufficient information
   - Contradictory information
   - Unresolved information
   - Not-applicable information where supported
7. No numeric scores or arbitrary confidence percentages are generated.
8. Strict boundaries: evaluator has no tools, cannot call RAG, cannot ask user, cannot delegate, cannot decide next action.
9. Lead Agent integration: ActionResult -> Evaluation Sub-Agent -> EvaluationResult -> Lead Agent State.
10. Lead Agent decision logic:
   - SUFFICIENT -> Proceed toward section generation
   - INSUFFICIENT -> RAG (if resolvable via knowledge) vs Ask User (if requiring human input)
11. RAG retry path and re-evaluation cycle reuse the same Evaluation Sub-Agent.
12. User clarification updates context and resolves open gaps.
13. Evaluation failures never silently succeed.
14. Async evaluation behaves identically to sync.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import MagicMock
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDEvaluationAgent,
    BRDLeadAgent,
    EvaluationContext,
    EvaluationFinding,
    EvaluationOutcome,
    EvaluationResult,
    InformationStatus,
    WorkflowDecision,
    create_brd_lead_agent,
    extract_section_requirements,
    get_evaluation_system_instruction_path,
    get_system_instruction_path,
    load_evaluation_system_instruction,
    load_system_instruction,
)
from agents.runtime.config import AgentConfig
from agents.runtime.state import (
    ActionResult,
    ActionSource,
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
)
from services.rag_service import RAGService, RetrievalResult, RetrievedChunk


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing evaluation cycles."""

    messages_to_return: list[AIMessage] = []
    invocations: list[list[Any]] = []
    index: int = 0
    tools_bound: list = []
    error_to_raise: Optional[Exception] = None

    def __init__(
        self,
        messages_to_return: Optional[list[AIMessage]] = None,
        error_to_raise: Optional[Exception] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.messages_to_return = list(messages_to_return or [])
        self.error_to_raise = error_to_raise
        self.invocations = []
        self.index = 0
        self.tools_bound = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations.append(messages)
        if self.error_to_raise is not None:
            raise self.error_to_raise
        if self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps({
                            "outcome": "SUFFICIENT",
                            "summary": "Default mock evaluation",
                            "findings": [],
                            "missing_information": [],
                            "unresolved_information": [],
                            "contradictions": [],
                        })
                    )
                )
            ]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-evaluation-chat-model"


# ---------------------------------------------------------------------------
# 1. Evaluator Construction & Separate System Instruction
# ---------------------------------------------------------------------------


def test_evaluator_construction_and_defaults():
    """Verify Evaluation Sub-Agent can be instantiated and has empty tools list."""
    model = MockChatModel()
    evaluator = BRDEvaluationAgent(model=model)

    assert evaluator.agent_name == "BRDEvaluationAgent"
    assert evaluator.model is model
    # Strict boundary: Evaluator has NO tools
    assert len(evaluator.tools) == 0
    assert "BRD Evaluation Sub-Agent" in evaluator.system_instruction


def test_evaluation_system_instruction_loaded_from_markdown():
    """Verify system instruction is loaded from external markdown file with required sections."""
    path = get_evaluation_system_instruction_path()
    assert path.is_file()
    assert path.name == "system_instruction.md"

    content = load_evaluation_system_instruction()
    assert len(content) > 100
    assert "## Identity" in content
    assert "## Core Responsibilities" in content
    assert "## Requirement Status Categories" in content
    assert "## Evidence Sufficiency Rule" in content
    assert "## No Numeric Scores" in content
    assert "## Strict Boundaries & Prohibitions" in content
    assert "## Output Contract" in content


def test_evaluation_system_instruction_is_separate_from_lead_agent():
    """Verify evaluator instruction is physically and conceptually separate from Lead Agent instruction."""
    eval_path = get_evaluation_system_instruction_path()
    lead_path = get_system_instruction_path()

    assert eval_path != lead_path
    assert eval_path.parent.name == "evaluation"
    assert lead_path.parent.name == "brd"

    eval_content = load_evaluation_system_instruction()
    lead_content = load_system_instruction()

    assert "BRD Evaluation Sub-Agent" in eval_content
    assert "BRD Lead Agent" in lead_content
    # Evaluator does NOT have Lead Agent's delegation or RAG determination sections
    assert "## Action Determination & Operational Branches" not in eval_content


# ---------------------------------------------------------------------------
# 2. Evaluation Input Contract & Section Requirements Extraction
# ---------------------------------------------------------------------------


def test_evaluation_context_structure_and_serialization():
    """Verify EvaluationContext encapsulates all required fields and serializes cleanly."""
    action_res = ActionResult(
        source=ActionSource.DIRECT_WORK,
        content="Analyzed authentication workflows: OAuth2 and SAML 2.0 supported.",
        context=AgentContext(project_id="proj-123"),
    )
    ctx = EvaluationContext(
        current_objective="Define authentication requirements",
        current_section="6.x Module: Authentication",
        section_requirements=["Identity Provider integration", "Session management", "MFA support"],
        relevant_working_context="Client requires enterprise SSO",
        action_result=action_res,
        metadata={"priority": "high"},
    )

    data = ctx.to_dict()
    assert data["current_objective"] == "Define authentication requirements"
    assert data["current_section"] == "6.x Module: Authentication"
    assert len(data["section_requirements"]) == 3
    assert data["action_result"]["source"] == "direct_work"
    assert data["metadata"]["priority"] == "high"

    rehydrated = EvaluationContext.from_dict(data)
    assert rehydrated.current_objective == ctx.current_objective
    assert rehydrated.current_section == ctx.current_section
    assert len(rehydrated.section_requirements) == 3
    assert rehydrated.action_result.source == ActionSource.DIRECT_WORK
    assert rehydrated.action_result.content == action_res.content


def test_extract_section_requirements_from_template():
    """Verify section requirements are extracted from authoritative BRD template."""
    # Test Section 1 Purpose & Scope
    reqs_sec1 = extract_section_requirements("1. Purpose & Scope of This Document")
    assert len(reqs_sec1) >= 2
    req_text = " ".join(reqs_sec1).lower()
    assert "purpose" in req_text or "scope" in req_text

    # Test Section 5.1 Personas
    reqs_personas = extract_section_requirements("Personas")
    assert len(reqs_personas) >= 1
    personas_text = " ".join(reqs_personas).lower()
    assert "persona" in personas_text or "responsibilities" in personas_text

    # Test Section 8 Integrations
    reqs_integrations = extract_section_requirements("8. Integrations (High-Level Business View)")
    assert len(reqs_integrations) >= 1
    int_text = " ".join(reqs_integrations).lower()
    assert "integration" in int_text or "source system" in int_text


# ---------------------------------------------------------------------------
# 3. Evaluation Behavior Cases (Sufficient, Missing, Irrelevant, Vague, etc.)
# ---------------------------------------------------------------------------


def test_case_sufficient_result():
    """Verify evaluation of a sufficient result returns SUFFICIENT outcome with structured findings."""
    eval_payload = {
        "outcome": "SUFFICIENT",
        "summary": "All required persona attributes and mapping details are fully covered.",
        "findings": [
            {
                "observation": "Admin and Member personas are defined with explicit goals and frustrations.",
                "significance": "Satisfies persona requirements for Section 5.1.",
                "status": "Present",
                "item": "Personas",
            }
        ],
        "missing_information": [],
        "unresolved_information": [],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Document primary user personas",
        current_section="5.1 Personas",
        section_requirements=["Persona name, responsibilities, goals, frustrations"],
        action_result=ActionResult(
            source=ActionSource.DIRECT_WORK,
            content="Persona 1: Admin, manages tenants, goals: security, frustrations: complex audits.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is True
    assert result.is_insufficient is False
    assert result.outcome == EvaluationOutcome.SUFFICIENT
    assert "All required persona attributes" in result.summary
    assert len(result.findings) == 1
    assert result.findings[0].status == InformationStatus.PRESENT
    assert len(result.missing_information) == 0


def test_case_missing_information():
    """Verify evaluation identifies missing requirements and reports INSUFFICIENT."""
    eval_payload = {
        "outcome": "INSUFFICIENT",
        "summary": "Manager permission boundaries and audit logging requirements are absent.",
        "findings": [
            {
                "observation": "Manager role permissions are not specified.",
                "significance": "Critical role boundary missing.",
                "status": "Missing",
                "item": "Permission Boundaries",
            }
        ],
        "missing_information": [
            "Manager permission boundaries",
            "Audit logging requirements",
        ],
        "unresolved_information": [],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Formulate Role-Based Access Control requirements",
        current_section="6.x Module: RBAC",
        section_requirements=["Role definitions", "Permission boundaries", "Audit logging"],
        action_result=ActionResult(
            source=ActionSource.DIRECT_WORK,
            content="Defined basic user role with read permissions.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert result.is_insufficient is True
    assert result.outcome == EvaluationOutcome.INSUFFICIENT
    assert len(result.missing_information) == 2
    assert "Manager permission boundaries" in result.missing_information
    assert "Audit logging requirements" in result.missing_information


def test_case_irrelevant_result():
    """Verify evaluation identifies completely irrelevant content as INSUFFICIENT."""
    eval_payload = {
        "outcome": "INSUFFICIENT",
        "summary": "Content discusses frontend styling instead of required database schema and storage.",
        "findings": [
            {
                "observation": "Output contains CSS animation guidelines instead of data persistence requirements.",
                "significance": "Completely off-scope for database module.",
                "status": "Missing",
                "item": "Storage Requirements",
            }
        ],
        "missing_information": ["Database persistence specifications", "Data retention policies"],
        "unresolved_information": [],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Specify database requirements",
        current_section="Data Storage",
        section_requirements=["Database technology", "Retention policies"],
        action_result=ActionResult(
            source=ActionSource.RAG,
            content="Use CSS grid and flexbox with lavender accent colors for the UI.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert result.outcome == EvaluationOutcome.INSUFFICIENT
    assert "styling" in result.summary.lower() or "off-scope" in result.findings[0].significance.lower()


def test_case_vague_information():
    """Verify evaluation detects vague, non-specific statements and flags them as insufficient."""
    eval_payload = {
        "outcome": "INSUFFICIENT",
        "summary": "Performance requirements are stated vaguely ('system should be fast') without metrics.",
        "findings": [
            {
                "observation": "Vague statement 'system should be fast' lacks quantitative latency or throughput metrics.",
                "significance": "Non-testable requirement.",
                "status": "Unclear / Unresolved",
                "item": "Performance Metrics",
            }
        ],
        "missing_information": ["Target latency p95/p99 (ms)", "Expected concurrent users"],
        "unresolved_information": ["Quantitative response time target"],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Define non-functional performance requirements",
        current_section="9. Assumptions & Constraints",
        section_requirements=["Latency targets", "Throughput metrics"],
        action_result=ActionResult(
            source=ActionSource.DELEGATION,
            content="The platform should be modern, reliable, and fast for all users.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert len(result.missing_information) >= 1
    assert len(result.unresolved_information) >= 1


def test_case_contradictory_information():
    """Verify evaluation detects contradictions and preserves them in contradictions list."""
    eval_payload = {
        "outcome": "INSUFFICIENT",
        "summary": "Direct contradiction regarding payment processing currency support.",
        "findings": [
            {
                "observation": "Section states multi-currency is required, but integration table lists USD-only Stripe account.",
                "significance": "Incompatible requirements prevent safe approval.",
                "status": "Contradictory",
                "item": "Currency Support",
            }
        ],
        "missing_information": [],
        "unresolved_information": ["Multi-currency gateway availability"],
        "contradictions": [
            "Requirement HL-BRD-4.1 mandates EUR and GBP support, but Gateway config specifies single-currency USD processing."
        ],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Define payment gateway requirements",
        current_section="8. Integrations",
        section_requirements=["Supported payment currencies", "Gateway provider specifications"],
        action_result=ActionResult(
            source=ActionSource.DIRECT_WORK,
            content="Payment module must support EUR, GBP, USD. Payment gateway integration: Stripe USD-only account.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert len(result.contradictions) == 1
    assert "EUR" in result.contradictions[0] and "USD" in result.contradictions[0]


def test_case_unresolved_information():
    """Verify evaluation detects unresolved ambiguities and open questions."""
    eval_payload = {
        "outcome": "INSUFFICIENT",
        "summary": "Approval threshold remains ambiguous.",
        "findings": [
            {
                "observation": "Text notes '$10,000 threshold or manager discretion, to be verified'.",
                "significance": "Unsettled business rule.",
                "status": "Unclear / Unresolved",
                "item": "Approval Threshold",
            }
        ],
        "missing_information": [],
        "unresolved_information": [
            "It is unclear whether managers can approve requests over $10,000 without VP sign-off."
        ],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Formulate approval workflow rules",
        current_section="7. Conceptual Business Workflows",
        action_result=ActionResult(
            source=ActionSource.DIRECT_WORK,
            content="Workflows: Approvals trigger above $10k, but maybe VP is needed. Open for verification.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert len(result.unresolved_information) == 1
    assert "unclear" in result.unresolved_information[0].lower()


def test_case_not_applicable_information_supported_by_context():
    """Verify evaluator recognizes legitimately not-applicable requirements as supported."""
    eval_payload = {
        "outcome": "SUFFICIENT",
        "summary": "External integrations are confirmed Not Applicable based on standalone deployment requirements.",
        "findings": [
            {
                "observation": "Project architecture specifies completely standalone air-gapped system; no third-party APIs.",
                "significance": "External Integrations requirement is legitimately Not Applicable.",
                "status": "Not Applicable",
                "item": "External Integrations",
            }
        ],
        "missing_information": [],
        "unresolved_information": [],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    evaluator = BRDEvaluationAgent(model=model)

    ctx = EvaluationContext(
        current_objective="Document system integrations",
        current_section="8. Integrations",
        section_requirements=["External APIs", "Third-party data feeds"],
        relevant_working_context="Client environment is 100% air-gapped with no external network connectivity.",
        action_result=ActionResult(
            source=ActionSource.DIRECT_WORK,
            content="Confirmed air-gapped deployment: No external integrations or third-party feeds exist or are required.",
        ),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is True
    assert result.outcome == EvaluationOutcome.SUFFICIENT
    assert len(result.findings) == 1
    assert result.findings[0].status == InformationStatus.NOT_APPLICABLE
    assert len(result.missing_information) == 0


# ---------------------------------------------------------------------------
# 4. Strict Boundaries Verification
# ---------------------------------------------------------------------------


def test_evaluator_strict_boundaries():
    """Verify Evaluation Sub-Agent enforces all architectural boundaries."""
    model = MockChatModel()
    evaluator = BRDEvaluationAgent(model=model)

    # 1. Has no tools
    assert len(evaluator.tools) == 0

    # 2. Cannot call RAG
    assert not any(getattr(t, "name", "") == "search_project_knowledge" for t in evaluator.tools)

    # 3. System instruction strictly prohibits workflow decisions and tool execution
    instr = evaluator.system_instruction
    assert "must NEVER" in instr or "must not" in instr.lower()
    assert "Call RAG" in instr
    assert "Directly access databases" in instr
    assert "Ask the user questions" in instr
    assert "Generate or draft BRD sections" in instr
    assert "Decide the next workflow action" in instr


def test_workflow_decision_owned_by_lead_agent_not_evaluation():
    """Verify WorkflowDecision is owned by the Lead Agent module and not exported by evaluation."""
    import agents.brd.agent as brd_agent_mod
    import agents.brd.evaluation as brd_eval_pkg
    import agents.brd.evaluation.agent as brd_eval_mod

    # Defined under BRD Lead Agent domain
    assert hasattr(brd_agent_mod, "WorkflowDecision")
    assert issubclass(brd_agent_mod.WorkflowDecision, str)
    assert hasattr(brd_agent_mod.WorkflowDecision, "PROCEED_TO_SECTION_GENERATION")
    assert hasattr(brd_agent_mod.WorkflowDecision, "RAG")
    assert hasattr(brd_agent_mod.WorkflowDecision, "ASK_USER")

    # NOT defined or exported by evaluation package
    assert not hasattr(brd_eval_mod, "WorkflowDecision")
    assert "WorkflowDecision" not in getattr(brd_eval_pkg, "__all__", [])
    assert not hasattr(brd_eval_pkg, "WorkflowDecision")


# ---------------------------------------------------------------------------
# 5. Lead Agent Integration: ActionResult -> Evaluator -> EvaluationResult -> State
# ---------------------------------------------------------------------------


def test_lead_agent_evaluation_integration_and_state():
    """Verify Lead Agent coordinates ActionResult -> Evaluator -> State update."""
    eval_json = json.dumps({
        "outcome": "SUFFICIENT",
        "summary": "Requirements are sufficiently detailed.",
        "findings": [{"observation": "All modules present", "significance": "Complete", "status": "Present"}],
        "missing_information": [],
        "unresolved_information": [],
        "contradictions": [],
    })
    mock_model = MockChatModel(messages_to_return=[
        AIMessage(content="Direct work output for personas"),
        AIMessage(content=eval_json),
    ])
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=mock_model)

    # 1. Lead Agent performs direct work
    resp = lead_agent.execute("Define module personas", current_task="Persona definition")
    assert resp.action_result is not None
    assert lead_agent.state.latest_action_result is resp.action_result

    # 2. Lead Agent evaluates the ActionResult
    eval_result = lead_agent.evaluate(section="5.1 Personas")

    # 3. Evaluation result is retained on state
    assert eval_result.is_sufficient is True
    assert lead_agent.state.latest_evaluation_result is eval_result
    assert len(lead_agent.state.evaluation_history) == 1


def test_lead_agent_decision_logic_sufficient():
    """Verify Lead Agent chooses PROCEED_TO_SECTION_GENERATION when evaluation is SUFFICIENT."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model)

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.SUFFICIENT,
        summary="All required criteria met",
        findings=[],
    )
    lead_agent.state.set_evaluation_result(eval_result)

    decision = lead_agent.decide_next_step()
    assert decision == WorkflowDecision.PROCEED_TO_SECTION_GENERATION


def test_lead_agent_decision_logic_insufficient_chooses_rag():
    """Verify Lead Agent chooses RAG when gaps reasonably exist in project knowledge base."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    # Equip RAG capability
    lead_agent = BRDLeadAgent(config=config, model=model, enable_rag=True)
    assert lead_agent.has_rag_capability is True

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.INSUFFICIENT,
        summary="Missing architectural details",
        missing_information=["Database replication schema", "OAuth2 endpoint URL"],
    )
    lead_agent.state.set_evaluation_result(eval_result)

    decision = lead_agent.decide_next_step()
    assert decision == WorkflowDecision.RAG


def test_lead_agent_decision_logic_insufficient_chooses_ask_user():
    """Verify Lead Agent chooses ASK_USER when gaps require user decision or approval."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model, enable_rag=True)

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.INSUFFICIENT,
        summary="Missing business budget decision",
        missing_information=["User preference on monthly budget limit", "Stakeholder sign-off on scope"],
    )
    lead_agent.state.set_evaluation_result(eval_result)

    decision = lead_agent.decide_next_step()
    assert decision == WorkflowDecision.ASK_USER


def test_lead_agent_decision_without_rag_capability_defaults_to_ask_user():
    """Verify Lead Agent without RAG capability always asks user when evaluation is INSUFFICIENT."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model, enable_rag=False)
    assert lead_agent.has_rag_capability is False

    eval_result = EvaluationResult(
        outcome=EvaluationOutcome.INSUFFICIENT,
        summary="Missing specifications",
        missing_information=["Architecture details"],
    )
    lead_agent.state.set_evaluation_result(eval_result)

    decision = lead_agent.decide_next_step()
    assert decision == WorkflowDecision.ASK_USER


# ---------------------------------------------------------------------------
# 6. Re-Evaluation & Full Workflow Cycle
# ---------------------------------------------------------------------------


def test_re_evaluation_cycle_via_same_evaluator():
    """Verify initial ActionResult -> INSUFFICIENT -> RAG -> new ActionResult -> SUFFICIENT cycle."""
    eval_insufficient = json.dumps({
        "outcome": "INSUFFICIENT",
        "summary": "Initial direct work missed SSO certificate specifications.",
        "findings": [],
        "missing_information": ["SSO x509 certificate validation rules"],
        "unresolved_information": [],
        "contradictions": [],
    })
    eval_sufficient = json.dumps({
        "outcome": "SUFFICIENT",
        "summary": "Retrieved RAG knowledge resolved certificate validation rules.",
        "findings": [
            {"observation": "Certificate thumbprint pinning rules specified", "significance": "Complete", "status": "Present"}
        ],
        "missing_information": [],
        "unresolved_information": [],
        "contradictions": [],
    })

    # Sequence of mock outputs:
    # 1. Direct work execution output
    # 2. First evaluation output (INSUFFICIENT)
    # 3. RAG search execution output
    # 4. Second evaluation output (SUFFICIENT)
    model = MockChatModel(messages_to_return=[
        AIMessage(content="Direct work: Drafted basic SSO overview."),
        AIMessage(content=eval_insufficient),
        AIMessage(content="RAG output: Certificate validation requires SHA-256 x509 thumbprint verification."),
        AIMessage(content=eval_sufficient),
    ])
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model, enable_rag=True)

    # Cycle 1: Direct Work
    resp1 = lead_agent.execute("Draft SSO overview", current_task="SSO")
    eval1 = lead_agent.evaluate(action_result=resp1.action_result)
    assert eval1.is_insufficient is True
    decision1 = lead_agent.decide_next_step()
    assert decision1 == WorkflowDecision.RAG

    # Cycle 2: RAG Retry
    resp2, eval2 = lead_agent.retry_with_rag(query="SSO certificate validation rules")
    assert resp2.action_result is not None
    assert eval2 is not None
    assert eval2.is_sufficient is True

    # Cycle 3: Lead Agent decision is now SUFFICIENT
    decision2 = lead_agent.decide_next_step()
    assert decision2 == WorkflowDecision.PROCEED_TO_SECTION_GENERATION
    assert len(lead_agent.state.evaluation_history) == 2


def test_user_clarification_path_updates_state():
    """Verify user clarification updates evidence and resolves unresolved items."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model)

    lead_agent.state.add_unresolved("Manager approval limit")
    assert "Manager approval limit" in lead_agent.state.unresolved_information

    # User answers the question
    lead_agent.receive_user_clarification(
        answer="The manager approval limit is $25,000 as agreed with CFO.",
        resolved_item="Manager approval limit",
    )

    # Gap is removed from unresolved list and recorded in evidence
    assert "Manager approval limit" not in lead_agent.state.unresolved_information
    assert len(lead_agent.state.evidence) == 1
    assert "25,000" in lead_agent.state.evidence[0]["content"]


# ---------------------------------------------------------------------------
# 7. Error Resilience & Fallback Handling
# ---------------------------------------------------------------------------


def test_evaluator_failure_never_silently_succeeds():
    """Verify runtime error during evaluation produces INSUFFICIENT result with failure metadata."""
    faulty_model = MockChatModel(error_to_raise=RuntimeError("LLM service unavailable"))

    evaluator = BRDEvaluationAgent(model=faulty_model)
    ctx = EvaluationContext(
        current_objective="Test error resilience",
        current_section="Error Section",
        action_result=ActionResult(source=ActionSource.DIRECT_WORK, content="Content"),
    )

    result = evaluator.evaluate(context=ctx)
    assert result.is_sufficient is False
    assert result.is_insufficient is True
    assert result.outcome == EvaluationOutcome.INSUFFICIENT
    assert "runtime error" in result.summary.lower() or "failed" in result.summary.lower() or "llm service unavailable" in result.summary.lower()
    assert result.metadata.get("error") is not None


# ---------------------------------------------------------------------------
# 8. Async Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_evaluation_and_lead_agent_integration():
    """Verify asynchronous evaluation executes identically to sync."""
    eval_payload = {
        "outcome": "SUFFICIENT",
        "summary": "Async evaluation verified completeness.",
        "findings": [{"observation": "Async complete", "significance": "Valid", "status": "Present"}],
        "missing_information": [],
        "unresolved_information": [],
        "contradictions": [],
    }
    model = MockChatModel(messages_to_return=[AIMessage(content=json.dumps(eval_payload))])
    config = AgentConfig(model="test-model", api_key="test-key")
    lead_agent = BRDLeadAgent(config=config, model=model)

    act_result = ActionResult(
        source=ActionSource.DELEGATION,
        content="Delegated sub-agent execution completed.",
    )
    eval_res = await lead_agent.evaluate_async(action_result=act_result)

    assert eval_res.is_sufficient is True
    assert eval_res.outcome == EvaluationOutcome.SUFFICIENT
    assert lead_agent.state.latest_evaluation_result is eval_res
