"""Unit and integration tests for BRD Section Validation Sub-Agent and Rework Loop capability.

Validates that:
1. Validator Construction & System Instruction: separate from Lead Agent and Evaluation/Generation instructions.
2. Correct Input Context: SectionValidationContext carries section_name, section_content, section_requirements,
   template_structure, available_information, prior_rework_feedback.
3. Valid Section: returns VALID outcome, is_valid=True, rework_feedback=None.
4. Missing Required Information: returns NEEDS_REWORK with COMPLETENESS finding and actionable required_change.
5. Template/Structure Violation: returns NEEDS_REWORK with TEMPLATE_COMPLIANCE finding and required_change.
6. Insufficient Specificity: returns NEEDS_REWORK with SPECIFICITY finding.
7. Unsupported/Fabricated Information: returns NEEDS_REWORK with GROUNDING finding.
8. Contradiction/Inconsistency: returns NEEDS_REWORK with CONSISTENCY finding.
9. Irrelevant Content: returns NEEDS_REWORK with RELEVANCE finding.
10. Actionable Rework Feedback: validates that rework_feedback is synthesized and actionable.
11. Async Execution: validate_async functions identically to synchronous validate.
12. Validator Execution Failure Handling: returns explicit failure state (NEEDS_REWORK), never silently treats as VALID.
13. Lead Agent Integration: Lead Agent can invoke validate_section directly.
14. Validation Result Stored in State: state tracks latest_validation_result and validation_history.
15. VALID Result Does Not Trigger Rework: marks section COMPLETED without rework.
16. NEEDS_REWORK Stores Rework Feedback: marks section NEEDS_REVISION and saves feedback in state.
17. Rework Feedback Reaches Section Generation: generator receives feedback during update.
18. Updated Section Can Be Validated Again: multi-step rework cycle progresses cleanly.
19. Repeated Validation/Rework Remains Controlled: retry safety bounds iteration count.
20. Section Generation Responsibility: generation remains responsible only for authoring/updating.
21. Section Validation Responsibility: validation remains responsible only for evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Optional
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDEvaluationAgent,
    BRDLeadAgent,
    BRDSectionGenerationAgent,
    BRDSectionStatus,
    BRDSectionValidationAgent,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
    create_brd_lead_agent,
    extract_section_requirements,
    extract_section_template,
    get_evaluation_system_instruction_path,
    get_generation_system_instruction_path,
    get_system_instruction_path,
    get_validation_system_instruction_path,
    load_evaluation_system_instruction,
    load_generation_system_instruction,
    load_system_instruction,
    load_validation_system_instruction,
)
from agents.runtime.config import AgentConfig
from agents.runtime.state import ActionResult, ActionSource, AgentContext


class MockValidationChatModel(BaseChatModel):
    """Deterministic mock chat model for testing section validation and rework cycles."""

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
        # Default mock: VALID validation
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps({
                            "outcome": "VALID",
                            "summary": "Section meets all template and requirement criteria.",
                            "findings": [],
                            "rework_feedback": None,
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
        return "mock-validation-chat-model"


# ---------------------------------------------------------------------------
# 1. Validator Construction & System Instruction Separation
# ---------------------------------------------------------------------------


def test_validator_construction_and_defaults():
    """Verify Section Validation Sub-Agent can be instantiated and has empty tools list."""
    model = MockValidationChatModel()
    validator = BRDSectionValidationAgent(model=model)

    assert validator.agent_name == "BRDSectionValidationAgent"
    assert validator.model is model
    # Strict boundary: Validator has NO tools
    assert len(validator.tools) == 0
    assert "BRD Section Validation Sub-Agent" in validator.system_instruction


def test_validation_system_instruction_loaded_from_markdown():
    """Verify system instruction is loaded from external markdown file with required sections."""
    path = get_validation_system_instruction_path()
    assert path.is_file()
    assert path.name == "system_instruction.md"
    assert path.parent.name == "section_validation"

    content = load_validation_system_instruction()
    assert len(content) > 100
    assert "## Identity" in content
    assert "## Objective" in content
    assert "## Core Validation Dimensions" in content
    assert "## Categorical Validation Outcome" in content
    assert "## Concrete Findings & Actionable Rework Feedback" in content
    assert "## Strict Boundaries & Prohibitions" in content
    assert "## Output Contract" in content


def test_system_instructions_are_distinct():
    """Verify that Lead Agent, Evaluation Sub-Agent, Generation Sub-Agent, and Validation Sub-Agent have distinct system instructions."""
    lead_si = load_system_instruction()
    eval_si = load_evaluation_system_instruction()
    gen_si = load_generation_system_instruction()
    val_si = load_validation_system_instruction()

    assert lead_si != eval_si
    assert lead_si != gen_si
    assert lead_si != val_si
    assert eval_si != gen_si
    assert eval_si != val_si
    assert gen_si != val_si


# ---------------------------------------------------------------------------
# 2. Correct Input Context
# ---------------------------------------------------------------------------


def test_validation_context_serialization():
    """Verify SectionValidationContext serializes and deserializes accurately."""
    ctx = SectionValidationContext(
        section_name="5. Stakeholders & Personas",
        section_content="# 5. Stakeholders & Personas\n\nContent here",
        section_requirements=["Identify key personas", "Define responsibilities"],
        template_structure="## 5. Stakeholders & Personas",
        available_information=["Persona A: Claims Officer", "Persona B: Policyholder"],
        prior_rework_feedback="Add responsibilities table",
        metadata={"project_id": "test-proj-123"},
    )

    data = ctx.to_dict()
    assert data["section_name"] == "5. Stakeholders & Personas"
    assert "Content here" in data["section_content"]
    assert len(data["section_requirements"]) == 2
    assert data["prior_rework_feedback"] == "Add responsibilities table"
    assert data["metadata"]["project_id"] == "test-proj-123"

    deserialized = SectionValidationContext.from_dict(data)
    assert deserialized.section_name == ctx.section_name
    assert deserialized.section_content == ctx.section_content
    assert deserialized.prior_rework_feedback == ctx.prior_rework_feedback


# ---------------------------------------------------------------------------
# 3. Valid Section Validation
# ---------------------------------------------------------------------------


def test_validate_valid_section():
    """Verify validating a compliant section returns VALID with no rework feedback."""
    response_json = {
        "outcome": "VALID",
        "summary": "Section 5 conforms to template structure and addresses all requirements.",
        "findings": [],
        "rework_feedback": None,
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    ctx = SectionValidationContext(
        section_name="5. Stakeholders & Personas",
        section_content="# 5. Stakeholders & Personas\n\n## 5.1 Personas\n- Claims Officer",
        section_requirements=["Define primary user personas"],
        template_structure="## 5. Stakeholders & Personas",
        available_information=["Claims Officer persona details"],
    )

    result = validator.validate(ctx)

    assert result.outcome == ValidationOutcome.VALID
    assert result.is_valid is True
    assert result.needs_rework is False
    assert len(result.findings) == 0
    assert result.rework_feedback is None


# ---------------------------------------------------------------------------
# 4. Missing Required Information
# ---------------------------------------------------------------------------


def test_validate_missing_required_information():
    """Verify validator flags missing required information with actionable required_change."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section 5 is missing required persona approval authorities.",
        "findings": [
            {
                "category": "Completeness",
                "issue": "Missing persona approval authorities",
                "explanation": "Section requirement 2 mandates defining approval authorities for each persona.",
                "required_change": "Add approval authority thresholds to each persona in Subsection 5.1.",
            }
        ],
        "rework_feedback": "Add approval authority thresholds to each persona in Subsection 5.1 using project information.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    ctx = SectionValidationContext(
        section_name="5. Stakeholders & Personas",
        section_content="# 5. Stakeholders & Personas\n\n- User A",
        section_requirements=["Define personas", "Define approval authority thresholds"],
    )

    result = validator.validate(ctx)

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.is_valid is False
    assert result.needs_rework is True
    assert len(result.findings) == 1
    assert result.findings[0].category == ValidationCategory.COMPLETENESS
    assert "approval authorit" in result.findings[0].issue.lower()
    assert "Add approval authority" in result.findings[0].required_change
    assert result.rework_feedback is not None


# ---------------------------------------------------------------------------
# 5. Template / Structure Violation
# ---------------------------------------------------------------------------


def test_validate_template_structure_violation():
    """Verify validator detects missing required table or heading hierarchy."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section fails template structure: missing required Markdown table.",
        "findings": [
            {
                "category": "Template Compliance",
                "issue": "Missing Stakeholder Matrix Table",
                "explanation": "Authoritative BRD template requires a Markdown table for Stakeholders.",
                "required_change": "Format the stakeholders into the prescribed Markdown table structure.",
            }
        ],
        "rework_feedback": "Format the stakeholders into the prescribed Markdown table structure with columns: Role, Department, Responsibilities.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "5. Stakeholders & Personas",
        "section_content": "Just raw bullet points without tables",
        "template_structure": "| Role | Department | Responsibilities |\n|---|---|---|",
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.findings[0].category == ValidationCategory.TEMPLATE_COMPLIANCE
    assert "Stakeholder Matrix Table" in result.findings[0].issue


# ---------------------------------------------------------------------------
# 6. Insufficient Specificity
# ---------------------------------------------------------------------------


def test_validate_insufficient_specificity():
    """Verify validator flags hand-waving or vague statements."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section content is generic and lacks concrete business requirements.",
        "findings": [
            {
                "category": "Specificity",
                "issue": "Vague SLA metrics",
                "explanation": "The section states 'claims will be processed quickly' without specific metrics.",
                "required_change": "Replace 'quickly' with the concrete SLA defined in evidence (under 24 hours).",
            }
        ],
        "rework_feedback": "Specify concrete SLA metrics (under 24 hours) as stated in project evidence.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "3. In-Scope Modules",
        "section_content": "The system processes claims very fast and efficiently.",
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.findings[0].category == ValidationCategory.SPECIFICITY
    assert "SLA" in result.findings[0].issue


# ---------------------------------------------------------------------------
# 7. Unsupported / Fabricated Information
# ---------------------------------------------------------------------------


def test_validate_unsupported_fabricated_information():
    """Verify validator detects fabricated project facts not supported by evidence."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section contains claims not supported by the provided evidence.",
        "findings": [
            {
                "category": "Grounding",
                "issue": "Fabricated blockchain integration requirement",
                "explanation": "The section specifies a private Ethereum blockchain, which appears nowhere in the evidence.",
                "required_change": "Remove all references to blockchain and rely solely on PostgreSQL audit logs as evidenced.",
            }
        ],
        "rework_feedback": "Remove ungrounded blockchain claims; align audit architecture with provided evidence.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "4. Technical Requirements",
        "section_content": "Architecture requires an Ethereum consortium blockchain for claim ledgers.",
        "available_information": ["Relational PostgreSQL database for audit tracking"],
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.findings[0].category == ValidationCategory.GROUNDING
    assert "blockchain" in result.findings[0].issue.lower()


# ---------------------------------------------------------------------------
# 8. Contradiction / Inconsistency
# ---------------------------------------------------------------------------


def test_validate_contradiction_and_inconsistency():
    """Verify validator flags internal contradictions and conflicts with evidence."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section has internal contradiction regarding user authentication.",
        "findings": [
            {
                "category": "Consistency",
                "issue": "Contradictory authentication methods",
                "explanation": "Subsection 4.1 specifies SSO via Okta, but Subsection 4.3 requires standalone username/password.",
                "required_change": "Reconcile authentication to corporate Okta SSO across all subsections.",
            }
        ],
        "rework_feedback": "Standardize authentication on Okta SSO across both subsections.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "4. Security & Access",
        "section_content": "4.1 Okta SSO. 4.3 Standalone passwords only.",
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.findings[0].category == ValidationCategory.CONSISTENCY
    assert "authentication" in result.findings[0].issue.lower()


# ---------------------------------------------------------------------------
# 9. Irrelevant Content
# ---------------------------------------------------------------------------


def test_validate_irrelevant_content():
    """Verify validator flags content that drifts into out-of-scope technical implementation."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Section includes out-of-scope code snippets and low-level SQL scripts.",
        "findings": [
            {
                "category": "Relevance",
                "issue": "Out-of-scope SQL table creation DDL scripts",
                "explanation": "High-level BRD sections focus on business capabilities, not physical database DDL.",
                "required_change": "Remove SQL DDL statements and describe data entities at business requirement level.",
            }
        ],
        "rework_feedback": "Remove SQL DDL code; replace with high-level entity requirements.",
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "3. In-Scope Business Modules",
        "section_content": "CREATE TABLE claims (id INT, amount DECIMAL);",
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.findings[0].category == ValidationCategory.RELEVANCE


# ---------------------------------------------------------------------------
# 10. Actionable Rework Feedback Synthesis
# ---------------------------------------------------------------------------


def test_rework_feedback_synthesized_if_omitted():
    """Verify that if model returns NEEDS_REWORK without rework_feedback, it is synthesized from findings."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Multiple compliance issues detected.",
        "findings": [
            {
                "category": "Completeness",
                "issue": "Missing escalation path",
                "explanation": "No escalation hierarchy defined.",
                "required_change": "Add escalation tiers 1 through 3.",
            }
        ],
        "rework_feedback": None,  # Intentionally omitted by model
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    result = validator.validate({
        "section_name": "6. Operational Workflow",
        "section_content": "Basic flow without escalation.",
    })

    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.rework_feedback is not None
    assert "escalation tiers 1 through 3" in result.rework_feedback


# ---------------------------------------------------------------------------
# 11. Async Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_async_execution():
    """Verify async validate_async execution behaves identically to synchronous validation."""
    response_json = {
        "outcome": "VALID",
        "summary": "Async validation successful.",
        "findings": [],
        "rework_feedback": None,
    }
    model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    validator = BRDSectionValidationAgent(model=model)

    ctx = SectionValidationContext(
        section_name="1. Purpose & Scope",
        section_content="# 1. Purpose & Scope\n\nDefines business purpose.",
    )

    result = await validator.validate_async(ctx)

    assert result.outcome == ValidationOutcome.VALID
    assert result.is_valid is True
    assert result.summary == "Async validation successful."


# ---------------------------------------------------------------------------
# 12. Validator Execution Failure Handling
# ---------------------------------------------------------------------------


def test_validator_execution_failure_handling():
    """Verify validator returns explicit failure state (NEEDS_REWORK) and never invents success on crash."""
    model = MockValidationChatModel(
        error_to_raise=RuntimeError("Simulated LLM network failure")
    )
    validator = BRDSectionValidationAgent(model=model)

    ctx = SectionValidationContext(
        section_name="5. Stakeholders & Personas",
        section_content="Content to validate",
    )

    result = validator.validate(ctx)

    # Must NOT silently treat as VALID
    assert result.outcome == ValidationOutcome.NEEDS_REWORK
    assert result.is_valid is False
    assert result.needs_rework is True
    assert "error" in result.metadata
    assert "Simulated LLM network failure" in result.metadata["error"]


# ---------------------------------------------------------------------------
# 13. Lead Agent Integration: Invoke Validator
# ---------------------------------------------------------------------------


def test_lead_agent_can_invoke_validator():
    """Verify Lead Agent can invoke section validation directly."""
    response_json = {
        "outcome": "VALID",
        "summary": "Section 5 passed validation.",
        "findings": [],
        "rework_feedback": None,
    }
    val_model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    # Seed section content in state
    lead.state.set_current_section("5. Stakeholders & Personas")
    lead.state.set_section_content("5. Stakeholders & Personas", "# 5. Stakeholders & Personas\n\nContent")

    val_res = lead.validate_section()

    assert val_res.outcome == ValidationOutcome.VALID
    assert val_res.is_valid is True


# ---------------------------------------------------------------------------
# 14. Validation Result Stored in State
# ---------------------------------------------------------------------------


def test_lead_agent_stores_validation_result_in_state():
    """Verify validation result is retained in BRDAgentState and recorded in history."""
    response_json = {
        "outcome": "VALID",
        "summary": "State tracking test.",
        "findings": [],
        "rework_feedback": None,
    }
    val_model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    lead.state.set_current_section("5. Stakeholders & Personas")
    lead.state.set_section_content("5. Stakeholders & Personas", "# Content")

    val_res = lead.validate_section()

    assert lead.state.latest_validation_result is val_res
    assert len(lead.state.validation_history) == 1
    assert lead.state.validation_history[0] is val_res
    assert lead.state.get_latest_validation_result() is val_res


# ---------------------------------------------------------------------------
# 15. VALID Result Does Not Trigger Rework & Accepts Section
# ---------------------------------------------------------------------------


def test_valid_result_accepts_section_and_clears_rework():
    """Verify VALID outcome transitions section status to COMPLETED and clears rework feedback."""
    response_json = {
        "outcome": "VALID",
        "summary": "Section 5 is complete and valid.",
        "findings": [],
        "rework_feedback": None,
    }
    val_model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    lead.state.set_current_section("5. Stakeholders & Personas")
    lead.state.set_section_content("5. Stakeholders & Personas", "# Content")
    lead.state.set_rework_feedback("5. Stakeholders & Personas", "Old feedback")

    val_res = lead.validate_section()

    assert val_res.is_valid is True
    # Existing section progress updated to COMPLETED (no duplicate accepted_sections list)
    assert lead.state.get_section_status("5. Stakeholders & Personas") == BRDSectionStatus.COMPLETED
    assert "5. Stakeholders & Personas" in lead.state.get_completed_sections()
    # Rework feedback cleared
    assert lead.state.get_rework_feedback("5. Stakeholders & Personas") is None


# ---------------------------------------------------------------------------
# 16. NEEDS_REWORK Stores Feedback & Sets Status
# ---------------------------------------------------------------------------


def test_needs_rework_stores_feedback_in_state():
    """Verify NEEDS_REWORK transitions section status to NEEDS_REVISION and saves rework feedback."""
    response_json = {
        "outcome": "NEEDS_REWORK",
        "summary": "Missing compliance requirements.",
        "findings": [
            {
                "category": "Requirement Coverage",
                "issue": "Missing GDPR compliance note",
                "explanation": "Data privacy requirements not met.",
                "required_change": "Add GDPR compliance subsection.",
            }
        ],
        "rework_feedback": "Add GDPR compliance subsection under Section 5.",
    }
    val_model = MockValidationChatModel(
        messages_to_return=[AIMessage(content=json.dumps(response_json))]
    )
    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    lead.state.set_current_section("5. Stakeholders & Personas")
    lead.state.set_section_content("5. Stakeholders & Personas", "# Incomplete Content")

    val_res = lead.validate_section()

    assert val_res.needs_rework is True
    assert lead.state.get_section_status("5. Stakeholders & Personas") == BRDSectionStatus.NEEDS_REVISION
    assert lead.state.get_rework_feedback("5. Stakeholders & Personas") == "Add GDPR compliance subsection under Section 5."


# ---------------------------------------------------------------------------
# 17. Rework Feedback Reaches Section Generation
# ---------------------------------------------------------------------------


def test_rework_feedback_reaches_section_generator():
    """Verify that stored rework feedback is passed to the Section Generation Sub-Agent during update."""
    gen_model = MockValidationChatModel(
        messages_to_return=[
            AIMessage(
                content=json.dumps({
                    "section_name": "5. Stakeholders & Personas",
                    "operation": "update",
                    "summary": "Incorporated GDPR feedback.",
                    "content": "# 5. Stakeholders & Personas\n\n## 5.2 GDPR Compliance\n- Data privacy compliant.",
                })
            )
        ]
    )
    generator = BRDSectionGenerationAgent(model=gen_model)
    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_generator=generator,
    )

    lead.state.set_current_section("5. Stakeholders & Personas")
    lead.state.set_section_content("5. Stakeholders & Personas", "# Initial baseline")
    lead.state.set_rework_feedback("5. Stakeholders & Personas", "Add GDPR subsection")

    update_res = lead.update_section()

    assert update_res.is_update is True
    # Verify generator prompt received the rework feedback
    assert len(gen_model.invocations) > 0
    prompt_sent = str(gen_model.invocations[0][-1].content)
    assert "Add GDPR subsection" in prompt_sent


# ---------------------------------------------------------------------------
# 18. Updated Section Can Be Validated Again
# ---------------------------------------------------------------------------


def test_rework_retry_loop_success():
    """Verify complete rework loop: Generate -> Validate (NEEDS_REWORK) -> Update -> Validate (VALID)."""
    # 1. Initial generation returns draft
    # 2. Update returns revised content
    gen_responses = [
        AIMessage(
            content=json.dumps({
                "section_name": "5. Stakeholders & Personas",
                "operation": "generate",
                "summary": "Initial draft",
                "content": "# 5. Stakeholders & Personas\n\nDraft content without metrics.",
            })
        ),
        AIMessage(
            content=json.dumps({
                "section_name": "5. Stakeholders & Personas",
                "operation": "update",
                "summary": "Revised draft with metrics",
                "content": "# 5. Stakeholders & Personas\n\nRevised content with SLA < 24h.",
            })
        ),
    ]

    # 1. First validation returns NEEDS_REWORK
    # 2. Second validation returns VALID
    val_responses = [
        AIMessage(
            content=json.dumps({
                "outcome": "NEEDS_REWORK",
                "summary": "Missing SLA metric.",
                "findings": [
                    {
                        "category": "Specificity",
                        "issue": "Missing SLA metric",
                        "explanation": "No SLA duration specified.",
                        "required_change": "Add SLA < 24h.",
                    }
                ],
                "rework_feedback": "Add SLA < 24h as defined in evidence.",
            })
        ),
        AIMessage(
            content=json.dumps({
                "outcome": "VALID",
                "summary": "Revised section now includes SLA metrics and is valid.",
                "findings": [],
                "rework_feedback": None,
            })
        ),
    ]

    gen_model = MockValidationChatModel(messages_to_return=gen_responses)
    val_model = MockValidationChatModel(messages_to_return=val_responses)

    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_generator=BRDSectionGenerationAgent(model=gen_model),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    lead.state.set_current_section("5. Stakeholders & Personas")

    gen_result, val_result = lead.generate_and_validate_section(
        section="5. Stakeholders & Personas",
        available_information=["Target SLA is under 24 hours"],
        max_rework_attempts=2,
    )

    # Final outcome must be VALID
    assert val_result.is_valid is True
    assert lead.state.get_section_status("5. Stakeholders & Personas") == BRDSectionStatus.COMPLETED
    assert "SLA < 24h" in gen_result.content
    # Verified 2 generations and 2 validations occurred
    assert len(gen_model.invocations) == 2
    assert len(val_model.invocations) == 2


# ---------------------------------------------------------------------------
# 19. Repeated Validation / Rework Remains Controlled (Retry Safety)
# ---------------------------------------------------------------------------


def test_rework_loop_bounded_by_max_attempts():
    """Verify that repeated validation failures are safely bounded by max_rework_attempts."""
    # Always return NEEDS_REWORK to simulate stubborn failure
    val_responses = [
        AIMessage(
            content=json.dumps({
                "outcome": "NEEDS_REWORK",
                "summary": f"Persistent issue attempt {i}",
                "findings": [
                    {
                        "category": "Requirement Coverage",
                        "issue": "Persistent gap",
                        "explanation": "Still missing required elements.",
                        "required_change": "Add required elements.",
                    }
                ],
                "rework_feedback": "Please fix persistent gap.",
            })
        )
        for i in range(10)
    ]

    gen_responses = [
        AIMessage(
            content=json.dumps({
                "section_name": "5. Stakeholders & Personas",
                "operation": "generate" if i == 0 else "update",
                "summary": f"Attempt {i}",
                "content": f"# Attempt {i} content",
            })
        )
        for i in range(10)
    ]

    gen_model = MockValidationChatModel(messages_to_return=gen_responses)
    val_model = MockValidationChatModel(messages_to_return=val_responses)

    lead = create_brd_lead_agent(
        model=MockValidationChatModel(),
        config=AgentConfig(model="mock-lead", api_key="test-key"),
        section_generator=BRDSectionGenerationAgent(model=gen_model),
        section_validator=BRDSectionValidationAgent(model=val_model),
    )

    max_retries = 2
    gen_result, val_result = lead.generate_and_validate_section(
        section="5. Stakeholders & Personas",
        max_rework_attempts=max_retries,
    )

    # Initial generation (1) + max_retries (2) = 3 total generations
    assert len(gen_model.invocations) == 1 + max_retries
    # Initial validation (1) + max_retries validations (2) = 3 total validations
    assert len(val_model.invocations) == 1 + max_retries

    # Result should reflect the final validation attempt (NEEDS_REWORK) without infinite looping
    assert val_result.needs_rework is True
    assert lead.state.get_section_status("5. Stakeholders & Personas") == BRDSectionStatus.NEEDS_REVISION


# ---------------------------------------------------------------------------
# 20. Section Generation Responsibility Separation
# ---------------------------------------------------------------------------


def test_generator_does_not_depend_on_validator():
    """Verify Section Generation Sub-Agent has no tools, no validator reference, and focuses only on authoring."""
    model = MockValidationChatModel()
    generator = BRDSectionGenerationAgent(model=model)

    assert not hasattr(generator, "validator")
    assert not hasattr(generator, "section_validator")
    assert len(generator.tools) == 0


# ---------------------------------------------------------------------------
# 21. Section Validation Responsibility Separation
# ---------------------------------------------------------------------------


def test_validator_does_not_depend_on_generator():
    """Verify Section Validation Sub-Agent has no tools, no generator reference, and focuses only on validation."""
    model = MockValidationChatModel()
    validator = BRDSectionValidationAgent(model=model)

    assert not hasattr(validator, "generator")
    assert not hasattr(validator, "section_generator")
    assert len(validator.tools) == 0
