"""Tests for document-level Final BRD Validation.

Verifies:
1. Valid complete BRD returns VALID outcome.
2. Cross-section contradiction returns NEEDS_REWORK and identifies affected sections.
3. Requirement conflict returns NEEDS_REWORK.
4. Terminology inconsistency identified across sections.
5. Unsupported claims flagged under Grounding.
6. Document-level completeness gaps identified.
7. Duplicate / conflicting requirements identified.
8. Coherent document returns VALID without spurious findings.
9. No tools: Final Validation Agent is equipped with zero tools.
10. No document rewriting: assembled BRD remains strictly unchanged.
11. State persistence: serialization/deserialization preserves final validation results.
12. Lead Agent integration: sync and async execution, state tracking, and history.
13. Error behavior: no assembled BRD raises ValueError; execution failure never produces VALID.
14. Meaningful lifecycle logging for started, completed, rework required, and errors.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest

from agents.brd import (
    BRDAgentState,
    BRDFinalValidationAgent,
    BRDLeadAgent,
    FinalValidationCategory,
    FinalValidationContext,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
    FinalValidationSeverity,
    create_brd_lead_agent,
    load_final_validation_system_instruction,
)
from agents.runtime.state import AgentContext


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------


class MockFinalValidationChatModel(BaseChatModel):
    """Deterministic mock chat model for final validation testing."""

    tools_bound: list[Any] = []
    invoke_count: int = 0

    def __init__(self, mode: str = "valid", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "invoke_count", 0)
        object.__setattr__(self, "tools_bound", [])

    @property
    def _llm_type(self) -> str:
        return "mock_final_validation_model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockFinalValidationChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        object.__setattr__(self, "invoke_count", getattr(self, "invoke_count", 0) + 1)
        prompt_text = " ".join([m.content for m in messages if isinstance(m.content, str)])

        # Error simulation mode
        if self._mode == "error":
            raise RuntimeError("Underlying LLM service unreachable")

        # Specific mock scenarios
        if self._mode == "cross_section_contradiction":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Cross-section contradiction detected between Section 3 and Section 7 regarding user scale.",
                "findings": [
                    {
                        "category": "Cross-Section Consistency",
                        "severity": "ERROR",
                        "issue": "User scale contradiction",
                        "explanation": "Section 3 limits product to 10,000 internal users, while Section 7 specifies 100,000 public users.",
                        "affected_sections": ["3. In-Scope Business Modules & Feature Groups", "7. Conceptual Business Workflows"],
                        "evidence": "Section 3: '10,000 internal users' vs Section 7: '100,000 public users'",
                        "required_change": "Align user scale between scope and workflows.",
                    }
                ],
                "rework_feedback": "Resolve user scale conflict between Section 3 and Section 7.",
            }
        elif self._mode == "requirement_conflict":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Conflicting requirements detected between REQ-001 and REQ-024.",
                "findings": [
                    {
                        "category": "Requirement Consistency",
                        "severity": "ERROR",
                        "issue": "Data retention vs immediate deletion conflict",
                        "explanation": "REQ-001 mandates permanent deletion upon request, but REQ-024 requires 7-year immutable retention.",
                        "affected_sections": ["6. High-Level Business Requirements by Module"],
                        "evidence": "REQ-001: 'immediate deletion' vs REQ-024: 'retain for 7 years'",
                        "required_change": "Clarify compliance retention override rules for user deletion requests.",
                    }
                ],
                "rework_feedback": "Clarify conflicting requirements REQ-001 and REQ-024.",
            }
        elif self._mode == "terminology_inconsistency":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Inconsistent terminology used across sections without definitions.",
                "findings": [
                    {
                        "category": "Terminology Consistency",
                        "severity": "WARNING",
                        "issue": "Synonymous terms used ambiguously",
                        "explanation": "Document refers to 'Customer' in Section 2, 'Client' in Section 5, and 'Account Holder' in Section 8.",
                        "affected_sections": ["2. Business Context", "5. Stakeholders & Personas", "8. Integrations"],
                        "evidence": "Customer vs Client vs Account Holder",
                        "required_change": "Standardize primary actor terminology across all sections.",
                    }
                ],
                "rework_feedback": "Standardize primary customer terminology across Sections 2, 5, and 8.",
            }
        elif self._mode == "grounding_unsupported":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Unsupported blockchain claim detected in Section 8.",
                "findings": [
                    {
                        "category": "Grounding",
                        "severity": "ERROR",
                        "issue": "Unsupported architecture claim",
                        "explanation": "Section 8 specifies an Ethereum consortium blockchain integration, which is not present in project discovery notes.",
                        "affected_sections": ["8. Integrations"],
                        "evidence": "Ethereum consortium blockchain",
                        "required_change": "Remove or ground blockchain requirement in verified project evidence.",
                    }
                ],
                "rework_feedback": "Remove ungrounded Ethereum blockchain integration.",
            }
        elif self._mode == "completeness_gap":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Document-level completeness gap detected: payment processing without gateway integration.",
                "findings": [
                    {
                        "category": "Completeness",
                        "severity": "ERROR",
                        "issue": "Orphaned payment workflow",
                        "explanation": "Section 6 mandates credit card payment processing, but Section 8 lists no payment gateway integrations.",
                        "affected_sections": ["6. High-Level Business Requirements by Module", "8. Integrations"],
                        "evidence": "Payment requirement without integration touchpoint",
                        "required_change": "Add payment gateway integration details in Section 8.",
                    }
                ],
                "rework_feedback": "Add required payment gateway integration to Section 8.",
            }
        elif self._mode == "duplication":
            payload = {
                "outcome": "NEEDS_REWORK",
                "summary": "Redundant requirements found describing identical functionality.",
                "findings": [
                    {
                        "category": "Duplication",
                        "severity": "WARNING",
                        "issue": "Duplicate authentication requirement",
                        "explanation": "REQ-101 and REQ-204 both specify Okta SAML 2.0 SSO identically.",
                        "affected_sections": ["6. High-Level Business Requirements by Module"],
                        "evidence": "REQ-101 and REQ-204 duplicate Okta SSO specification",
                        "required_change": "Consolidate duplicate authentication requirements.",
                    }
                ],
                "rework_feedback": "Consolidate redundant requirements REQ-101 and REQ-204.",
            }
        elif self._mode == "raw_valid_text":
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="VALID: Document is coherent."))])
        elif self._mode == "raw_invalid_text":
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="The document contains contradictions."))])
        else:
            # Default: valid coherent document
            payload = {
                "outcome": "VALID",
                "summary": "Complete assembled BRD is coherent, consistent, well-grounded, and meets template standards.",
                "findings": [],
                "rework_feedback": None,
            }

        import json
        content = f"```json\n{json.dumps(payload)}\n```"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def sample_assembled_brd() -> str:
    return (
        "# 1. Purpose & Scope of This Document\n\n"
        "This High-Level Business Requirements Document outlines the core scope for Dexmiq Claims System.\n\n"
        "# 2. Business Context (Summary from Discovery)\n\n"
        "Current manual claims adjudication takes 14 days. Objective is automated straight-through processing.\n\n"
        "# 3. In-Scope Business Modules & Feature Groups\n\n"
        "| Module ID | Module Name | Description |\n"
        "|---|---|---|\n"
        "| HL-MOD-1 | Claims Intake | Captures claims from web portal |\n"
        "| HL-MOD-2 | Adjudication Engine | Automated rule-based adjudication |\n\n"
        "# 5. Stakeholders & Personas\n\n"
        "## 5.1 Personas\n"
        "- Claims Officer: Evaluates flagged claims.\n"
        "- Policyholder: Submits new claims.\n\n"
        "# 6. High-Level Business Requirements by Module\n\n"
        "| HL Requirement ID | Description | Related Persona |\n"
        "|---|---|---|\n"
        "| HL-BRD-1.1 | Policyholder submits claim online | Policyholder |\n"
        "| HL-BRD-2.1 | Engine auto-adjudicates under $500 | Claims Officer |\n\n"
        "# 8. Integrations\n\n"
        "| Integration ID | Source System | Target System | Purpose |\n"
        "|---|---|---|---|\n"
        "| INT-01 | Core Policy DB | Claims System | Verify policy coverage |"
    )


@pytest.fixture
def sample_project_evidence() -> list[dict[str, str]]:
    return [
        {"source": "discovery_transcript", "content": "Dexmiq insurance claims automation. Target under $500 auto-approval."},
        {"source": "policy_api_spec", "content": "REST API v2 for Core Policy verification."},
    ]


# ---------------------------------------------------------------------------
# Unit Tests: System Instruction & Configuration Invariants
# ---------------------------------------------------------------------------


def test_system_instruction_loads_properly() -> None:
    content = load_final_validation_system_instruction()
    assert "BRD Final Validation Sub-Agent" in content
    assert "Cross-Section Consistency" in content
    assert "VALID" in content and "NEEDS_REWORK" in content


def test_final_validation_agent_has_no_tools() -> None:
    """Strict invariant: Final Validation Sub-Agent must have NO tools."""
    mock_model = MockFinalValidationChatModel()
    agent = BRDFinalValidationAgent(model=mock_model)
    assert agent.tools == []
    assert len(agent.tools) == 0


# ---------------------------------------------------------------------------
# Unit Tests: Validation Dimensions & Scenarios
# ---------------------------------------------------------------------------


def test_valid_complete_brd(sample_assembled_brd: str, sample_project_evidence: list[dict[str, str]]) -> None:
    mock_model = MockFinalValidationChatModel(mode="valid")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(
        assembled_document=sample_assembled_brd,
        available_project_information=sample_project_evidence,
    )
    result = agent.validate(ctx)

    assert result.is_valid is True
    assert result.needs_rework is False
    assert result.outcome == FinalValidationOutcome.VALID
    assert len(result.findings) == 0
    assert result.rework_feedback is None


def test_cross_section_contradiction_detection(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="cross_section_contradiction")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.is_valid is False
    assert result.needs_rework is True
    assert result.outcome == FinalValidationOutcome.NEEDS_REWORK
    assert len(result.findings) == 1

    finding = result.findings[0]
    assert finding.category == FinalValidationCategory.CROSS_SECTION_CONSISTENCY
    assert "3. In-Scope Business Modules & Feature Groups" in finding.affected_sections
    assert "7. Conceptual Business Workflows" in finding.affected_sections
    assert len(result.affected_sections) == 2
    assert "user scale" in finding.issue.lower()


def test_requirement_conflict_detection(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="requirement_conflict")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.is_valid is False
    assert result.outcome == FinalValidationOutcome.NEEDS_REWORK
    assert len(result.findings) == 1
    assert result.findings[0].category == FinalValidationCategory.REQUIREMENT_CONSISTENCY
    assert "REQ-001" in result.findings[0].explanation


def test_terminology_inconsistency_detection(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="terminology_inconsistency")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.needs_rework is True
    assert result.findings[0].category == FinalValidationCategory.TERMINOLOGY_CONSISTENCY
    assert result.findings[0].severity == FinalValidationSeverity.WARNING
    assert "Customer" in result.findings[0].explanation


def test_grounding_unsupported_claim(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="grounding_unsupported")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.needs_rework is True
    assert result.findings[0].category == FinalValidationCategory.GROUNDING
    assert "Ethereum" in result.findings[0].explanation


def test_completeness_gap_detection(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="completeness_gap")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.needs_rework is True
    assert result.findings[0].category == FinalValidationCategory.COMPLETENESS
    assert "payment" in result.findings[0].issue.lower()


def test_duplication_detection(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="duplication")
    agent = BRDFinalValidationAgent(model=mock_model)

    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert result.needs_rework is True
    assert result.findings[0].category == FinalValidationCategory.DUPLICATION
    assert "duplicate" in result.findings[0].issue.lower()


def test_document_is_not_rewritten(sample_assembled_brd: str) -> None:
    """Strict invariant: Assembled document must NEVER be modified by final validation."""
    mock_model = MockFinalValidationChatModel(mode="cross_section_contradiction")
    agent = BRDFinalValidationAgent(model=mock_model)

    original_doc = str(sample_assembled_brd)
    ctx = FinalValidationContext(assembled_document=sample_assembled_brd)
    result = agent.validate(ctx)

    assert ctx.assembled_document == original_doc
    assert sample_assembled_brd == original_doc
    assert not hasattr(result, "assembled_document")  # Result only holds outcome and findings


# ---------------------------------------------------------------------------
# Unit Tests: Fallback Parsing & Safety Invariants
# ---------------------------------------------------------------------------


def test_fallback_parsing_valid_text(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="raw_valid_text")
    agent = BRDFinalValidationAgent(model=mock_model)

    result = agent.validate(FinalValidationContext(assembled_document=sample_assembled_brd))
    assert result.is_valid is True
    assert result.metadata.get("fallback_parsing") is True


def test_fallback_parsing_invalid_text(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="raw_invalid_text")
    agent = BRDFinalValidationAgent(model=mock_model)

    result = agent.validate(FinalValidationContext(assembled_document=sample_assembled_brd))
    assert result.is_valid is False
    assert result.outcome == FinalValidationOutcome.NEEDS_REWORK
    assert result.metadata.get("fallback_parsing") is True


def test_validator_runtime_error_never_produces_valid(sample_assembled_brd: str) -> None:
    """Invariant: A runtime execution failure must return NEEDS_REWORK, never VALID."""
    mock_model = MockFinalValidationChatModel(mode="error")
    agent = BRDFinalValidationAgent(model=mock_model)

    result = agent.validate(FinalValidationContext(assembled_document=sample_assembled_brd))
    assert result.is_valid is False
    assert result.outcome == FinalValidationOutcome.NEEDS_REWORK
    assert "error" in result.summary.lower()
    assert len(result.findings) == 1
    assert "error" in result.metadata


# ---------------------------------------------------------------------------
# Unit Tests: State Integration & Persistence
# ---------------------------------------------------------------------------


def test_state_serialization_with_final_validation(sample_assembled_brd: str) -> None:
    state = BRDAgentState()
    state.set_assembled_brd(sample_assembled_brd)

    finding = FinalValidationFinding(
        category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Contradiction",
        explanation="Contradiction detail",
        affected_sections=["Section 1", "Section 2"],
        required_change="Fix inconsistency",
    )
    val_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Requires fixes",
        findings=[finding],
        rework_feedback="Fix Section 1 and 2.",
    )
    state.set_final_validation_result(val_res)

    serialized = state.to_dict()
    assert "latest_final_validation_result" in serialized
    assert serialized["latest_final_validation_result"]["outcome"] == "NEEDS_REWORK"
    assert "final_validation_history" in serialized
    assert len(serialized["final_validation_history"]) == 1

    deserialized = BRDAgentState.from_dict(serialized)
    assert deserialized.latest_final_validation_result is not None
    assert deserialized.latest_final_validation_result.outcome == FinalValidationOutcome.NEEDS_REWORK
    assert deserialized.latest_final_validation_result.findings[0].category == FinalValidationCategory.CROSS_SECTION_CONSISTENCY
    assert deserialized.latest_final_validation_result.affected_sections == ["Section 1", "Section 2"]
    assert len(deserialized.final_validation_history) == 1


# ---------------------------------------------------------------------------
# Unit Tests: Lead Agent Integration
# ---------------------------------------------------------------------------


def test_lead_agent_validate_final_brd_sync(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="valid")
    lead = create_brd_lead_agent(model=mock_model)
    lead.state.set_assembled_brd(sample_assembled_brd)

    result = lead.validate_final_brd(context=AgentContext(project_id="lead_proj_1"))

    assert result.is_valid is True
    assert lead.latest_final_validation_result is result
    assert len(lead.state.final_validation_history) == 1


@pytest.mark.asyncio
async def test_lead_agent_validate_final_brd_async(sample_assembled_brd: str) -> None:
    mock_model = MockFinalValidationChatModel(mode="valid")
    lead = create_brd_lead_agent(model=mock_model)
    lead.state.set_assembled_brd(sample_assembled_brd)

    result = await lead.validate_final_brd_async(context=AgentContext(project_id="lead_proj_async"))

    assert result.is_valid is True
    assert lead.latest_final_validation_result is result
    assert len(lead.state.final_validation_history) == 1


def test_lead_agent_fails_when_no_assembled_brd() -> None:
    """Lead Agent must refuse to perform final validation if assembled BRD is missing."""
    mock_model = MockFinalValidationChatModel(mode="valid")
    lead = create_brd_lead_agent(model=mock_model)
    # assembled_brd is None!

    with pytest.raises(ValueError, match="no assembled BRD document is available"):
        lead.validate_final_brd()


def test_lead_agent_fails_when_assembled_brd_empty() -> None:
    mock_model = MockFinalValidationChatModel(mode="valid")
    lead = create_brd_lead_agent(model=mock_model)
    lead.state.set_assembled_brd("   \n\t  ")

    with pytest.raises(ValueError, match="no assembled BRD document is available"):
        lead.validate_final_brd()


# ---------------------------------------------------------------------------
# Unit Tests: Lifecycle Logging
# ---------------------------------------------------------------------------


def test_final_validation_lifecycle_logging_valid(
    sample_assembled_brd: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_model = MockFinalValidationChatModel(mode="valid")
    lead = create_brd_lead_agent(model=mock_model)
    lead.state.set_assembled_brd(sample_assembled_brd)

    with caplog.at_level(logging.INFO):
        lead.validate_final_brd(context=AgentContext(project_id="log_project_valid"))

    log_text = caplog.text
    assert "BRD final validation started" in log_text
    assert "BRD final validation completed" in log_text
    assert "log_project_valid" in log_text


def test_final_validation_lifecycle_logging_rework(
    sample_assembled_brd: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_model = MockFinalValidationChatModel(mode="cross_section_contradiction")
    lead = create_brd_lead_agent(model=mock_model)
    lead.state.set_assembled_brd(sample_assembled_brd)

    with caplog.at_level(logging.INFO):
        lead.validate_final_brd(context=AgentContext(project_id="log_project_rework"))

    log_text = caplog.text
    assert "BRD final validation started" in log_text
    assert "BRD final validation requires rework" in log_text
    assert "log_project_rework" in log_text
