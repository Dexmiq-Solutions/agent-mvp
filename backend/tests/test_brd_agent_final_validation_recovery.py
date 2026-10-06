"""Tests for BRD Final Validation -> BRD Rewriter Architecture.

Comprehensive test suite verifying the new Phase 7 & 8 workflow:
1. Final Validation VALID requires no rewriter call.
2. Final Validation NEEDS_REWORK invokes BRDRewriterAgent.
3. Narrowed Evidence Authority: Rewriter receives only findings and referenced evidence.
4. Minimal Change Semantics: Edits applied minimally to assembled_brd.
5. Downstream State Synchronization: Section content updated without reopening sections.
6. Section Validation Invariant: Section Validation is NEVER called after Phase 5.
7. Two-Pass Final Validation: Pass 1 -> Rewriter -> Pass 2.
8. Pass 2 VALID completes and delivers document.
9. Pass 2 NEEDS_REWORK terminates correction cycle immediately (no loops).
10. Unresolved findings converted into numbered ## Open Questions / Clarifications and appended to final BRD.
11. Bounded Limit: At most 1 Rewriter call and at most 2 Final Validation calls.
12. State Serialization / Deserialization of rewriter result and open questions.
13. Async execution parity.
14. Full workflow integration in run_workflow_async.
15. Streaming progress event parity in stream_workflow_async.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDFinalValidationAgent,
    BRDLeadAgent,
    BRDRewriterAgent,
    BRDRewriterContext,
    BRDRewriterResult,
    BRDSectionGenerationAgent,
    BRDSectionStatus,
    BRDSectionValidationAgent,
    DocumentEdit,
    FinalValidationCategory,
    FinalValidationContext,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationRecoveryResult,
    FinalValidationResult,
    FinalValidationSeverity,
    FindingResolutionStatus,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
    SectionValidationContext,
    ValidationOutcome,
    ValidationResult,
    create_brd_lead_agent,
)
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.context import AgentContext


# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------

class MockChatModel(BaseChatModel):
    """Deterministic mock chat model."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Mock response"))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


class MockFinalValidator:
    """Mock final validator returning configured outcomes in order."""

    def __init__(self, outcomes: Optional[list[FinalValidationResult]] = None) -> None:
        self.calls: list[FinalValidationContext] = []
        self._outcomes: list[FinalValidationResult] = list(outcomes or [])
        self._default_valid = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="Complete BRD satisfies all cross-section validation gates.",
            findings=[],
        )

    def validate(self, context: FinalValidationContext) -> FinalValidationResult:
        self.calls.append(context)
        if self._outcomes:
            return self._outcomes.pop(0)
        return self._default_valid

    async def validate_async(self, context: FinalValidationContext) -> FinalValidationResult:
        return self.validate(context)


class MockRewriter:
    """Mock BRD Rewriter tracking calls and returning configured edits."""

    def __init__(self, result: Optional[BRDRewriterResult] = None) -> None:
        self.calls: list[BRDRewriterContext] = []
        self._result = result or BRDRewriterResult(
            summary="Applied 1 targeted edit",
            edits=[
                DocumentEdit(
                    finding_id="FV-001",
                    target_location="Section 1",
                    original_fragment="Original text fragment",
                    corrected_fragment="Corrected replacement text",
                    explanation="Applied fix",
                )
            ],
            unapplied_findings=[],
        )

    def rewrite(self, context: BRDRewriterContext) -> BRDRewriterResult:
        self.calls.append(context)
        return self._result

    async def rewrite_async(self, context: BRDRewriterContext) -> BRDRewriterResult:
        return self.rewrite(context)


class MockSectionValidator:
    """Mock section validator to verify Section Validation is never called in Phase 8."""

    def __init__(self) -> None:
        self.calls: list[SectionValidationContext] = []

    def validate(self, context: SectionValidationContext) -> ValidationResult:
        self.calls.append(context)
        return ValidationResult(outcome=ValidationOutcome.VALID)

    async def validate_async(self, context: SectionValidationContext) -> ValidationResult:
        return self.validate(context)


def make_test_lead_agent(
    final_validator: Optional[MockFinalValidator] = None,
    rewriter: Optional[MockRewriter] = None,
    section_validator: Optional[MockSectionValidator] = None,
) -> BRDLeadAgent:
    """Helper to instantiate a BRDLeadAgent with mock sub-agents."""
    fv = final_validator or MockFinalValidator()
    rw = rewriter or MockRewriter()
    sv = section_validator or MockSectionValidator()
    return BRDLeadAgent(
        model=MockChatModel(),
        final_validator=fv,  # type: ignore
        rewriter=rw,  # type: ignore
        section_validator=sv,  # type: ignore
    )


# ---------------------------------------------------------------------------
# Tests: Final Validation & BRD Rewriter Architecture
# ---------------------------------------------------------------------------

def test_final_validation_valid_requires_no_rewriter():
    """1. When Final Validation is VALID on Pass 1, rewriter is never invoked."""
    mock_fv = MockFinalValidator([
        FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="BRD is completely valid.",
            findings=[],
        )
    ])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    # Populate assembled BRD
    agent.state.set_assembled_brd("# 1. Project Overview\n\nContent is valid.")

    res = agent.recover_final_validation()
    assert res.is_valid is True
    assert res.recovery_cycles == 0
    assert res.exhausted is False
    assert len(mock_rw.calls) == 0
    assert len(mock_fv.calls) == 1


def test_final_validation_needs_rework_invokes_rewriter():
    """2. When Pass 1 returns NEEDS_REWORK, BRDRewriterAgent is invoked."""
    finding = FinalValidationFinding(
        finding_id="FV-001",
        category=FinalValidationCategory.CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Inconsistent term",
        location="Section 1",
        problematic_content="Original text fragment",
        evidence="Glossary specifies Corrected replacement text",
        required_correction="Change to Corrected replacement text",
    )
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Found 1 inconsistency",
        findings=[finding],
    )
    pass2_res = FinalValidationResult(
        outcome=FinalValidationOutcome.VALID,
        summary="Document is now valid.",
        findings=[],
    )
    mock_fv = MockFinalValidator([pass1_res, pass2_res])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    initial_doc = "# 1. Project Overview\n\nOriginal text fragment here."
    agent.state.set_assembled_brd(initial_doc)

    res = agent.recover_final_validation()
    assert len(mock_rw.calls) == 1
    assert len(mock_fv.calls) == 2  # Pass 1 and Pass 2
    assert res.is_valid is True
    assert agent.state.final_validation_recovery_cycles == 1


def test_rewriter_receives_only_findings_referenced_evidence():
    """3. Narrowed Evidence Authority: Rewriter context receives ONLY evidence attached to findings."""
    finding = FinalValidationFinding(
        finding_id="FV-002",
        category=FinalValidationCategory.GROUNDING,
        severity=FinalValidationSeverity.ERROR,
        issue="SLA mismatch",
        location="Section 2",
        evidence="Service Contract: 99.9% uptime",
        required_correction="Update uptime to 99.9%",
    )
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Unreferenced SLA",
        findings=[finding],
    )
    mock_fv = MockFinalValidator([pass1_res, FinalValidationResult(outcome=FinalValidationOutcome.VALID)])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    # Provide broad state evidence
    agent.state.add_evidence({"source": "broad_raw_notes.txt", "content": "Unrelated meeting notes"})
    agent.state.set_assembled_brd("# 1. Overview\n\nUptime is 95%.")

    agent.recover_final_validation()

    assert len(mock_rw.calls) == 1
    rewriter_ctx = mock_rw.calls[0]
    # Check that finding_evidence contains the finding's attached evidence
    assert len(rewriter_ctx.finding_evidence) == 1
    assert rewriter_ctx.finding_evidence[0]["finding_id"] == "FV-002"
    assert "99.9% uptime" in rewriter_ctx.finding_evidence[0]["evidence"]
    # Check that broad_raw_notes was NOT dumped as finding_evidence
    assert not any("broad_raw_notes" in str(e) for e in rewriter_ctx.finding_evidence)


def test_rewriter_edits_applied_minimally_to_assembled_brd():
    """4. Minimal Change Semantics: apply_rewriter_edits modifies only targeted verbatim text."""
    agent = make_test_lead_agent()
    original_doc = (
        "# 1. Purpose & Scope\n\n"
        "This project delivers an automated gateway.\n\n"
        "Target response latency is 500ms.\n\n"
        "# 2. Functional Requirements\n\n"
        "FR-1: User can authenticate via OAuth2."
    )
    edits = [
        DocumentEdit(
            finding_id="FV-010",
            target_location="Section 1",
            original_fragment="Target response latency is 500ms.",
            corrected_fragment="Target response latency is 200ms.",
            explanation="Aligned with performance baseline",
        )
    ]
    updated_doc, failed_ids = agent.apply_rewriter_edits(original_doc, edits)
    assert failed_ids == []
    assert "Target response latency is 200ms." in updated_doc
    assert "Target response latency is 500ms." not in updated_doc
    # Untouched text must remain identical
    assert "This project delivers an automated gateway." in updated_doc
    assert "FR-1: User can authenticate via OAuth2." in updated_doc


def test_section_content_synchronized_downstream_without_reopening_sections():
    """5. State Synchronization: section_content updated from assembled_brd for persistence."""
    agent = make_test_lead_agent()
    # Assume sections from agent.sections
    first_sec = agent.sections[0]
    second_sec = agent.sections[1]

    doc = f"# {first_sec}\n\nEdited content for section 1.\n\n# {second_sec}\n\nContent for section 2."
    agent.synchronize_section_content_from_assembled_brd(doc)

    assert first_sec in agent.state.section_content
    assert "Edited content for section 1." in agent.state.section_content[first_sec]
    assert second_sec in agent.state.section_content
    assert "Content for section 2." in agent.state.section_content[second_sec]


def test_section_validation_never_called_after_phase_5():
    """6. Crucial Invariant: Section Validation is NEVER executed in Phase 8."""
    finding = FinalValidationFinding(
        finding_id="FV-003",
        category=FinalValidationCategory.CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Inconsistency",
        location="Section 1",
        evidence="Evidence",
        required_correction="Fix",
    )
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[finding],
    )
    pass2_res = FinalValidationResult(
        outcome=FinalValidationOutcome.VALID,
        summary="Passed Pass 2",
    )
    mock_fv = MockFinalValidator([pass1_res, pass2_res])
    mock_rw = MockRewriter()
    mock_sv = MockSectionValidator()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw, section_validator=mock_sv)

    agent.state.set_assembled_brd("# Title\n\nOriginal text fragment")

    res = agent.recover_final_validation()
    assert res.is_valid is True
    # Section validator must NOT have been called
    assert len(mock_sv.calls) == 0


def test_pass2_valid_completes_successfully():
    """7. When Pass 2 Final Validation is VALID, delivery succeeds without Open Questions."""
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[FinalValidationFinding(finding_id="FV-004", issue="Minor issue", location="Section 1")],
    )
    pass2_res = FinalValidationResult(
        outcome=FinalValidationOutcome.VALID,
        summary="Pass 2 is fully valid.",
    )
    mock_fv = MockFinalValidator([pass1_res, pass2_res])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    agent.state.set_assembled_brd("# Doc\n\nOriginal text fragment")
    res = agent.recover_final_validation()

    assert res.is_valid is True
    assert res.exhausted is False
    assert agent.state.final_validation_recovery_exhausted is False
    assert "Open Questions / Clarifications" not in agent.state.assembled_brd


def test_pass2_needs_rework_terminates_immediately_and_appends_open_questions():
    """8. When Pass 2 is NEEDS_REWORK, cycle terminates immediately and appends Open Questions."""
    finding_unresolved = FinalValidationFinding(
        finding_id="FV-999",
        category=FinalValidationCategory.COMPLETENESS,
        severity=FinalValidationSeverity.ERROR,
        issue="Unresolved architectural choice",
        location="Section 3",
        problematic_content="TBD cloud architecture",
        required_correction="Specify whether AWS or Azure is chosen",
        resolution_status=FindingResolutionStatus.OPEN_QUESTION,
        open_question="Which cloud provider (AWS or Azure) should host the production workload?",
    )
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework",
        findings=[finding_unresolved],
    )
    pass2_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Still unresolved",
        findings=[finding_unresolved],
    )
    mock_fv = MockFinalValidator([pass1_res, pass2_res])
    mock_rw = MockRewriter(
        BRDRewriterResult(
            summary="Unable to resolve open question without user input",
            edits=[],
            unapplied_findings=["FV-999"],
        )
    )
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    agent.state.set_assembled_brd("# System Architecture\n\nTBD cloud architecture")
    res = agent.recover_final_validation()

    # Recovery must terminate immediately after Pass 2
    assert res.is_valid is False
    assert res.exhausted is True
    assert agent.state.final_validation_recovery_exhausted is True
    assert len(mock_rw.calls) == 1
    assert len(mock_fv.calls) == 2

    # Open Questions must be stored in state and appended to assembled_brd
    assert len(agent.open_questions) > 0
    assert "FV-999" in agent.open_questions[0]
    assert "## Open Questions / Clarifications" in agent.state.assembled_brd
    assert "Which cloud provider (AWS or Azure)" in agent.state.assembled_brd


def test_bounded_limit_maximum_one_rewriter_invocation():
    """9. Strict invariant: At most 1 Rewriter invocation per workflow run."""
    pass1 = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Rework",
        findings=[FinalValidationFinding(finding_id="F-1", issue="Problem", location="Sec 1")],
    )
    pass2 = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Still rework",
        findings=[FinalValidationFinding(finding_id="F-1", issue="Problem", location="Sec 1")],
    )
    mock_fv = MockFinalValidator([pass1, pass2])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    agent.state.set_assembled_brd("# BRD\n\nOriginal text fragment")
    agent.recover_final_validation()

    assert len(mock_rw.calls) == 1
    assert agent.state.final_validation_recovery_cycles == 1


def test_state_serialization_preserves_rewriter_and_open_questions():
    """10. State serialization / deserialization preserves latest_rewriter_result and open_questions."""
    state = BRDAgentState.initialize_from_template(sections=["1. Overview"])
    rewriter_res = BRDRewriterResult(
        summary="Applied targeted replacement",
        edits=[
            DocumentEdit(
                finding_id="FV-005",
                target_location="Section 1",
                original_fragment="Old",
                corrected_fragment="New",
                explanation="Fix",
            )
        ],
        unapplied_findings=["FV-006"],
    )
    state.set_rewriter_result(rewriter_res)
    state.set_open_questions(["**[FV-006]**: Clarification question text"])

    d = state.to_dict()
    assert "latest_rewriter_result" in d
    assert "open_questions" in d
    assert d["latest_rewriter_result"]["summary"] == "Applied targeted replacement"
    assert len(d["open_questions"]) == 1

    restored = BRDAgentState.from_dict(d)
    assert restored.latest_rewriter_result is not None
    assert restored.latest_rewriter_result.summary == "Applied targeted replacement"
    assert len(restored.latest_rewriter_result.edits) == 1
    assert restored.latest_rewriter_result.edits[0].finding_id == "FV-005"
    assert len(restored.open_questions) == 1
    assert "FV-006" in restored.open_questions[0]


@pytest.mark.asyncio
async def test_async_recovery_parity():
    """11. Async recovery parity: recover_final_validation_async operates identically."""
    finding = FinalValidationFinding(
        finding_id="FV-007",
        category=FinalValidationCategory.CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Issue",
        location="Section 1",
        problematic_content="Original text fragment",
        evidence="Evidence",
        required_correction="Corrected replacement text",
    )
    mock_fv = MockFinalValidator([
        FinalValidationResult(outcome=FinalValidationOutcome.NEEDS_REWORK, findings=[finding]),
        FinalValidationResult(outcome=FinalValidationOutcome.VALID),
    ])
    mock_rw = MockRewriter()
    agent = make_test_lead_agent(final_validator=mock_fv, rewriter=mock_rw)

    agent.state.set_assembled_brd("# Doc\n\nOriginal text fragment")
    res = await agent.recover_final_validation_async()

    assert res.is_valid is True
    assert len(mock_rw.calls) == 1
    assert len(mock_fv.calls) == 2
    assert "Corrected replacement text" in agent.state.assembled_brd


@pytest.mark.asyncio
async def test_run_workflow_async_integrates_rewriter_and_open_questions():
    """12. Full workflow test: run_workflow_async executes Phase 7 -> Rewriter -> Phase 7 Pass 2."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = AgentContext(project_id="test-proj")

    # Mock all phases up to assembly
    for sec in agent.sections:
        agent.state.set_section_content(sec, f"# {sec}\nContent for {sec}")
        agent.state.update_section_status(sec, BRDSectionStatus.COMPLETED)

    finding = FinalValidationFinding(
        finding_id="FV-888",
        category=FinalValidationCategory.REQUIREMENT_CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Unresolved spec gap",
        location="Section 1",
        open_question="Confirm SLA threshold (200ms vs 500ms)?",
    )
    pass1_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[finding],
    )
    pass2_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[finding],
    )

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_asm, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "rewrite_brd_async", new_callable=AsyncMock) as mock_rw:

        from agents.brd.agent import ActionDecision, ActionType, GapResolutionAction, GapResolutionDecision
        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK, reasoning="Direct")
        mock_eval.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION)
        mock_asm.return_value = BRDAssemblyResult(
            assembled_document="# 1. Overview\nContent for 1. Overview",
            assembly_complete=True,
        )
        mock_val.side_effect = [pass1_res, pass2_res]
        mock_rw.return_value = BRDRewriterResult(
            summary="Could not resolve without user input",
            edits=[],
            unapplied_findings=["FV-888"],
        )

        resp = await agent.run_workflow_async(
            objective="Build gateway",
            context=ctx,
        )

        assert mock_rw.await_count == 1
        assert mock_val.await_count == 2
        # Document delivered with Open Questions
        assert "## Open Questions / Clarifications" in resp.output_text
        assert "Confirm SLA threshold" in resp.output_text
        assert agent.state.final_validation_recovery_exhausted is True


@pytest.mark.asyncio
async def test_stream_workflow_async_emits_rewriter_events():
    """13. stream_workflow_async yields progress events for BRD Rewriter and Pass 2."""
    agent = BRDLeadAgent(model=MockChatModel())
    ctx = AgentContext(project_id="test-proj")

    for sec in agent.sections:
        agent.state.set_section_content(sec, f"# {sec}\nContent for {sec}")
        agent.state.update_section_status(sec, BRDSectionStatus.COMPLETED)

    finding = FinalValidationFinding(
        finding_id="FV-777",
        issue="Inconsistency",
        location="Section 1",
        problematic_content="Old fragment",
        required_correction="New fragment",
    )
    pass1 = FinalValidationResult(outcome=FinalValidationOutcome.NEEDS_REWORK, findings=[finding])
    pass2 = FinalValidationResult(outcome=FinalValidationOutcome.VALID, findings=[])

    with patch.object(agent, "decide_action_async", new_callable=AsyncMock) as mock_decide, \
         patch.object(agent, "interpret_evaluation_async", new_callable=AsyncMock) as mock_eval, \
         patch.object(agent, "assemble_brd_async", new_callable=AsyncMock) as mock_asm, \
         patch.object(agent, "validate_final_brd_async", new_callable=AsyncMock) as mock_val, \
         patch.object(agent, "rewrite_brd_async", new_callable=AsyncMock) as mock_rw:

        from agents.brd.agent import ActionDecision, ActionType, GapResolutionAction, GapResolutionDecision
        mock_decide.return_value = ActionDecision(action_type=ActionType.DIRECT_WORK)
        mock_eval.return_value = GapResolutionDecision(action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION)
        mock_asm.return_value = BRDAssemblyResult(
            assembled_document="# Title\n\nOld fragment",
            assembly_complete=True,
        )
        mock_val.side_effect = [pass1, pass2]
        mock_rw.return_value = BRDRewriterResult(
            summary="Applied edit",
            edits=[DocumentEdit(finding_id="FV-777", target_location="Section 1", original_fragment="Old fragment", corrected_fragment="New fragment")],
        )

        events: list[dict[str, Any]] = []
        async for event in agent.stream_workflow_async(
            objective="Build gateway",
            context=ctx,
        ):
            events.append(event)

        progress_phases = [e.get("phase") for e in events if e.get("type") == "progress"]
        assert "7_FINAL_VALIDATION" in progress_phases
        assert "8_BRD_REWRITER" in progress_phases
        assert "9_COMPLETION" in progress_phases
