"""Tests for BRD Final Validation recovery loop.

Covers:
1. Final Validation VALID requires no recovery.
2. Final Validation NEEDS_REWORK identifies an affected section.
3. The affected section is passed to Section Generation/Update.
4. Final Validation feedback is passed into the section update.
5. Updated sections are passed through Section Validation.
6. A valid updated section proceeds to assembly.
7. The assembled BRD is passed through Final Validation again.
8. Multiple affected sections can be reworked in one recovery cycle.
9. A subsequent Final Validation VALID completes successfully.
10. Recovery repeats when Final Validation continues to return NEEDS_REWORK.
11. Recovery stops after the maximum of 3 cycles.
12. Exhausted recovery does not mark the BRD as valid or complete.
13. Existing section-level retry behavior remains intact.
14. Recovery state/history is preserved correctly across serialization.
15. Meaningful recovery lifecycle logging occurs.
16. Errors fail safely and do not produce a false VALID result.
17. Async recovery parity.
18. Edge case: unresolvable affected sections stops safely.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import pytest

from agents.brd import (
    BRDAgentState,
    BRDFinalValidationAgent,
    BRDLeadAgent,
    BRDSectionGenerationAgent,
    BRDSectionStatus,
    BRDSectionValidationAgent,
    FinalValidationCategory,
    FinalValidationContext,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationRecoveryResult,
    FinalValidationResult,
    FinalValidationSeverity,
    MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
    create_brd_lead_agent,
    format_section_rework_guidance,
    resolve_affected_sections,
)
from agents.brd.context import AgentContext


# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------


class MockSectionGenerator:
    """Mock section generator tracking generation calls and returning configurable content."""

    def __init__(self, prefix: str = "Updated content for ") -> None:
        self.calls: list[SectionGenerationContext] = []
        self.prefix = prefix
        self.custom_responses: dict[str, str] = {}

    def generate(self, context: SectionGenerationContext) -> SectionGenerationResult:
        self.calls.append(context)
        content = self.custom_responses.get(
            context.section_name,
            f"{self.prefix}{context.section_name}\n\nValidated requirements text.",
        )
        return SectionGenerationResult(
            section_name=context.section_name,
            content=content,
            operation=context.operation,
            metadata={"call_count": len(self.calls)},
        )

    async def generate_async(self, context: SectionGenerationContext) -> SectionGenerationResult:
        return self.generate(context)


class MockSectionValidator:
    """Mock section validator tracking validation calls and returning configured outcomes."""

    def __init__(self, outcomes: Optional[list[ValidationResult]] = None) -> None:
        self.calls: list[SectionValidationContext] = []
        self._outcomes: list[ValidationResult] = list(outcomes or [])
        self._default_valid = ValidationResult(
            outcome=ValidationOutcome.VALID,
            summary="Section content is valid.",
            findings=[],
        )

    def validate(self, context: SectionValidationContext) -> ValidationResult:
        self.calls.append(context)
        if self._outcomes:
            return self._outcomes.pop(0)
        return self._default_valid

    async def validate_async(self, context: SectionValidationContext) -> ValidationResult:
        return self.validate(context)


class MockFinalValidator:
    """Mock final validator tracking calls and returning configured FinalValidationResult sequence."""

    def __init__(self, outcomes: Optional[list[FinalValidationResult]] = None) -> None:
        self.calls: list[FinalValidationContext] = []
        self._outcomes: list[FinalValidationResult] = list(outcomes or [])
        self._default_valid = FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="Complete BRD document is valid.",
            findings=[],
        )

    def validate(self, context: FinalValidationContext) -> FinalValidationResult:
        self.calls.append(context)
        if self._outcomes:
            return self._outcomes.pop(0)
        return self._default_valid

    async def validate_async(self, context: FinalValidationContext) -> FinalValidationResult:
        return self.validate(context)


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


SAMPLE_TEMPLATE = (
    "# 1. Purpose & Scope of This Document\n\n"
    "Initial Purpose.\n\n"
    "# 2. Business Context (Summary from Discovery)\n\n"
    "Initial Business Context.\n\n"
    "# 3. In-Scope Business Modules & Feature Groups\n\n"
    "Initial Scope Modules.\n\n"
    "# 4. Out-of-Scope\n\n"
    "Initial Out of Scope.\n\n"
    "# 5. Stakeholders & Personas\n\n"
    "Initial Stakeholders.\n\n"
    "# 6. High-Level Business Requirements by Module\n\n"
    "Initial Requirements.\n\n"
    "# 7. Conceptual Business Workflows\n\n"
    "Initial Workflows.\n\n"
    "# 8. Integrations & External System Dependencies\n\n"
    "Initial Integrations.\n\n"
    "# 9. Business Data & Reporting Needs\n\n"
    "Initial Reporting.\n\n"
    "# 10. Non-Functional Business Expectations\n\n"
    "Initial Non-Functional.\n\n"
    "# 11. Assumptions & Open Questions\n\n"
    "Initial Assumptions.\n"
)


def create_test_lead_agent(
    final_outcomes: Optional[list[FinalValidationResult]] = None,
    section_outcomes: Optional[list[ValidationResult]] = None,
    template: str = SAMPLE_TEMPLATE,
) -> tuple[BRDLeadAgent, MockSectionGenerator, MockSectionValidator, MockFinalValidator]:
    """Helper to instantiate BRDLeadAgent with mock sub-agents and all sections populated."""
    sec_gen = MockSectionGenerator()
    sec_val = MockSectionValidator(section_outcomes)
    final_val = MockFinalValidator(final_outcomes)

    lead = create_brd_lead_agent(
        template=template,
        section_generator=sec_gen,  # type: ignore
        section_validator=sec_val,  # type: ignore
        final_validator=final_val,  # type: ignore
    )

    # Populate sections in state and mark all as COMPLETED
    for sec in lead.sections:
        lead.state.set_section_content(sec, f"Content of {sec}")
        lead.state.update_section_status(sec, BRDSectionStatus.COMPLETED)

    # Initial assembly
    lead.assemble_brd()
    return lead, sec_gen, sec_val, final_val


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_final_validation_valid_requires_no_recovery() -> None:
    """1. Final Validation VALID requires no recovery."""
    valid_res = FinalValidationResult(
        outcome=FinalValidationOutcome.VALID,
        summary="Document is coherent.",
        findings=[],
    )
    lead, sec_gen, sec_val, final_val = create_test_lead_agent([valid_res])

    # Run recovery
    rec_res = lead.recover_final_validation(context=AgentContext(project_id="p1"))

    assert rec_res.is_valid is True
    assert rec_res.recovery_cycles == 0
    assert rec_res.exhausted is False
    assert len(sec_gen.calls) == 0
    assert len(sec_val.calls) == 0
    assert len(final_val.calls) == 1
    assert lead.final_validation_recovery_cycles == 0
    assert lead.is_final_validation_recovery_exhausted is False


def test_final_validation_needs_rework_identifies_affected_section() -> None:
    """2. Final Validation NEEDS_REWORK identifies an affected section."""
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Contradiction in scope.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Scope mismatch",
                explanation="Section 3 does not align with Section 7.",
                affected_sections=["3. In-Scope Business Modules & Feature Groups"],
                required_change="Align module count.",
            )
        ],
        rework_feedback="Fix Section 3.",
    )
    valid_res = FinalValidationResult(
        outcome=FinalValidationOutcome.VALID,
        summary="Now valid.",
    )

    lead, sec_gen, sec_val, final_val = create_test_lead_agent([rework_res, valid_res])

    rec_res = lead.recover_final_validation(context=AgentContext(project_id="p2"))

    assert rec_res.is_valid is True
    assert rec_res.recovery_cycles == 1
    assert "3. In-Scope Business Modules & Feature Groups" in rec_res.reworked_sections


def test_affected_section_passed_to_section_generation_update() -> None:
    """3. The affected section is passed to Section Generation/Update."""
    target_sec = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Scope issue.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Missing module",
                explanation="Module X missing.",
                affected_sections=[target_sec],
                required_change="Add Module X.",
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res, valid_res])
    lead.recover_final_validation()

    assert len(sec_gen.calls) == 1
    assert sec_gen.calls[0].section_name == target_sec
    assert sec_gen.calls[0].operation == SectionOperation.UPDATE
    assert sec_gen.calls[0].existing_content == f"Content of {target_sec}"


def test_final_validation_feedback_passed_into_section_update() -> None:
    """4. Final Validation feedback is passed into the section update."""
    target_sec = "5. Stakeholders & Personas"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Stakeholder mismatch.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.COMPLETENESS,
                severity=FinalValidationSeverity.ERROR,
                issue="Missing Admin Persona",
                explanation="System Admin persona is mentioned in Section 8 but omitted in Section 5.",
                affected_sections=[target_sec],
                required_change="Define System Administrator persona with access privileges.",
            )
        ],
        rework_feedback="Consolidate persona roles.",
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res, valid_res])
    lead.recover_final_validation()

    gen_call = sec_gen.calls[0]
    assert gen_call.rework_feedback is not None
    assert "System Admin persona is mentioned in Section 8" in gen_call.rework_feedback
    assert "Define System Administrator persona" in gen_call.rework_feedback
    assert "Consolidate persona roles." in gen_call.rework_feedback


def test_updated_sections_passed_through_section_validation() -> None:
    """5. Updated sections are passed through Section Validation."""
    target_sec = "2. Business Context (Summary from Discovery)"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Context defect.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.GROUNDING,
                severity=FinalValidationSeverity.ERROR,
                issue="Ungrounded metrics",
                affected_sections=[target_sec],
                required_change="Ground claims.",
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, sec_val, _ = create_test_lead_agent([rework_res, valid_res])
    lead.recover_final_validation()

    assert len(sec_val.calls) == 1
    val_call = sec_val.calls[0]
    assert val_call.section_name == target_sec
    assert val_call.section_content == sec_gen.calls[0].existing_content or "Updated content" in val_call.section_content


def test_valid_updated_section_proceeds_to_assembly() -> None:
    """6. A valid updated section proceeds to assembly."""
    target_sec = "2. Business Context (Summary from Discovery)"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Context defect.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.GROUNDING,
                severity=FinalValidationSeverity.ERROR,
                issue="Ungrounded metrics",
                affected_sections=[target_sec],
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res, valid_res])
    lead.recover_final_validation()

    # Verify state assembled_brd contains the newly updated section content
    assembled = lead.assembled_brd
    assert assembled is not None
    assert f"Updated content for {target_sec}" in assembled


def test_assembled_brd_passed_through_final_validation_again() -> None:
    """7. The assembled BRD is passed through Final Validation again."""
    target_sec = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                issue="Contradiction",
                affected_sections=[target_sec],
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, _, _, final_val = create_test_lead_agent([rework_res, valid_res])
    lead.recover_final_validation()

    assert len(final_val.calls) == 2
    # Second call receives newly assembled document
    first_doc = final_val.calls[0].assembled_document
    second_doc = final_val.calls[1].assembled_document
    assert f"Updated content for {target_sec}" in second_doc


def test_multiple_affected_sections_reworked_in_one_cycle() -> None:
    """8. Multiple affected sections can be reworked in one recovery cycle."""
    sec1 = "2. Business Context (Summary from Discovery)"
    sec2 = "5. Stakeholders & Personas"

    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Multiple sections affected.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                issue="Inconsistency across context and personas",
                affected_sections=[sec1, sec2],
                required_change="Align roles across context and personas.",
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, sec_val, _ = create_test_lead_agent([rework_res, valid_res])
    rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is True
    assert rec_res.recovery_cycles == 1
    assert sec1 in rec_res.reworked_sections
    assert sec2 in rec_res.reworked_sections
    assert len(sec_gen.calls) == 2
    assert len(sec_val.calls) == 2
    assert {c.section_name for c in sec_gen.calls} == {sec1, sec2}


def test_subsequent_final_validation_valid_completes_successfully() -> None:
    """9. A subsequent Final Validation VALID completes successfully."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                issue="Issue in Section 3",
                affected_sections=[sec1],
            )
        ],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, _, _, _ = create_test_lead_agent([rework_res, valid_res])
    rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is True
    assert rec_res.needs_rework is False
    assert rec_res.exhausted is False
    assert lead.is_final_validation_recovery_exhausted is False


def test_recovery_repeats_when_final_validation_continues_to_return_needs_rework() -> None:
    """10. Recovery repeats when Final Validation continues to return NEEDS_REWORK."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res_1 = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue 1")],
    )
    rework_res_2 = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue 2")],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, _, final_val = create_test_lead_agent([rework_res_1, rework_res_2, valid_res])
    rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is True
    assert rec_res.recovery_cycles == 2
    assert len(final_val.calls) == 3
    assert len(sec_gen.calls) == 2


def test_recovery_stops_after_maximum_of_3_cycles() -> None:
    """11. Recovery stops after the maximum of 3 cycles."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Persistent defect.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Unresolved defect")],
    )

    # 4 NEEDS_REWORK results (initial + 3 recovery cycles)
    lead, sec_gen, _, final_val = create_test_lead_agent(
        [rework_res, rework_res, rework_res, rework_res]
    )
    rec_res = lead.recover_final_validation()

    assert rec_res.recovery_cycles == 3
    assert rec_res.exhausted is True
    assert len(sec_gen.calls) == 3
    assert len(final_val.calls) == 4  # Initial + 3 cycles


def test_exhausted_recovery_does_not_mark_brd_as_valid() -> None:
    """12. Exhausted recovery does not mark the BRD as valid or complete."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Still invalid.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Cannot resolve")],
    )

    lead, _, _, _ = create_test_lead_agent([rework_res] * 5)
    rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is False
    assert rec_res.needs_rework is True
    assert rec_res.exhausted is True
    assert lead.latest_final_validation_result is not None
    assert lead.latest_final_validation_result.is_valid is False
    assert lead.is_final_validation_recovery_exhausted is True


def test_existing_section_level_retry_behavior_remains_intact() -> None:
    """13. Existing section-level retry behavior remains intact."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue")],
    )
    # Section validator returns NEEDS_REWORK on first check, then VALID on section retry
    sec_needs_rework = ValidationResult(
        outcome=ValidationOutcome.NEEDS_REWORK,
        summary="Section needs rework.",
        findings=[
            ValidationFinding(
                category=ValidationCategory.COMPLETENESS,
                issue="Missing sub-points",
                explanation="Sub-points 3.1 and 3.2 are missing.",
                required_change="Add sub-points 3.1 and 3.2.",
            )
        ],
        rework_feedback="Add sub-points",
    )
    sec_valid = ValidationResult(outcome=ValidationOutcome.VALID)

    final_valid = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, sec_val, _ = create_test_lead_agent(
        final_outcomes=[rework_res, final_valid],
        section_outcomes=[sec_needs_rework, sec_valid],
    )
    rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is True
    # Section generator was called twice (initial section rework + section-level retry)
    assert len(sec_gen.calls) == 2
    # Section validator was called twice (initial validation + retry validation)
    assert len(sec_val.calls) == 2


def test_recovery_state_and_history_preserved_correctly() -> None:
    """14. Recovery state/history is preserved correctly across serialization."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res_1 = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Cycle 1 issue",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue 1")],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID, summary="Fixed in cycle 1")

    lead, _, _, _ = create_test_lead_agent([rework_res_1, valid_res])
    lead.recover_final_validation()

    state = lead.state
    assert state.final_validation_recovery_cycles == 1
    assert len(state.final_validation_history) == 2

    # Serialize and deserialize
    data = state.to_dict()
    assert data["final_validation_recovery_cycles"] == 1
    assert data["final_validation_recovery_exhausted"] is False
    assert len(data["final_validation_history"]) == 2

    restored = BRDAgentState.from_dict(data)
    assert restored.final_validation_recovery_cycles == 1
    assert restored.final_validation_recovery_exhausted is False
    assert len(restored.final_validation_history) == 2
    assert restored.latest_final_validation_result.is_valid is True


def test_meaningful_recovery_lifecycle_logging(caplog: pytest.LogCaptureFixture) -> None:
    """15. Meaningful recovery lifecycle logging occurs."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Contradiction found.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Contradiction")],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, _, _, _ = create_test_lead_agent([rework_res, valid_res])

    with caplog.at_level(logging.INFO):
        lead.recover_final_validation(context=AgentContext(project_id="test_log_proj"))

    logs = caplog.text
    assert "BRD final validation recovery started" in logs
    assert "Final validation recovery identified affected sections" in logs
    assert "Final validation section rework started" in logs
    assert "Final validation section rework completed" in logs
    assert "Final validation section validation completed" in logs
    assert "Final validation recovery cycle 1 completed" in logs
    assert "BRD final validation passed after recovery" in logs


def test_recovery_lifecycle_logging_exhausted(caplog: pytest.LogCaptureFixture) -> None:
    """15b. Lifecycle logging records recovery exhausted when max cycles reached."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Persistent issue.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Persistent")],
    )

    lead, _, _, _ = create_test_lead_agent([rework_res] * 5)

    with caplog.at_level(logging.INFO):
        lead.recover_final_validation(context=AgentContext(project_id="exhaust_proj"))

    logs = caplog.text
    assert "BRD final validation recovery exhausted" in logs


def test_errors_fail_safely_and_do_not_produce_false_valid(caplog: pytest.LogCaptureFixture) -> None:
    """16. Errors fail safely and do not produce a false VALID result."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Needs rework.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue")],
    )

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res])

    # Force section generator to raise an unexpected runtime error
    def failing_generate(context: Any) -> Any:
        raise RuntimeError("Unexpected LLM failure during section generation")

    sec_gen.generate = failing_generate  # type: ignore

    with caplog.at_level(logging.ERROR):
        rec_res = lead.recover_final_validation(context=AgentContext(project_id="err_proj"))

    assert rec_res.is_valid is False
    assert rec_res.needs_rework is True
    assert "Error during final validation recovery section rework" in caplog.text


@pytest.mark.asyncio
async def test_async_final_validation_recovery_parity() -> None:
    """17. Asynchronous final validation recovery parity."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Cross-section error.",
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue")],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, sec_val, final_val = create_test_lead_agent([rework_res, valid_res])

    rec_res = await lead.recover_final_validation_async(context=AgentContext(project_id="async_p1"))

    assert rec_res.is_valid is True
    assert rec_res.recovery_cycles == 1
    assert len(sec_gen.calls) == 1
    assert len(sec_val.calls) == 1
    assert len(final_val.calls) == 2


def test_unresolvable_affected_sections_stops_safely(caplog: pytest.LogCaptureFixture) -> None:
    """18. Unresolvable affected section edge case stops safely."""
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Issue in unknown section.",
        findings=[
            FinalValidationFinding(
                category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                issue="Unknown section defect",
                affected_sections=["NonExistentSection 999"],
            )
        ],
    )

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res])

    with caplog.at_level(logging.WARNING):
        rec_res = lead.recover_final_validation()

    assert rec_res.is_valid is False
    assert len(sec_gen.calls) == 0  # No sections updated
    assert "identified no resolvable affected sections" in caplog.text


def test_validate_final_brd_with_auto_recover() -> None:
    """Auto-recover parameter on validate_final_brd invokes recovery loop automatically."""
    sec1 = "3. In-Scope Business Modules & Feature Groups"
    rework_res = FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        findings=[FinalValidationFinding(category="Cross-Section Consistency", affected_sections=[sec1], issue="Issue")],
    )
    valid_res = FinalValidationResult(outcome=FinalValidationOutcome.VALID)

    lead, sec_gen, _, _ = create_test_lead_agent([rework_res, valid_res])

    # validate_final_brd with auto_recover=True
    res = lead.validate_final_brd(auto_recover=True)

    assert res.is_valid is True
    assert len(sec_gen.calls) == 1


def test_resolve_affected_sections_variations() -> None:
    """Test resolution of section names across various formats."""
    template_sections = [
        "1. Purpose & Scope of This Document",
        "2. Business Context (Summary from Discovery)",
        "3. In-Scope Business Modules & Feature Groups",
        "8. Integrations & External System Dependencies",
    ]

    candidates = [
        "Purpose & Scope of This Document",  # Number stripped
        "Section 3",  # Section N format
        "Integrations",  # Substring
        "2. Business Context (Summary from Discovery)",  # Exact match
        "Nonexistent Section",  # Ignored
    ]

    resolved = resolve_affected_sections(candidates, template_sections)
    assert resolved == [
        "1. Purpose & Scope of This Document",
        "2. Business Context (Summary from Discovery)",
        "3. In-Scope Business Modules & Feature Groups",
        "8. Integrations & External System Dependencies",
    ]
