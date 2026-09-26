"""Tests for deterministic BRD section progression and completion detection.

Verifies:
1. Authoritative ordered section retrieval from template.
2. Current section identification and status tracking via existing section_progress.
3. Deterministic next-section selection in template order.
4. Next section activation as IN_PROGRESS.
5. Detection of section processing completion when all top-level sections complete.
6. Lightweight SectionProgressionResult and dynamic remaining sections calculation.
7. Error handling for missing section, invalid section, uncompleted section, empty template.
8. Safe idempotent handling of already-complete BRDs.
9. State integrity preservation (content, validation history, metadata).
10. Lead Agent integration (sync and async workflows, auto_progress, logging).
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
    BRDLeadAgent,
    BRDSectionStatus,
    SectionGenerationResult,
    SectionOperation,
    SectionProgressionResult,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
    create_brd_lead_agent,
    determine_next_section,
    extract_brd_sections,
    get_remaining_sections,
    initialize_progression,
    is_section_processing_complete,
    progress_to_next_section,
)
from agents.brd.config import AgentConfig
from agents.brd.context import AgentContext


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------


class MockProgressionChatModel(BaseChatModel):
    """Deterministic mock chat model for generation and validation testing."""

    tools_bound: list[Any] = []

    def __init__(self, mode: str = "valid", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "tools_bound", [])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = " ".join([m.content for m in messages if isinstance(m.content, str)])

        # Section Validation response
        if (
            "Target Section to Validate" in prompt_text
            or "validation dimensions" in prompt_text
            or "SECTION VALIDATION SPECIALIST" in prompt_text
            or "EVALUATE AND SCORE" in prompt_text
        ):
            if self._mode == "rework":
                content = (
                    "```json\n"
                    "{\n"
                    '  "outcome": "NEEDS_REWORK",\n'
                    '  "findings": [\n'
                    "    {\n"
                    '      "category": "Completeness",\n'
                    '      "severity": "HIGH",\n'
                    '      "issue": "Missing required details",\n'
                    '      "recommendation": "Add missing module specifications"\n'
                    "    }\n"
                    "  ],\n"
                    '  "rework_feedback": "Please add required module details."\n'
                    "}\n"
                    "```"
                )
            else:
                content = (
                    "```json\n"
                    "{\n"
                    '  "outcome": "VALID",\n'
                    '  "findings": [],\n'
                    '  "rework_feedback": null\n'
                    "}\n"
                    "```"
                )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        # Section Generation response
        content = (
            "```markdown\n"
            "# Generated Section Content\n\n"
            "This section provides validated, comprehensive business requirements.\n"
            "```"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        object.__setattr__(self, "tools_bound", list(tools))
        return self

    @property
    def _llm_type(self) -> str:
        return "mock_progression_chat_model"


THREE_SECTION_TEMPLATE = (
    "# 1. Purpose & Scope\n\n"
    "- Purpose of the engagement.\n\n"
    "# 2. Business Context\n\n"
    "- Summary from discovery.\n\n"
    "# 3. Requirements\n\n"
    "- High-level requirements.\n"
)


# ---------------------------------------------------------------------------
# 1. Authoritative Ordered Section Retrieval Tests
# ---------------------------------------------------------------------------


def test_extract_top_level_sections_from_authoritative_brd_template():
    """Verify extract_brd_sections dynamically parses top-level sections from brd_template.md."""
    sections = extract_brd_sections()
    assert isinstance(sections, list)
    assert len(sections) >= 5

    # Top-level sections must include core Dexmiq sections
    assert "1. Purpose & Scope of This Document" in sections
    assert "2. Business Context (Summary from Discovery)" in sections
    assert "5. Stakeholders & Personas" in sections
    assert "6. High-Level Business Requirements by Module" in sections
    assert "14. HL-BRD Quality Gate & Completeness Snapshot" in sections

    # Nested subsections must NOT be top-level progression units
    assert "5.1 Personas" not in sections
    assert "5.2 Persona–Module Mapping" not in sections
    assert "Quality Gate Checklist" not in sections


def test_extract_sections_custom_template_h1_and_h2():
    """Verify template parsing works for standard markdown where h1 is title and h2 are sections."""
    custom_tmpl = (
        "# Overall Document Title\n\n"
        "## 1. Introduction\nContent\n\n"
        "## 2. Architecture\nContent\n\n"
        "## 3. Deployment\nContent\n"
    )
    sections = extract_brd_sections(custom_tmpl)
    assert sections == ["1. Introduction", "2. Architecture", "3. Deployment"]


def test_template_is_single_source_of_truth_no_hardcoding():
    """Verify progression logic derives section order dynamically without hardcoding."""
    custom_tmpl = (
        "# Custom Alpha Spec\n\n"
        "# Custom Beta Spec\n\n"
        "# Custom Gamma Spec\n"
    )
    sections = extract_brd_sections(custom_tmpl)
    assert sections == ["Custom Alpha Spec", "Custom Beta Spec", "Custom Gamma Spec"]

    state = BRDAgentState.initialize_from_template(sections)
    state.set_current_section("Custom Alpha Spec")
    state.update_section_status("Custom Alpha Spec", BRDSectionStatus.COMPLETED)

    result = progress_to_next_section(state, sections)
    assert result.completed_section == "Custom Alpha Spec"
    assert result.next_section == "Custom Beta Spec"
    assert result.current_section == "Custom Beta Spec"


# ---------------------------------------------------------------------------
# 2. Section Progression Mechanics Tests
# ---------------------------------------------------------------------------


def test_normal_progression_first_to_second_section():
    """Verify Section 1 completed leads to Section 2 becoming IN_PROGRESS."""
    sections = ["Section 1", "Section 2", "Section 3"]
    state = BRDAgentState.initialize_from_template(sections)
    state.set_current_section("Section 1")
    state.update_section_status("Section 1", BRDSectionStatus.COMPLETED)

    result = progress_to_next_section(state, sections)

    # Completed section is marked COMPLETED
    assert state.get_section_status("Section 1") == BRDSectionStatus.COMPLETED
    # Next section becomes IN_PROGRESS
    assert state.get_section_status("Section 2") == BRDSectionStatus.IN_PROGRESS
    # Section 3 is still NOT_STARTED
    assert state.get_section_status("Section 3") == BRDSectionStatus.NOT_STARTED

    # Result properties
    assert result.completed_section == "Section 1"
    assert result.next_section == "Section 2"
    assert result.current_section == "Section 2"
    assert result.has_remaining_sections is True
    assert result.section_processing_complete is False
    assert result.remaining_sections == ["Section 2", "Section 3"]


def test_middle_section_progression():
    """Verify middle section progression from Section 2 to Section 3."""
    sections = ["Section 1", "Section 2", "Section 3", "Section 4"]
    state = BRDAgentState.initialize_from_template(sections)
    state.update_section_status("Section 1", BRDSectionStatus.COMPLETED)
    state.set_current_section("Section 2")
    state.update_section_status("Section 2", BRDSectionStatus.COMPLETED)

    result = progress_to_next_section(state, sections)

    assert result.completed_section == "Section 2"
    assert result.next_section == "Section 3"
    assert result.current_section == "Section 3"
    assert state.get_section_status("Section 3") == BRDSectionStatus.IN_PROGRESS
    assert result.remaining_sections == ["Section 3", "Section 4"]
    assert result.section_processing_complete is False


def test_final_section_progression_completes_brd():
    """Verify completing the final section triggers section_processing_complete."""
    sections = ["Section 1", "Section 2", "Section 3"]
    state = BRDAgentState.initialize_from_template(sections)
    state.update_section_status("Section 1", BRDSectionStatus.COMPLETED)
    state.update_section_status("Section 2", BRDSectionStatus.COMPLETED)
    state.set_current_section("Section 3")
    state.update_section_status("Section 3", BRDSectionStatus.COMPLETED)

    result = progress_to_next_section(state, sections)

    assert result.completed_section == "Section 3"
    assert result.next_section is None
    assert result.current_section is None
    assert result.has_remaining_sections is False
    assert result.section_processing_complete is True
    assert result.remaining_sections == []

    # State also confirms completion
    assert state.is_complete is True
    assert state.section_processing_complete is True


def test_incomplete_brd_completion_detection():
    """Verify is_section_processing_complete is False when some sections are not completed."""
    sections = ["1. Intro", "2. Context", "3. Requirements"]
    state = BRDAgentState.initialize_from_template(sections)
    state.update_section_status("1. Intro", BRDSectionStatus.COMPLETED)
    state.update_section_status("2. Context", BRDSectionStatus.IN_PROGRESS)
    state.update_section_status("3. Requirements", BRDSectionStatus.NOT_STARTED)

    assert is_section_processing_complete(state, sections) is False
    assert state.is_complete is False
    assert state.section_processing_complete is False


# ---------------------------------------------------------------------------
# 3. Dynamic Remaining Sections Derivation Tests
# ---------------------------------------------------------------------------


def test_remaining_sections_dynamically_derived_no_stale_list():
    """Verify remaining sections are dynamically derived from template order + section_progress."""
    sections = ["S1", "S2", "S3", "S4", "S5"]
    state = BRDAgentState.initialize_from_template(sections)

    # Initial: all remaining
    assert get_remaining_sections(state, sections) == ["S1", "S2", "S3", "S4", "S5"]

    # Mark S1 COMPLETED, S2 IN_PROGRESS
    state.update_section_status("S1", BRDSectionStatus.COMPLETED)
    state.update_section_status("S2", BRDSectionStatus.IN_PROGRESS)
    assert get_remaining_sections(state, sections) == ["S2", "S3", "S4", "S5"]

    # Mark S2 and S3 COMPLETED
    state.update_section_status("S2", BRDSectionStatus.COMPLETED)
    state.update_section_status("S3", BRDSectionStatus.COMPLETED)
    assert get_remaining_sections(state, sections) == ["S4", "S5"]

    # Mark all COMPLETED
    state.update_section_status("S4", BRDSectionStatus.COMPLETED)
    state.update_section_status("S5", BRDSectionStatus.COMPLETED)
    assert get_remaining_sections(state, sections) == []


# ---------------------------------------------------------------------------
# 4. Error Handling & Guardrails Tests
# ---------------------------------------------------------------------------


def test_missing_current_section_raises_error():
    """Verify attempting progression with no current_section set raises ValueError."""
    sections = ["Section 1", "Section 2"]
    state = BRDAgentState.initialize_from_template(sections)
    state.current_section = None

    with pytest.raises(ValueError, match="current_section is not set"):
        progress_to_next_section(state, sections)


def test_current_section_not_in_template_raises_error():
    """Verify current section outside template raises ValueError without guessing next."""
    sections = ["Section 1", "Section 2"]
    state = BRDAgentState.initialize_from_template(sections)
    state.set_current_section("Unrelated Phantom Section")
    state.update_section_status("Unrelated Phantom Section", BRDSectionStatus.COMPLETED)

    with pytest.raises(ValueError, match="does not exist in template"):
        progress_to_next_section(state, sections)


def test_uncompleted_current_section_cannot_advance():
    """Verify progression does not advance when current section is still IN_PROGRESS or NEEDS_REVISION."""
    sections = ["Section 1", "Section 2"]
    state = BRDAgentState.initialize_from_template(sections)
    state.set_current_section("Section 1")
    # Section 1 is IN_PROGRESS (not COMPLETED)
    assert state.get_section_status("Section 1") == BRDSectionStatus.IN_PROGRESS

    with pytest.raises(ValueError, match="is not completed"):
        progress_to_next_section(state, sections)

    # Verify no advancement occurred
    assert state.current_section == "Section 1"
    assert state.get_section_status("Section 2") == BRDSectionStatus.NOT_STARTED

    # Test with NEEDS_REVISION status
    state.update_section_status("Section 1", BRDSectionStatus.NEEDS_REVISION)
    with pytest.raises(ValueError, match="is not completed"):
        progress_to_next_section(state, sections)


def test_empty_template_sections_raises_error():
    """Verify progression with an empty template list raises ValueError."""
    state = BRDAgentState()
    state.set_current_section("Any Section")

    with pytest.raises(ValueError, match="contains no top-level sections"):
        progress_to_next_section(state, template_sections=[])


def test_already_complete_state_is_safe_and_idempotent():
    """Verify calling progression when all sections are complete is safe and does not reset state."""
    sections = ["Section 1", "Section 2"]
    state = BRDAgentState.initialize_from_template(sections)
    state.update_section_status("Section 1", BRDSectionStatus.COMPLETED)
    state.update_section_status("Section 2", BRDSectionStatus.COMPLETED)
    state.current_section = None

    result = progress_to_next_section(state, sections)

    assert result.section_processing_complete is True
    assert result.has_remaining_sections is False
    assert result.next_section is None
    # Progress is intact
    assert state.get_section_status("Section 1") == BRDSectionStatus.COMPLETED
    assert state.get_section_status("Section 2") == BRDSectionStatus.COMPLETED


# ---------------------------------------------------------------------------
# 5. State Integrity & Preservation Tests
# ---------------------------------------------------------------------------


def test_state_integrity_preserved_during_progression():
    """Verify section content, evidence, validation results, and metadata remain intact across progression."""
    sections = ["1. Purpose", "2. Scope"]
    state = BRDAgentState.initialize_from_template(sections, metadata={"project_id": "proj-999"})
    state.set_current_section("1. Purpose")
    state.set_section_content("1. Purpose", "# 1. Purpose\n\nFull documented business scope.")
    state.add_evidence({"source": "interview", "notes": "client reqs"})
    state.add_unresolved("Pending SLA sign-off")
    state.update_section_status("1. Purpose", BRDSectionStatus.COMPLETED)

    result = progress_to_next_section(state, sections)

    # 1. Purpose preserves its completed content
    assert state.get_section_content("1. Purpose") == "# 1. Purpose\n\nFull documented business scope."
    assert state.get_section_status("1. Purpose") == BRDSectionStatus.COMPLETED
    assert state.evidence == [{"source": "interview", "notes": "client reqs"}]
    assert state.unresolved_information == ["Pending SLA sign-off"]
    assert state.metadata["project_id"] == "proj-999"

    # 2. Scope is now current and IN_PROGRESS
    assert state.current_section == "2. Scope"
    assert state.get_section_status("2. Scope") == BRDSectionStatus.IN_PROGRESS


def test_progression_result_serialization_roundtrip():
    """Verify SectionProgressionResult and BRDAgentState serialization roundtrip cleanly."""
    prog_result = SectionProgressionResult(
        current_section="Section 2",
        completed_section="Section 1",
        next_section="Section 2",
        has_remaining_sections=True,
        section_processing_complete=False,
        remaining_sections=["Section 2", "Section 3"],
    )

    data = prog_result.to_dict()
    restored = SectionProgressionResult.from_dict(data)

    assert restored.current_section == "Section 2"
    assert restored.completed_section == "Section 1"
    assert restored.next_section == "Section 2"
    assert restored.has_remaining_sections is True
    assert restored.section_processing_complete is False
    assert restored.remaining_sections == ["Section 2", "Section 3"]

    # State roundtrip
    state = BRDAgentState.initialize_from_template(["S1", "S2"])
    state.set_progression_result(prog_result)
    state_dict = state.to_dict()
    assert "latest_progression_result" in state_dict
    assert state_dict["latest_progression_result"]["completed_section"] == "Section 1"

    restored_state = BRDAgentState.from_dict(state_dict)
    assert restored_state.latest_progression_result is not None
    assert restored_state.latest_progression_result.completed_section == "Section 1"
    assert len(restored_state.progression_history) == 1


# ---------------------------------------------------------------------------
# 6. Lead Agent Integration Tests
# ---------------------------------------------------------------------------


def test_lead_agent_progress_section_direct():
    """Verify Lead Agent progress_section method deterministically advances state."""
    model = MockProgressionChatModel()
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)

    # Initialize at first section
    agent.state.set_current_section("1. Purpose & Scope")
    agent.state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)

    prog_result = agent.progress_section()

    assert prog_result.completed_section == "1. Purpose & Scope"
    assert prog_result.next_section == "2. Business Context"
    assert agent.state.current_section == "2. Business Context"
    assert agent.state.get_section_status("2. Business Context") == BRDSectionStatus.IN_PROGRESS
    assert agent.state.latest_progression_result is prog_result


def test_lead_agent_generate_and_validate_with_auto_progress():
    """Verify generate_and_validate_section with auto_progress=True advances to next section on VALID."""
    model = MockProgressionChatModel(mode="valid")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)
    agent.state.set_current_section("1. Purpose & Scope")

    gen_res, val_res = agent.generate_and_validate_section(
        section="1. Purpose & Scope",
        available_information="Project goals",
        auto_progress=True,
    )

    assert val_res.is_valid is True
    # Section 1 is COMPLETED
    assert agent.state.get_section_status("1. Purpose & Scope") == BRDSectionStatus.COMPLETED
    # Section 2 automatically became current and IN_PROGRESS
    assert agent.state.current_section == "2. Business Context"
    assert agent.state.get_section_status("2. Business Context") == BRDSectionStatus.IN_PROGRESS
    assert agent.state.latest_progression_result is not None
    assert agent.state.latest_progression_result.completed_section == "1. Purpose & Scope"


def test_lead_agent_generate_and_validate_without_auto_progress_does_not_advance():
    """Verify generate_and_validate_section defaults to auto_progress=False and preserves current section."""
    model = MockProgressionChatModel(mode="valid")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)
    agent.state.set_current_section("1. Purpose & Scope")

    gen_res, val_res = agent.generate_and_validate_section(
        section="1. Purpose & Scope",
        available_information="Project goals",
        auto_progress=False,
    )

    assert val_res.is_valid is True
    assert agent.state.get_section_status("1. Purpose & Scope") == BRDSectionStatus.COMPLETED
    # Current section stays on 1. Purpose & Scope
    assert agent.state.current_section == "1. Purpose & Scope"


def test_lead_agent_process_current_section_valid_advances():
    """Verify process_current_section returns gen_result, val_result, and prog_result."""
    model = MockProgressionChatModel(mode="valid")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)
    agent.state.set_current_section("1. Purpose & Scope")

    gen_res, val_res, prog_res = agent.process_current_section(
        section="1. Purpose & Scope",
        available_information="Scope evidence",
    )

    assert val_res.is_valid is True
    assert prog_res is not None
    assert prog_res.completed_section == "1. Purpose & Scope"
    assert prog_res.next_section == "2. Business Context"
    assert agent.state.current_section == "2. Business Context"


def test_lead_agent_process_current_section_rework_does_not_advance():
    """Verify process_current_section does not progress when validation requires rework."""
    model = MockProgressionChatModel(mode="rework")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)
    agent.state.set_current_section("1. Purpose & Scope")

    gen_res, val_res, prog_res = agent.process_current_section(
        section="1. Purpose & Scope",
        available_information="Scope evidence",
        max_rework_attempts=1,
    )

    assert val_res.is_valid is False
    assert prog_res is None
    # Still on Section 1 with NEEDS_REVISION
    assert agent.state.current_section == "1. Purpose & Scope"
    assert agent.state.get_section_status("1. Purpose & Scope") == BRDSectionStatus.NEEDS_REVISION


def test_lead_agent_process_all_sections_sequential_flow():
    """Verify full sequential processing of all template sections to completion."""
    model = MockProgressionChatModel(mode="valid")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)

    results = agent.process_all_sections()

    # All 3 sections processed
    assert len(results) == 3
    sec1_gen, sec1_val, sec1_prog = results[0]
    sec2_gen, sec2_val, sec2_prog = results[1]
    sec3_gen, sec3_val, sec3_prog = results[2]

    # Section 1 progressed to 2
    assert sec1_prog.completed_section == "1. Purpose & Scope"
    assert sec1_prog.next_section == "2. Business Context"

    # Section 2 progressed to 3
    assert sec2_prog.completed_section == "2. Business Context"
    assert sec2_prog.next_section == "3. Requirements"

    # Section 3 completed all sections
    assert sec3_prog.completed_section == "3. Requirements"
    assert sec3_prog.next_section is None
    assert sec3_prog.section_processing_complete is True

    # Final agent state
    assert agent.is_section_processing_complete is True
    assert agent.state.is_complete is True
    assert agent.get_remaining_sections() == []
    for sec in agent.sections:
        assert agent.state.get_section_status(sec) == BRDSectionStatus.COMPLETED


@pytest.mark.asyncio
async def test_lead_agent_async_progression_workflow():
    """Verify asynchronous section processing and progression execution."""
    model = MockProgressionChatModel(mode="valid")
    agent = create_brd_lead_agent(model=model, template=THREE_SECTION_TEMPLATE)

    results = await agent.process_all_sections_async()

    assert len(results) == 3
    assert agent.is_section_processing_complete is True
    assert agent.state.section_processing_complete is True


# ---------------------------------------------------------------------------
# 7. Logging & Observability Tests
# ---------------------------------------------------------------------------


def test_progression_lifecycle_logging(caplog):
    """Verify lifecycle events are logged clearly using existing logging infrastructure."""
    caplog.set_level(logging.INFO)
    sections = ["Section A", "Section B"]
    state = BRDAgentState.initialize_from_template(sections)
    state.set_current_section("Section A")
    state.update_section_status("Section A", BRDSectionStatus.COMPLETED)

    progress_to_next_section(state, sections, project_id="test_proj_log")

    log_messages = [rec.message for rec in caplog.records]
    combined_logs = " ".join(log_messages)

    # 1. Section completed event
    assert "BRD section completed: Section A" in combined_logs
    # 2. Progression started event
    assert "BRD section progression started" in combined_logs
    # 3. Next section selected event
    assert "BRD progressing to next section: Section A completed; progressing to Section B" in combined_logs

    # Now complete Section B and verify processing completed log
    caplog.clear()
    state.update_section_status("Section B", BRDSectionStatus.COMPLETED)
    progress_to_next_section(state, sections, project_id="test_proj_log")

    log_messages2 = [rec.message for rec in caplog.records]
    combined_logs2 = " ".join(log_messages2)

    # 4. Section processing completed event
    assert "BRD section processing completed" in combined_logs2
