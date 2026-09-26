"""Tests for deterministic BRD document assembly.

Verifies:
1. Successful assembly with completed sections in template order.
2. Template order is authoritative regardless of insertion order.
3. Incomplete section processing (IN_PROGRESS, NOT_STARTED, NEEDS_REVISION) prevents assembly.
4. Missing or empty completed section content raises explicit error.
5. Empty template raises explicit error.
6. Idempotency: repeated assembly calls produce identical output without duplication.
7. Content preservation: markdown tables, bullets, IDs, and exact text are preserved.
8. Section boundary and heading formatting: prevents duplicate headings and ensures top-level headings.
9. State persistence: serialization and deserialization preserve assembled_brd and latest_assembly_result.
10. Lead Agent integration: assemble_brd (sync and async) and auto_assemble in process_all_sections.
11. Zero LLM involvement during assembly.
12. Lifecycle logging for assembly started, section included, assembly completed, and failure events.
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
    BRDAssemblyResult,
    BRDLeadAgent,
    BRDSectionStatus,
    SectionGenerationResult,
    SectionOperation,
    SectionProgressionResult,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
    assemble_brd_document,
    create_brd_lead_agent,
    format_section_for_assembly,
    get_heading_prefix_for_section,
)
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------


class MockAssemblyChatModel(BaseChatModel):
    """Mock chat model tracking invocation counts to verify assembly does not call LLMs."""

    invoke_count: int = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "invoke_count", 0)
        object.__setattr__(self, "tools_bound", [])

    @property
    def _llm_type(self) -> str:
        return "mock_assembly_model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockAssemblyChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invoke_count += 1
        prompt_text = " ".join([m.content for m in messages if isinstance(m.content, str)])

        # Validation prompt response
        if (
            "target section to validate" in prompt_text.lower()
            or "validation dimensions" in prompt_text.lower()
            or "section validation specialist" in prompt_text.lower()
            or "evaluate and score" in prompt_text.lower()
        ):
            val_json = (
                "```json\n"
                "{\n"
                '  "outcome": "VALID",\n'
                '  "findings": [],\n'
                '  "rework_feedback": null\n'
                "}\n"
                "```"
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=val_json))])

        # Generation prompt response
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


@pytest.fixture
def sample_template() -> str:
    return (
        "# 1. Purpose & Scope\n\n"
        "- Scope statement\n\n"
        "# 2. Business Objectives\n\n"
        "- Objectives\n\n"
        "# 3. Requirements\n\n"
        "## 3.1 Functional Requirements\n"
        "- FR-1\n"
    )


@pytest.fixture
def sample_sections() -> list[str]:
    return [
        "1. Purpose & Scope",
        "2. Business Objectives",
        "3. Requirements",
    ]


@pytest.fixture
def completed_state(sample_sections: list[str]) -> BRDAgentState:
    state = BRDAgentState.initialize_from_template(sample_sections)
    for sec in sample_sections:
        state.update_section_status(sec, BRDSectionStatus.COMPLETED)
        state.set_section_content(sec, f"# {sec}\n\nValidated content for {sec}.")
    return state


# ---------------------------------------------------------------------------
# Unit Tests: Heading & Formatting Preservation
# ---------------------------------------------------------------------------


def test_format_section_preserves_existing_exact_heading() -> None:
    content = "# 1. Purpose & Scope\n\nThis is the scope statement."
    formatted = format_section_for_assembly("1. Purpose & Scope", content)
    assert formatted == content
    assert formatted.count("# 1. Purpose & Scope") == 1


def test_format_section_preserves_number_stripped_heading() -> None:
    content = "# Purpose & Scope\n\nThis is the scope statement."
    formatted = format_section_for_assembly("1. Purpose & Scope", content)
    assert formatted == content
    assert formatted.count("# Purpose & Scope") == 1


def test_format_section_prepends_missing_heading() -> None:
    raw_content = "This is the body text without any heading."
    formatted = format_section_for_assembly("1. Purpose & Scope", raw_content)
    assert formatted.startswith("# 1. Purpose & Scope\n\n")
    assert "This is the body text without any heading." in formatted


def test_format_section_prepends_heading_when_starting_with_subsection() -> None:
    content = "## 5.1 Personas\n\n- Claims Officer: Processes daily claims."
    formatted = format_section_for_assembly("5. Stakeholders & Personas", content)
    assert formatted.startswith("# 5. Stakeholders & Personas\n\n")
    assert "## 5.1 Personas" in formatted


def test_format_section_handles_custom_template_heading_level() -> None:
    custom_tmpl = "## 1. Introduction\n\nText\n\n## 2. Scope\n\nText"
    prefix = get_heading_prefix_for_section("1. Introduction", template_content=custom_tmpl)
    assert prefix == "##"

    raw_content = "Body of introduction."
    formatted = format_section_for_assembly("1. Introduction", raw_content, template_content=custom_tmpl)
    assert formatted.startswith("## 1. Introduction\n\n")


# ---------------------------------------------------------------------------
# Unit Tests: Successful Assembly & Order Authority
# ---------------------------------------------------------------------------


def test_successful_assembly(completed_state: BRDAgentState, sample_sections: list[str]) -> None:
    result = assemble_brd_document(completed_state, template_sections=sample_sections)

    assert isinstance(result, BRDAssemblyResult)
    assert result.assembly_complete is True
    assert result.section_count == 3
    assert result.sections_assembled == sample_sections
    assert "# 1. Purpose & Scope" in result.assembled_document
    assert "# 2. Business Objectives" in result.assembled_document
    assert "# 3. Requirements" in result.assembled_document
    assert completed_state.assembled_brd == result.assembled_document
    assert completed_state.is_assembled is True
    assert completed_state.get_latest_assembly_result() == result


def test_template_order_is_authoritative(sample_sections: list[str]) -> None:
    """Verify sections are assembled strictly in template order, not insertion order."""
    state = BRDAgentState.initialize_from_template(sample_sections)
    # Insert contents in reverse order: Section 3, then 1, then 2
    state.update_section_status("3. Requirements", BRDSectionStatus.COMPLETED)
    state.set_section_content("3. Requirements", "# 3. Requirements\n\nContent 3")

    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "# 1. Purpose & Scope\n\nContent 1")

    state.update_section_status("2. Business Objectives", BRDSectionStatus.COMPLETED)
    state.set_section_content("2. Business Objectives", "# 2. Business Objectives\n\nContent 2")

    result = assemble_brd_document(state, template_sections=sample_sections)

    # Document must have Section 1 before Section 2 before Section 3
    pos1 = result.assembled_document.find("# 1. Purpose & Scope")
    pos2 = result.assembled_document.find("# 2. Business Objectives")
    pos3 = result.assembled_document.find("# 3. Requirements")

    assert pos1 != -1 and pos2 != -1 and pos3 != -1
    assert pos1 < pos2 < pos3


def test_assembly_content_preservation(sample_sections: list[str]) -> None:
    """Verify markdown tables, bullet lists, IDs, and exact text are preserved without alteration."""
    state = BRDAgentState.initialize_from_template(sample_sections)
    for sec in sample_sections:
        state.update_section_status(sec, BRDSectionStatus.COMPLETED)

    rich_table_content = (
        "# 3. Requirements\n\n"
        "| HL Requirement ID | Description (WHAT, not HOW) | Related Persona(s) |\n"
        "|---|---|---|\n"
        "| HL-BRD-3.1 | Automated claims triage | Claims Officer |\n"
        "| HL-BRD-3.2 | Real-time settlement audit | Compliance Lead |\n\n"
        "- Bullet item A\n"
        "- Bullet item B"
    )
    state.set_section_content("1. Purpose & Scope", "# 1. Purpose & Scope\n\nBasic scope.")
    state.set_section_content("2. Business Objectives", "# 2. Business Objectives\n\nReduce triage by 40%.")
    state.set_section_content("3. Requirements", rich_table_content)

    result = assemble_brd_document(state, template_sections=sample_sections)

    assert "| HL-BRD-3.1 | Automated claims triage | Claims Officer |" in result.assembled_document
    assert "| HL-BRD-3.2 | Real-time settlement audit | Compliance Lead |" in result.assembled_document
    assert "- Bullet item A" in result.assembled_document
    assert "Reduce triage by 40%." in result.assembled_document


def test_assembly_is_idempotent(completed_state: BRDAgentState, sample_sections: list[str]) -> None:
    """Verify running assembly multiple times produces identical output without duplicating sections."""
    result1 = assemble_brd_document(completed_state, template_sections=sample_sections)
    doc1 = result1.assembled_document

    result2 = assemble_brd_document(completed_state, template_sections=sample_sections)
    doc2 = result2.assembled_document

    assert doc1 == doc2
    # Ensure Section 1 heading occurs exactly once
    assert doc2.count("# 1. Purpose & Scope") == 1
    assert completed_state.assembled_brd == doc1


# ---------------------------------------------------------------------------
# Unit Tests: Error Handling & Readiness Checks
# ---------------------------------------------------------------------------


def test_assembly_fails_when_section_processing_incomplete(sample_sections: list[str]) -> None:
    """Assembly must refuse to proceed when any section is not COMPLETED."""
    state = BRDAgentState.initialize_from_template(sample_sections)
    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "# Content 1")

    state.update_section_status("2. Business Objectives", BRDSectionStatus.IN_PROGRESS)
    state.set_section_content("2. Business Objectives", "# Content 2")

    state.update_section_status("3. Requirements", BRDSectionStatus.NOT_STARTED)

    with pytest.raises(ValueError, match="section processing is incomplete"):
        assemble_brd_document(state, template_sections=sample_sections)


def test_assembly_fails_when_section_needs_revision(sample_sections: list[str]) -> None:
    state = BRDAgentState.initialize_from_template(sample_sections)
    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "# Content 1")
    state.update_section_status("2. Business Objectives", BRDSectionStatus.COMPLETED)
    state.set_section_content("2. Business Objectives", "# Content 2")
    state.update_section_status("3. Requirements", BRDSectionStatus.NEEDS_REVISION)
    state.set_section_content("3. Requirements", "# Content 3")

    with pytest.raises(ValueError, match="section processing is incomplete"):
        assemble_brd_document(state, template_sections=sample_sections)


def test_assembly_fails_on_missing_section_content(sample_sections: list[str]) -> None:
    """When a section is marked COMPLETED but has missing or empty content, fail explicitly."""
    state = BRDAgentState.initialize_from_template(sample_sections)
    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "# Content 1")

    state.update_section_status("2. Business Objectives", BRDSectionStatus.COMPLETED)
    # Section 2 marked completed but content is missing!

    state.update_section_status("3. Requirements", BRDSectionStatus.COMPLETED)
    state.set_section_content("3. Requirements", "# Content 3")

    with pytest.raises(ValueError, match="missing or empty content"):
        assemble_brd_document(state, template_sections=sample_sections)


def test_assembly_fails_on_empty_string_section_content(sample_sections: list[str]) -> None:
    state = BRDAgentState.initialize_from_template(sample_sections)
    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "   \n  ")  # whitespace only

    state.update_section_status("2. Business Objectives", BRDSectionStatus.COMPLETED)
    state.set_section_content("2. Business Objectives", "# Content 2")
    state.update_section_status("3. Requirements", BRDSectionStatus.COMPLETED)
    state.set_section_content("3. Requirements", "# Content 3")

    with pytest.raises(ValueError, match="missing or empty content"):
        assemble_brd_document(state, template_sections=sample_sections)


def test_assembly_fails_on_empty_template() -> None:
    state = BRDAgentState()
    with pytest.raises(ValueError, match="template contains no top-level sections"):
        assemble_brd_document(state, template_sections=[])


# ---------------------------------------------------------------------------
# Unit Tests: State Persistence & Serialization
# ---------------------------------------------------------------------------


def test_assembly_state_serialization(completed_state: BRDAgentState, sample_sections: list[str]) -> None:
    result = assemble_brd_document(completed_state, template_sections=sample_sections)

    serialized = completed_state.to_dict()
    assert "assembled_brd" in serialized
    assert serialized["assembled_brd"] == result.assembled_document
    assert "latest_assembly_result" in serialized
    assert serialized["latest_assembly_result"]["section_count"] == 3
    assert serialized["latest_assembly_result"]["assembly_complete"] is True

    deserialized = BRDAgentState.from_dict(serialized)
    assert deserialized.assembled_brd == result.assembled_document
    assert deserialized.is_assembled is True
    assert deserialized.latest_assembly_result is not None
    assert deserialized.latest_assembly_result.section_count == 3
    assert deserialized.latest_assembly_result.sections_assembled == sample_sections


# ---------------------------------------------------------------------------
# Unit Tests: Lead Agent Integration
# ---------------------------------------------------------------------------


def test_lead_agent_assemble_brd_sync(sample_sections: list[str], sample_template: str) -> None:
    mock_model = MockAssemblyChatModel()
    agent = create_brd_lead_agent(
        model=mock_model,
        template=sample_template,
    )
    for sec in sample_sections:
        agent.state.update_section_status(sec, BRDSectionStatus.COMPLETED)
        agent.state.set_section_content(sec, f"# {sec}\n\nApproved content.")

    assert agent.is_section_processing_complete is True
    assert agent.is_assembled is False

    res = agent.assemble_brd(context=AgentContext(project_id="test_project_123"))

    assert res.assembly_complete is True
    assert agent.is_assembled is True
    assert agent.assembled_brd == res.assembled_document
    assert res.metadata.get("project_id") == "test_project_123"
    # Verification: Assembly must NEVER invoke an LLM
    assert mock_model.invoke_count == 0


@pytest.mark.asyncio
async def test_lead_agent_assemble_brd_async(sample_sections: list[str], sample_template: str) -> None:
    mock_model = MockAssemblyChatModel()
    agent = create_brd_lead_agent(
        model=mock_model,
        template=sample_template,
    )
    for sec in sample_sections:
        agent.state.update_section_status(sec, BRDSectionStatus.COMPLETED)
        agent.state.set_section_content(sec, f"# {sec}\n\nApproved content.")

    res = await agent.assemble_brd_async(context=AgentContext(project_id="test_async_proj"))

    assert res.assembly_complete is True
    assert agent.is_assembled is True
    assert mock_model.invoke_count == 0


def test_lead_agent_auto_assemble_in_process_all_sections(
    sample_sections: list[str],
    sample_template: str,
) -> None:
    """Verify process_all_sections automatically triggers assembly when auto_assemble=True."""
    mock_model = MockAssemblyChatModel()
    agent = create_brd_lead_agent(
        model=mock_model,
        template=sample_template,
    )

    results = agent.process_all_sections(auto_assemble=True)

    assert len(results) == 3
    assert agent.is_section_processing_complete is True
    assert agent.is_assembled is True
    assert agent.assembled_brd is not None
    assert "# 1. Purpose & Scope" in agent.assembled_brd
    assert "# 2. Business Objectives" in agent.assembled_brd
    assert "# 3. Requirements" in agent.assembled_brd


@pytest.mark.asyncio
async def test_lead_agent_auto_assemble_in_process_all_sections_async(
    sample_sections: list[str],
    sample_template: str,
) -> None:
    """Verify process_all_sections_async automatically triggers assembly when auto_assemble=True."""
    mock_model = MockAssemblyChatModel()
    agent = create_brd_lead_agent(
        model=mock_model,
        template=sample_template,
    )

    results = await agent.process_all_sections_async(auto_assemble=True)

    assert len(results) == 3
    assert agent.is_section_processing_complete is True
    assert agent.is_assembled is True
    assert agent.assembled_brd is not None


# ---------------------------------------------------------------------------
# Unit Tests: Lifecycle Logging
# ---------------------------------------------------------------------------


def test_assembly_lifecycle_logging(
    completed_state: BRDAgentState,
    sample_sections: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify lifecycle logging events: started, section included, and completed."""
    with caplog.at_level(logging.INFO):
        assemble_brd_document(
            completed_state,
            template_sections=sample_sections,
            project_id="logging_project",
        )

    log_text = caplog.text
    assert "BRD assembly started" in log_text
    assert "BRD section included in assembly" in log_text
    assert "BRD assembly completed" in log_text
    assert "logging_project" in log_text


def test_assembly_failure_logging(
    sample_sections: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify lifecycle error logging on incomplete processing."""
    state = BRDAgentState.initialize_from_template(sample_sections)
    state.update_section_status("1. Purpose & Scope", BRDSectionStatus.COMPLETED)
    state.set_section_content("1. Purpose & Scope", "# Content")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            assemble_brd_document(state, template_sections=sample_sections, project_id="fail_project")

    assert "BRD assembly error" in caplog.text
    assert "section processing is incomplete" in caplog.text
