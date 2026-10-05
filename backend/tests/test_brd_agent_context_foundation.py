"""Phase 4.3 — BRD Agent Context Foundation: Template Integration + Agent State tests.

Validates that:
1. The BRD template exists in the BRD Agent domain (src/agents/brd/brd_template.md).
2. The approved template is used rather than an invented structure.
3. The template is loaded reliably regardless of working directory.
4. The template is not hardcoded into Python; the template remains the single source of truth.
5. The System Instruction and BRD Template remain strictly separate artifacts.
6. The BRD Lead Agent can access the template, its sections, and its working state.
7. Agent State is available through the existing Agent runtime and DeepAgents harness.
8. State contains only information justified by the current BRD workflow:
   - Objective vs immediate task context
   - Template sections and section progress (Not Started, In Progress, Completed, Needs Revision)
   - Working information and evidence
   - Unresolved information and gaps
9. State supports section status transitions, evidence accumulation, gap resolution, and serialization.
10. Existing Agent and RAG tool functionality remain intact.
"""

import inspect
from pathlib import Path
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDDeepAgentState,
    BRDLeadAgent,
    BRDSectionStatus,
    SectionStatus,
    create_brd_lead_agent,
    extract_brd_sections,
    get_brd_template_path,
    get_system_instruction_path,
    load_brd_template,
    load_system_instruction,
)
from agents.brd.config import AgentConfig
from agents.brd.context import AgentContext, AgentRunRequest, AgentRunResponse
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for context foundation verification."""

    messages_to_return: list[AIMessage]
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default mock output"))]
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


# ---------------------------------------------------------------------------
# 1. Template Domain Artifact & Loading Tests
# ---------------------------------------------------------------------------


def test_brd_template_file_exists_in_domain():
    """Verify that brd_template.md is located within the BRD agent domain."""
    path = get_brd_template_path()
    assert path.is_file(), f"BRD template must exist at {path}"
    assert path.name == "brd_template.md"
    assert path.parent.name == "brd"
    assert path.parent.parent.name == "agents"


def test_brd_template_loads_authoritative_content():
    """Verify load_brd_template reliably loads non-empty markdown content."""
    content = load_brd_template()
    assert isinstance(content, str)
    assert len(content) > 100
    assert "Dexmiq Standard Document Header" in content


def test_missing_brd_template_file_raises_filenotfound(monkeypatch):
    """Verify load_brd_template raises FileNotFoundError when the file is missing."""
    import agents.brd.agent as brd_module

    fake_path = Path("/nonexistent/brd_template.md")
    monkeypatch.setattr(brd_module, "get_brd_template_path", lambda: fake_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_brd_template()


def test_empty_brd_template_file_raises_valueerror(tmp_path, monkeypatch):
    """Verify load_brd_template raises ValueError when the file is empty."""
    import agents.brd.agent as brd_module

    empty_file = tmp_path / "brd_template.md"
    empty_file.write_text("   \n\t   ", encoding="utf-8")
    monkeypatch.setattr(brd_module, "get_brd_template_path", lambda: empty_file)

    with pytest.raises(ValueError, match="is empty"):
        load_brd_template()


# ---------------------------------------------------------------------------
# 2. Approved Structure & Single Source of Truth Tests
# ---------------------------------------------------------------------------


def test_brd_template_contains_approved_sections():
    """Verify the approved BRD template contains the canonical required sections."""
    sections = extract_brd_sections()

    # Must contain canonical template sections
    section_text = " ".join(sections)
    assert "Dexmiq Standard Document Header" in section_text
    assert "Purpose & Scope of This Document" in section_text
    assert "Business Context" in section_text
    assert "In-Scope Business Modules & Feature Groups" in section_text

    # Preserves exact document order
    assert len(sections) == 16
    assert "Dexmiq Standard Document Header" in sections[0]
    assert "Version History" in sections[1]
    assert "Purpose & Scope" in sections[2]


def test_template_is_single_source_of_truth_dynamic_extraction():
    """Verify extract_brd_sections dynamically parses Markdown without hardcoded section lists."""
    custom_template = (
        "# Custom Spec\n\n"
        "## Alpha Section\nContent A\n\n"
        "## Beta Section\nContent B\n\n"
        "## Gamma Section\nContent C\n"
    )
    parsed = extract_brd_sections(custom_template)
    assert parsed == ["Alpha Section", "Beta Section", "Gamma Section"]


def test_no_hardcoded_brd_section_constants_in_python_code():
    """Verify agent.py and state.py do not hardcode the list of template sections."""
    import agents.brd.agent as brd_agent_module
    import agents.brd.state as brd_state_module

    agent_src = inspect.getsource(brd_agent_module)
    state_src = inspect.getsource(brd_state_module)

    # Neither module should hardcode a list of section names
    assert "BRD_SECTIONS" not in agent_src
    assert "BRD_SECTIONS" not in state_src
    assert "extract_brd_sections" in agent_src

    # The Markdown file itself remains the source of truth
    assert '["1. Introduction"' not in agent_src
    assert '["1. Introduction"' not in state_src


# ---------------------------------------------------------------------------
# 3. Separation of System Instruction and Template Tests
# ---------------------------------------------------------------------------


def test_system_instruction_and_template_are_distinct_files():
    """Verify system instruction and template are separate physical files with distinct purposes."""
    instr_path = get_system_instruction_path()
    tmpl_path = get_brd_template_path()

    assert instr_path != tmpl_path
    assert instr_path.name == "system_instruction.md"
    assert tmpl_path.name == "brd_template.md"

    instr_content = load_system_instruction()
    tmpl_content = load_brd_template()

    # System instruction defines behavioral identity and principles
    assert "## Identity" in instr_content
    assert "## Behavioral Principles" in instr_content

    # Template defines required document structure and headings
    assert "# Dexmiq Standard Document Header" in tmpl_content
    assert "# 1. Purpose & Scope of This Document" in tmpl_content


def test_brd_lead_agent_exposes_instruction_and_template_separately():
    """Verify BRDLeadAgent exposes system_instruction and template as distinct properties."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    assert agent.system_instruction == load_system_instruction()
    assert agent.template == load_brd_template()
    assert agent.brd_template == load_brd_template()
    assert agent.system_instruction != agent.template


# ---------------------------------------------------------------------------
# 4. Agent State Structure & Lifecycle Tests
# ---------------------------------------------------------------------------


def test_section_status_enum_values_and_normalization():
    """Verify SectionStatus contains required values and supports tolerant conversion."""
    assert BRDSectionStatus.NOT_STARTED.value == "Not Started"
    assert BRDSectionStatus.IN_PROGRESS.value == "In Progress"
    assert BRDSectionStatus.COMPLETED.value == "Completed"
    assert BRDSectionStatus.NEEDS_REVISION.value == "Needs Revision"

    # Alias check
    assert SectionStatus is BRDSectionStatus

    # Tolerant string parsing
    assert BRDSectionStatus.from_string("not_started") == BRDSectionStatus.NOT_STARTED
    assert BRDSectionStatus.from_string("IN_PROGRESS") == BRDSectionStatus.IN_PROGRESS
    assert BRDSectionStatus.from_string("completed") == BRDSectionStatus.COMPLETED
    assert BRDSectionStatus.from_string("needs revision") == BRDSectionStatus.NEEDS_REVISION

    with pytest.raises(ValueError, match="Invalid section status"):
        BRDSectionStatus.from_string("InvalidStatus")


def test_brd_agent_state_initialization_from_template():
    """Verify BRDAgentState initializes all template sections as Not Started."""
    sections = ["1. Introduction", "2. Business Requirements", "3. Functional Requirements"]
    state = BRDAgentState.initialize_from_template(
        sections=sections,
        objective="Generate requirements for billing engine",
        current_task="Initial scope analysis",
    )

    assert state.objective == "Generate requirements for billing engine"
    assert state.current_task == "Initial scope analysis"
    assert state.template_sections == sections
    assert state.current_section is None
    assert state.evidence == []
    assert state.unresolved_information == []
    assert state.is_complete is False

    for s in sections:
        assert state.get_section_status(s) == BRDSectionStatus.NOT_STARTED


def test_brd_agent_state_section_progress_transitions():
    """Verify section progress transitions between Not Started, In Progress, Completed, Needs Revision."""
    sections = ["1. Introduction", "2. Business Requirements"]
    state = BRDAgentState.initialize_from_template(sections=sections)

    # Set active section updates status to In Progress
    state.set_current_section("1. Introduction")
    assert state.current_section == "1. Introduction"
    assert state.get_section_status("1. Introduction") == BRDSectionStatus.IN_PROGRESS

    # Matching works by clean title without prefix
    assert state.get_section_status("Introduction") == BRDSectionStatus.IN_PROGRESS

    # Mark as completed
    state.update_section_status("1. Introduction", BRDSectionStatus.COMPLETED)
    assert state.get_section_status("1. Introduction") == BRDSectionStatus.COMPLETED
    assert "1. Introduction" in state.get_completed_sections()

    # Move to next section
    state.set_current_section("Business Requirements")
    assert state.get_section_status("2. Business Requirements") == BRDSectionStatus.IN_PROGRESS

    # Needs revision transition
    state.update_section_status("Business Requirements", BRDSectionStatus.NEEDS_REVISION)
    assert state.get_section_status("2. Business Requirements") == BRDSectionStatus.NEEDS_REVISION
    assert "2. Business Requirements" in state.get_needs_revision_sections()

    # Complete second section
    state.update_section_status("2. Business Requirements", BRDSectionStatus.COMPLETED)
    assert state.is_complete is True


def test_brd_agent_state_working_evidence_accumulation():
    """Verify evidence and working information can be added to working context."""
    state = BRDAgentState()
    assert state.evidence == []

    state.add_evidence({"source": "prd_v1.pdf", "fact": "SLA is 99.9%"})
    state.add_evidence("Customer requires SSO integration with Okta")

    assert len(state.evidence) == 2
    assert state.evidence[0]["fact"] == "SLA is 99.9%"
    assert state.evidence[1] == "Customer requires SSO integration with Okta"


def test_brd_agent_state_unresolved_information_tracking():
    """Verify unresolved information can be added and resolved."""
    state = BRDAgentState()
    state.add_unresolved("Unclear whether HIPAA compliance is required")
    state.add_unresolved("Expected peak concurrent user volume")

    assert len(state.unresolved_information) == 2

    # Duplicate addition is a no-op
    state.add_unresolved("Expected peak concurrent user volume")
    assert len(state.unresolved_information) == 2

    # Resolve an unresolved item
    resolved = state.resolve_unresolved("Expected peak concurrent user volume")
    assert resolved is True
    assert len(state.unresolved_information) == 1
    assert "HIPAA" in state.unresolved_information[0]

    # Resolving nonexistent item returns False
    assert state.resolve_unresolved("Nonexistent gap") is False


def test_brd_agent_state_serialization_roundtrip():
    """Verify BRDAgentState serializes to dict and deserializes back faithfully."""
    sections = ["1. Introduction", "2. Business Requirements"]
    state = BRDAgentState.initialize_from_template(
        sections=sections,
        objective="BRD for Auth service",
        current_task="Drafting Scope",
        metadata={"project_id": "proj-xyz-99"},
    )
    state.update_section_status("1. Introduction", BRDSectionStatus.COMPLETED)
    state.add_evidence("Okta SSO confirmed")
    state.add_unresolved("Audit retention period undefined")

    serialized = state.to_dict()
    assert serialized["objective"] == "BRD for Auth service"
    assert serialized["section_progress"]["1. Introduction"] == "Completed"
    assert serialized["section_progress"]["2. Business Requirements"] == "Not Started"
    assert serialized["metadata"]["project_id"] == "proj-xyz-99"

    restored = BRDAgentState.from_dict(serialized)
    assert restored.objective == state.objective
    assert restored.current_task == state.current_task
    assert restored.template_sections == state.template_sections
    assert restored.get_section_status("1. Introduction") == BRDSectionStatus.COMPLETED
    assert restored.evidence == state.evidence
    assert restored.unresolved_information == state.unresolved_information
    assert restored.metadata == state.metadata


# ---------------------------------------------------------------------------
# 5. Agent Integration: Instruction + Template + State
# ---------------------------------------------------------------------------


def test_brd_lead_agent_has_all_three_pillars():
    """Verify BRDLeadAgent encapsulates System Instruction, BRD Template, and Agent State."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # 1. System Instruction (Behavior)
    assert "BRD Lead Agent" in agent.system_instruction

    # 2. BRD Template (Required Document Structure)
    assert len(agent.sections) == 16
    assert "Dexmiq Standard Document Header" in agent.sections[0]
    assert "Completeness Snapshot" in agent.sections[-1]

    # 3. Agent State (Working Context)
    assert isinstance(agent.state, BRDAgentState)
    assert agent.state.template_sections == agent.sections
    for sec in agent.sections:
        assert agent.state.get_section_status(sec) == BRDSectionStatus.NOT_STARTED


def test_brd_lead_agent_custom_template_override():
    """Verify custom BRD template override is honored by the agent and state."""
    custom_template = (
        "# Custom Template\n\n"
        "## Scope\nContent\n\n"
        "## Architecture\nContent\n"
    )
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, template=custom_template)

    assert agent.template == custom_template
    assert agent.sections == ["Scope", "Architecture"]
    assert agent.state.template_sections == ["Scope", "Architecture"]
    assert agent.state.get_section_status("Scope") == BRDSectionStatus.NOT_STARTED


def test_brd_lead_agent_injected_state():
    """Verify an existing BRDAgentState can be injected into BRDLeadAgent."""
    custom_state = BRDAgentState(
        objective="Custom Injected Objective",
        current_task="Analyzing Gaps",
        unresolved_information=["Need DB schema"],
    )
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, state=custom_state)

    assert agent.state is custom_state
    assert agent.state.objective == "Custom Injected Objective"
    assert agent.state.unresolved_information == ["Need DB schema"]


def test_brd_lead_agent_reset_state():
    """Verify reset_state clears modifications and restores initial unstarted state."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    agent.state.update_section_status(agent.sections[0], BRDSectionStatus.COMPLETED)
    agent.state.add_evidence("Temp evidence")
    assert agent.state.get_section_status(agent.sections[0]) == BRDSectionStatus.COMPLETED

    agent.reset_state()
    assert agent.state.get_section_status(agent.sections[0]) == BRDSectionStatus.NOT_STARTED
    assert agent.state.evidence == []


def test_brd_lead_agent_execution_propagates_state_and_preserves_project_id():
    """Verify execution preserves project context in state metadata and returns state in response."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD analysis in progress.")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, tools=[echo_diagnostic_tool])

    context = AgentContext(project_id="proj_context_foundation_001")
    agent.state.set_current_section("1. Introduction")

    response = agent.execute("Analyze introduction", context=context)

    assert isinstance(response, AgentRunResponse)
    assert response.success is True
    assert response.output_text == "BRD analysis in progress."
    assert response.state is agent.state
    assert response.state.metadata.get("project_id") == "proj_context_foundation_001"
    assert response.state.current_section == "1. Introduction"


@pytest.mark.asyncio
async def test_brd_lead_agent_async_execution_propagates_state():
    """Verify async execution returns response with active working state."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Async BRD output.")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    context = AgentContext(project_id="proj_async_state_002")
    response = await agent.execute_async("Async test", context=context)

    assert isinstance(response, AgentRunResponse)
    assert response.state is agent.state
    assert response.state.metadata.get("project_id") == "proj_async_state_002"


def test_brd_lead_agent_execution_adopts_request_state():
    """Verify request.state in AgentRunRequest is adopted by the agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="State adopted.")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    incoming_state = BRDAgentState(
        objective="Incoming State Objective",
        current_task="Task from request",
    )
    request = AgentRunRequest(
        input_text="Continue workflow",
        context=AgentContext(project_id="proj_incoming_003"),
        state=incoming_state,
    )

    response = agent.execute(request)
    assert response.state.objective == "Incoming State Objective"
    assert response.state.current_task == "Task from request"
    assert response.state.metadata.get("project_id") == "proj_incoming_003"


def test_create_brd_lead_agent_factory_passes_template_and_state():
    """Verify create_brd_lead_agent forwards custom template and state."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Factory ready")])
    config = AgentConfig(model="test-model", api_key="test-key")

    custom_state = BRDAgentState(objective="Factory Objective")
    agent = create_brd_lead_agent(config=config, model=model, tools=[echo_diagnostic_tool], state=custom_state)

    assert isinstance(agent, BRDLeadAgent)
    assert agent.state.objective == "Factory Objective"


def test_deepagents_brd_state_schema_compatibility():
    """Verify BRDDeepAgentState integrates cleanly with the DeepAgents harness."""
    from deepagents import create_deep_agent
    from langchain_core.messages import HumanMessage

    model = MockChatModel(messages_to_return=[AIMessage(content="DeepAgents graph OK")])
    graph = create_deep_agent(model=model, state_schema=BRDDeepAgentState)

    result = graph.invoke({"messages": [HumanMessage(content="Test graph")]})
    assert "messages" in result
    assert len(result["messages"]) >= 2
