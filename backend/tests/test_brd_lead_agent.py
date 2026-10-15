"""Unit and execution tests for the BRD Lead Agent.

Validates that:
1. The BRD Lead Agent can be instantiated through the existing Agent runtime.
2. It executes through the existing DeepAgents harness.
3. Existing tools (including search_project_knowledge and echo_diagnostic_tool) remain available.
4. The existing RAG tool boundary is respected without direct database/vector access.
5. Project context (project_id) is strictly preserved through AgentContext.
6. No duplicate runtime/model/tool architecture is introduced.
"""

from unittest.mock import MagicMock
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDLeadAgent,
    create_brd_lead_agent,
    get_system_instruction_path,
    load_system_instruction,
)
from agents.brd.config import AgentConfig
from agents.brd.context import AgentContext, AgentRunRequest, AgentRunResponse
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for unit testing BRD Lead Agent execution."""

    messages_to_return: list[AIMessage]
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default fallback"))]
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


def test_brd_lead_agent_initialization_with_tools():
    """Verify BRDLeadAgent can be instantiated directly with tools."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD response")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = BRDLeadAgent(config=config, model=model, tools=[echo_diagnostic_tool])

    assert agent.agent_name == "BRDLeadAgent"
    assert agent.config.model == "test-model"
    assert agent.model is model
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "echo_diagnostic_tool"
    assert agent.graph is not None


def test_brd_lead_agent_instantiated_through_factory():
    """Verify BRDLeadAgent can be instantiated directly via create_brd_lead_agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD response")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = create_brd_lead_agent(config=config, model=model, tools=[echo_diagnostic_tool])

    assert isinstance(agent, BRDLeadAgent)
    assert agent.agent_name == "BRDLeadAgent"
    assert agent.config.model == "test-model"


def test_brd_lead_agent_standalone_initialization_equips_default_tools():
    """Verify BRDLeadAgent equips default tools (diagnostic + search_project_knowledge) when none supplied."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD ready")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = BRDLeadAgent(config=config, model=model)

    tool_names = [t.name for t in agent.tools]
    assert "echo_diagnostic_tool" in tool_names
    assert "search_project_knowledge" in tool_names
    assert "You are the BRD Lead Agent" in agent.default_system_prompt


def test_create_brd_lead_agent_factory():
    """Verify create_brd_lead_agent factory creates a configured BRDLeadAgent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD ready")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = create_brd_lead_agent(config=config, model=model, tools=[echo_diagnostic_tool])

    assert isinstance(agent, BRDLeadAgent)
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "echo_diagnostic_tool"


def test_brd_lead_agent_synchronous_execution_through_deepagents_harness():
    """Verify BRDLeadAgent executes through the existing DeepAgents harness and returns AgentRunResponse."""
    expected_content = "BRD analysis completed for requirements specification."
    model = MockChatModel(messages_to_return=[AIMessage(content=expected_content)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, tools=[echo_diagnostic_tool])

    context = AgentContext(project_id="proj_brd_001")
    response = agent.execute("Analyze requirements for notification service", context=context)

    assert isinstance(response, AgentRunResponse)
    assert response.success is True
    assert response.output_text == expected_content
    assert response.model == "test-model"
    assert response.context.project_id == "proj_brd_001"
    assert len(response.messages) >= 2


@pytest.mark.asyncio
async def test_brd_lead_agent_async_execution_through_deepagents_harness():
    """Verify BRDLeadAgent async execution works through the existing DeepAgents harness."""
    expected_content = "Async BRD evaluation finished."
    model = MockChatModel(messages_to_return=[AIMessage(content=expected_content)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, tools=[echo_diagnostic_tool])

    context = AgentContext(project_id="proj_brd_async_002")
    response = await agent.execute_async("Async prompt", context=context)

    assert isinstance(response, AgentRunResponse)
    assert response.success is True
    assert response.output_text == expected_content
    assert response.context.project_id == "proj_brd_async_002"


def test_brd_lead_agent_tool_invocation_via_boundary():
    """Verify BRDLeadAgent interacts with tools through the existing tool boundary."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "echo_diagnostic_tool",
            "args": {"message": "brd-evidence-check"},
            "id": "call_brd_tool_001",
            "type": "tool_call",
        }],
    )
    final_reply = "Diagnostic confirmed: [DIAGNOSTIC_OK] Echo: brd-evidence-check"
    final_msg = AIMessage(content=final_reply)

    model = MockChatModel(messages_to_return=[tool_call_msg, final_msg])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, tools=[echo_diagnostic_tool, search_project_knowledge])

    context = AgentContext(project_id="proj_brd_tool_003")
    response = agent.execute("Check diagnostic tool", context=context)

    assert response.success is True
    assert response.output_text == final_reply
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "echo_diagnostic_tool"
    assert response.tool_calls[0]["args"] == {"message": "brd-evidence-check"}
    assert response.context.project_id == "proj_brd_tool_003"


def test_brd_lead_agent_project_isolation():
    """Verify project isolation is strictly maintained and context cannot be modified by model."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Isolated output")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    context = AgentContext(
        project_id="strictly-enforced-project-uuid",
        conversation_id="conv-session-123",
        user_id="user-456",
    )
    request = AgentRunRequest(
        input_text="Generate BRD section",
        context=context,
    )

    response = agent.execute(request)

    assert response.context.project_id == "strictly-enforced-project-uuid"
    assert response.context.conversation_id == "conv-session-123"
    assert response.context.user_id == "user-456"


def test_brd_lead_agent_no_duplicate_architectures():
    """Verify that BRDLeadAgent is a direct agent using standard components."""
    model = MockChatModel(messages_to_return=[AIMessage(content="OK")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Model is standard BaseChatModel
    assert isinstance(agent.model, BaseChatModel)
    # Config is standard AgentConfig
    assert isinstance(agent.config, AgentConfig)
    # Graph is compiled DeepAgents graph
    assert agent.graph is not None


def test_brd_system_instruction_file_exists():
    """Verify the BRD system instruction Markdown file exists and is non-empty."""
    path = get_system_instruction_path()
    assert path.is_file(), f"Expected system instruction at {path}"
    content = path.read_text(encoding="utf-8").strip()
    assert len(content) > 0
    assert "# BRD Lead Agent" in content


def test_brd_system_instruction_loaded_and_structured():
    """Verify BRD Lead Agent system instruction loads with required sections."""
    instruction = load_system_instruction()
    assert "BRD Lead Agent" in instruction
    assert "## Identity" in instruction
    assert "## Objective" in instruction
    assert "## Responsibilities" in instruction
    assert "## Evidence Principles" in instruction
    assert "## Project Context" in instruction
    assert "## Boundaries" in instruction
    assert "## Behavioral Principles" in instruction

    # Evidence distinctions
    assert "Confirmed Information" in instruction
    assert "User-Provided Information" in instruction
    assert "Assumptions" in instruction
    assert "Unresolved Information" in instruction

    # Core principles & boundaries
    assert "Do not invent project facts" in instruction
    assert "Qdrant" in instruction
    assert "PostgreSQL" in instruction


def test_brd_lead_agent_loads_system_instruction_property():
    """Verify BRDLeadAgent exposes the loaded system instruction as a property."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    assert agent.system_instruction == load_system_instruction()
    assert agent.default_system_prompt == load_system_instruction()
    assert "BRD Lead Agent" in agent.system_instruction


def test_system_instruction_supplied_to_model():
    """Verify system instruction is properly loaded and set on agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    assert agent.system_instruction == load_system_instruction()
    assert "BRD Lead Agent" in agent.system_instruction


def test_brd_lead_agent_custom_instruction_override():
    """Verify custom system instruction override is honored."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Custom ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    custom_instruction = "Custom BRD Agent test instruction."

    agent = BRDLeadAgent(config=config, model=model, system_instruction=custom_instruction)

    assert agent.system_instruction == custom_instruction


def test_missing_system_instruction_file_raises_error(monkeypatch):
    """Verify clear error is raised if the system instruction file is missing."""
    from pathlib import Path
    import agents.brd.agent as brd_module

    fake_path = Path("/nonexistent/system_instruction.md")
    monkeypatch.setattr(brd_module, "get_system_instruction_path", lambda: fake_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_system_instruction()


def test_empty_system_instruction_file_raises_error(tmp_path, monkeypatch):
    """Verify clear error is raised if the system instruction file is empty."""
    import agents.brd.agent as brd_module

    empty_file = tmp_path / "system_instruction.md"
    empty_file.write_text("   \n  \t  ", encoding="utf-8")
    monkeypatch.setattr(brd_module, "get_system_instruction_path", lambda: empty_file)

    with pytest.raises(ValueError, match="is empty"):
        load_system_instruction()


def test_no_duplicate_hardcoded_brd_system_instruction_in_agent_code():
    """Verify agent.py does not duplicate the system instruction content."""
    import inspect
    import agents.brd.agent as brd_agent_module

    source_code = inspect.getsource(brd_agent_module)

    # The file should delegate to load_system_instruction() and not embed the markdown instruction
    assert "load_system_instruction" in source_code
    assert "system_instruction.md" in source_code
    # The full instruction headers and body text should not be duplicated as python strings
    assert "## Evidence Principles" not in source_code
    assert "## Behavioral Principles" not in source_code
    assert "Do not invent project facts, requirements" not in source_code

