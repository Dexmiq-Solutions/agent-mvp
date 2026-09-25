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

from agents.brd import BRDLeadAgent, create_brd_lead_agent
from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest, AgentRunResponse
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


def test_brd_lead_agent_initialization_with_runtime():
    """Verify BRDLeadAgent can be instantiated wrapping an existing AgentRuntime."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD response")])
    config = AgentConfig(model="test-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model, tools=[echo_diagnostic_tool])

    agent = BRDLeadAgent(runtime=runtime)

    assert agent.runtime is runtime
    assert agent.agent_name == "BRDLeadAgent"
    assert agent.config.model == "test-model"
    assert agent.model is model
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "echo_diagnostic_tool"
    assert agent.graph is runtime.graph


def test_brd_lead_agent_instantiated_through_agent_runtime():
    """Verify BRDLeadAgent can be instantiated directly via AgentRuntime.create_brd_lead_agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="BRD response")])
    config = AgentConfig(model="test-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model, tools=[echo_diagnostic_tool])

    agent = runtime.create_brd_lead_agent()

    assert isinstance(agent, BRDLeadAgent)
    assert agent.runtime is runtime
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
    """Verify that BRDLeadAgent is a specialization using existing AgentRuntime, not a parallel runtime."""
    model = MockChatModel(messages_to_return=[AIMessage(content="OK")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Runtime is an existing AgentRuntime instance
    assert isinstance(agent.runtime, AgentRuntime)
    # Model is standard BaseChatModel
    assert isinstance(agent.model, BaseChatModel)
    # Config is standard AgentConfig
    assert isinstance(agent.config, AgentConfig)
    # Graph is compiled DeepAgents graph
    assert agent.graph is not None
