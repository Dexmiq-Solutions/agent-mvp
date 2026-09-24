"""Unit and execution tests for the Agent runtime harness, context, and deterministic tool flow."""

from unittest.mock import patch
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.runtime.agent import AgentRuntime, create_runtime_agent
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest
from exceptions.agent import (
    AgentExecutionError,
    AgentInitializationError,
)
from tools.diagnostic import echo_diagnostic_tool


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for unit testing Agent runtime."""

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


def test_agent_runtime_initialization():
    """Verify AgentRuntime initializes successfully with injected model and configuration."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Hello")])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model, tools=[echo_diagnostic_tool])

    assert runtime.config.model == "test-mock-model"
    assert len(runtime.tools) == 1
    assert runtime.tools[0].name == "echo_diagnostic_tool"
    assert runtime.graph is not None


def test_agent_runtime_initialization_error_wrapped():
    """Verify that errors during agent graph initialization are wrapped in AgentInitializationError."""
    with patch("agents.runtime.agent.create_deep_agent", side_effect=ValueError("Graph failed")):
        with pytest.raises(AgentInitializationError, match="Failed to initialize DeepAgents runtime"):
            create_runtime_agent(model=MockChatModel(messages_to_return=[]))


def test_agent_runtime_execution_basic():
    """Verify complete Agent execution path for a basic text interaction."""
    expected_reply = "Greetings! I am the foundational test agent."
    model = MockChatModel(messages_to_return=[AIMessage(content=expected_reply)])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model)

    response = runtime.execute("Hello agent")

    assert response.success is True
    assert response.output_text == expected_reply
    assert response.model == "test-mock-model"
    assert len(response.tool_calls) == 0
    assert len(response.messages) >= 2


def test_agent_runtime_execution_with_deterministic_tool():
    """Verify complete Agent -> Tool Decision -> Tool Execution -> Tool Result -> Agent Response path."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "echo_diagnostic_tool",
            "args": {"message": "smoke-test-payload"},
            "id": "call_diagnostic_001",
            "type": "tool_call",
        }],
    )
    final_agent_reply = "Diagnostic confirmed: [DIAGNOSTIC_OK] Echo: smoke-test-payload"
    final_msg = AIMessage(content=final_agent_reply)

    model = MockChatModel(messages_to_return=[tool_call_msg, final_msg])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model, tools=[echo_diagnostic_tool])

    response = runtime.execute("Run the diagnostic tool")

    assert response.success is True
    assert response.output_text == final_agent_reply
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "echo_diagnostic_tool"
    assert response.tool_calls[0]["args"] == {"message": "smoke-test-payload"}

    # Verify message sequence contains HumanMessage -> AIMessage (tool call) -> ToolMessage -> AIMessage (final)
    message_types = [type(m).__name__ for m in response.messages]
    assert "HumanMessage" in message_types
    assert "ToolMessage" in message_types
    assert "AIMessage" in message_types


@pytest.mark.asyncio
async def test_agent_runtime_async_execution():
    """Verify asynchronous execute_async works seamlessly."""
    expected_reply = "Async execution completed successfully."
    model = MockChatModel(messages_to_return=[AIMessage(content=expected_reply)])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model)

    response = await runtime.execute_async("Run async request")

    assert response.success is True
    assert response.output_text == expected_reply
    assert response.model == "test-mock-model"


def test_agent_runtime_project_isolation_boundary():
    """Verify project isolation: project_id is preserved through the AgentContext and cannot be altered by model."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Tenant isolated response")])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model)

    context = AgentContext(
        project_id="proj_alpha_999",
        conversation_id="conv_beta_888",
        user_id="user_gamma_777",
    )
    request = AgentRunRequest(
        input_text="Process request within project boundary",
        context=context,
    )

    response = runtime.execute(request)

    assert response.context.project_id == "proj_alpha_999"
    assert response.context.conversation_id == "conv_beta_888"
    assert response.context.user_id == "user_gamma_777"


def test_agent_runtime_execution_failure_wrapped():
    """Verify unhandled graph execution exceptions are wrapped in AgentExecutionError."""
    model = MockChatModel(messages_to_return=[])
    config = AgentConfig(model="test-mock-model", api_key="test-key")
    runtime = AgentRuntime(config=config, model=model)

    with patch.object(runtime._graph, "invoke", side_effect=RuntimeError("Graph crash")):
        with pytest.raises(AgentExecutionError, match="Agent execution failed: Graph crash"):
            runtime.execute("Trigger error")


def test_agent_runtime_observability_logging(caplog):
    """Verify structured logging records execution started and completed without leaking secrets."""
    import logging
    model = MockChatModel(messages_to_return=[AIMessage(content="Observed response")])
    config = AgentConfig(model="test-mock-model", api_key="secret-api-key-test")
    runtime = AgentRuntime(config=config, model=model)

    with caplog.at_level(logging.INFO):
        runtime.execute("Observe this execution", context=AgentContext(project_id="proj_obs_123"))

    assert "Agent execution started" in caplog.text
    assert "Agent execution completed" in caplog.text
    assert "proj_obs_123" in caplog.text
    assert "secret-api-key-test" not in caplog.text
