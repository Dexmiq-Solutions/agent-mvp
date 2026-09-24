"""Unit tests for the developer manual Agent runner script."""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure scripts directory is on sys.path for test imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BACKEND_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_agent import (
    DEFAULT_DEVELOPMENT_PROJECT_ID,
    build_developer_runtime,
    execute_prompt,
    interactive_loop,
    main,
)
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunResponse
from exceptions.agent import AgentConfigurationError, AgentExecutionError


def test_build_developer_runtime_constructs_runtime():
    """Verify build_developer_runtime configures runtime with diagnostic tool."""
    config = AgentConfig(model="mock-model", api_key="test-key")
    with patch("agents.runtime.agent.create_agent_model"):
        with patch("agents.runtime.agent.create_runtime_agent"):
            runtime = build_developer_runtime(config=config)
            assert runtime.config.model == "mock-model"
            assert len(runtime.tools) == 1
            assert runtime.tools[0].name == "echo_diagnostic_tool"


def test_build_developer_runtime_raises_on_missing_config():
    """Verify build_developer_runtime raises AgentConfigurationError when API key is missing."""
    config = AgentConfig(model="mock-model", api_key=None)
    with pytest.raises(AgentConfigurationError, match="Missing API key"):
        build_developer_runtime(config=config)


def test_execute_prompt_passes_development_context():
    """Verify execute_prompt forwards prompt and development-test project_id to runtime."""
    mock_runtime = MagicMock()
    mock_response = AgentRunResponse(output_text="Test answer", success=True)
    mock_runtime.execute.return_value = mock_response

    response = execute_prompt(mock_runtime, "Hello developer", project_id="custom-dev-id")

    assert response == mock_response
    mock_runtime.execute.assert_called_once()
    called_prompt, kwargs = mock_runtime.execute.call_args
    assert called_prompt[0] == "Hello developer"
    context = kwargs["context"]
    assert isinstance(context, AgentContext)
    assert context.project_id == "custom-dev-id"


def test_interactive_loop_handles_prompts_and_exits():
    """Verify interactive_loop executes prompts, reports tool calls, and exits cleanly on 'exit'."""
    mock_runtime = MagicMock()
    mock_runtime.config.model = "test-model"
    mock_tool = MagicMock()
    mock_tool.name = "echo_diagnostic_tool"
    mock_runtime.tools = [mock_tool]

    mock_response = AgentRunResponse(
        output_text="Diagnostic complete.",
        tool_calls=[{"name": "echo_diagnostic_tool", "args": {"message": "ping"}}],
        success=True,
    )
    mock_runtime.execute.return_value = mock_response

    inputs = ["Run diagnostic test", "exit"]
    input_generator = iter(inputs)
    outputs = []

    interactive_loop(
        mock_runtime,
        project_id="development-test",
        input_func=lambda _: next(input_generator),
        print_func=lambda msg="": outputs.append(str(msg)),
    )

    combined_output = "\n".join(outputs)
    assert "Agent MVP - Development Runner" in combined_output
    assert "Model:      test-model" in combined_output
    assert "Project ID: development-test" in combined_output
    assert "[Tool Invocation: echo_diagnostic_tool" in combined_output
    assert "Diagnostic complete." in combined_output
    assert "Exiting development runner. Goodbye!" in combined_output


def test_interactive_loop_handles_keyboard_interrupt():
    """Verify interactive_loop catches KeyboardInterrupt and exits without crashing."""
    mock_runtime = MagicMock()
    mock_runtime.config.model = "test-model"
    mock_runtime.tools = []

    outputs = []

    def mock_input(_):
        raise KeyboardInterrupt()

    interactive_loop(
        mock_runtime,
        input_func=mock_input,
        print_func=lambda msg="": outputs.append(str(msg)),
    )

    combined_output = "\n".join(outputs)
    assert "Exiting development runner. Goodbye!" in combined_output


def test_interactive_loop_handles_agent_errors_gracefully():
    """Verify interactive_loop catches AgentError, displays message, and allows continuing."""
    mock_runtime = MagicMock()
    mock_runtime.config.model = "test-model"
    mock_runtime.tools = []
    mock_runtime.execute.side_effect = AgentExecutionError("Upstream timeout")

    inputs = ["Faulty prompt", "quit"]
    input_generator = iter(inputs)
    outputs = []

    interactive_loop(
        mock_runtime,
        input_func=lambda _: next(input_generator),
        print_func=lambda msg="": outputs.append(str(msg)),
    )

    combined_output = "\n".join(outputs)
    assert "[Agent Error]: Upstream timeout" in combined_output
    assert "Exiting development runner. Goodbye!" in combined_output


def test_main_missing_config_returns_exit_code_1():
    """Verify main returns 1 and helpful error message when configuration is invalid."""
    with patch("run_agent.build_developer_runtime", side_effect=AgentConfigurationError("Missing key")):
        exit_code = main()
        assert exit_code == 1
