"""Real integration smoke test verifying DeepAgents + OpenRouter end-to-end execution.

Runs only when valid credentials are present in the environment (e.g. from .env).
Verifies live agent execution and tool orchestration against the configured OpenRouter model.
Never logs or exposes secret credentials.
"""

import pytest

from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext
from core.config import get_settings
from tools.diagnostic import echo_diagnostic_tool


@pytest.mark.integration
def test_openrouter_live_agent_smoke():
    """Live smoke test executing a real Agent interaction against OpenRouter using DeepAgents."""
    settings = get_settings()
    config = AgentConfig.from_settings(settings)

    if not config.api_key:
        pytest.skip("OpenRouter API key is not configured; skipping live smoke test.")

    # Initialize live runtime with configured OpenRouter model and diagnostic tool
    runtime = AgentRuntime(config=config, tools=[echo_diagnostic_tool])

    context = AgentContext(project_id="integration_smoke_project")
    prompt = "Please run the echo_diagnostic_tool with message 'smoke_ok'."

    try:
        response = runtime.execute(prompt, context=context)
    except Exception as exc:
        pytest.fail(f"Live OpenRouter DeepAgents execution failed: {exc}")

    assert response.success is True
    assert response.output_text is not None
    assert len(response.output_text) > 0
    assert response.context.project_id == "integration_smoke_project"

    # Ensure secret credentials are never present in the output
    assert config.api_key not in response.output_text
