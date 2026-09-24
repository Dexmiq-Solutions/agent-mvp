"""Developer manual Agent runner for interactive terminal testing and debugging.

Provides a lightweight, interactive developer-facing entry point to exercise the
Phase 2 DeepAgents + OpenRouter runtime foundation.

Usage:
    From the backend directory:
    uv run python scripts/run_agent.py
"""

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Ensure src/ is on the Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunResponse
from exceptions.agent import AgentConfigurationError, AgentError
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import search_project_knowledge

DEFAULT_DEVELOPMENT_PROJECT_ID = "development-test"


def build_developer_runtime(
    config: Optional[AgentConfig] = None,
    tools: Optional[Sequence[Any]] = None,
    enable_rag: bool = False,
) -> AgentRuntime:
    """Build and initialize the AgentRuntime equipped with developer diagnostic tools and RAG.

    Args:
        config: Optional AgentConfig instance. Defaults to loading from application settings.
        tools: Optional explicit sequence of tools to equip.
        enable_rag: If True, equips search_project_knowledge alongside echo_diagnostic_tool.
            Defaults to False for backward compatibility with Phase 2 unit tests.

    Returns:
        Configured AgentRuntime instance.

    Raises:
        AgentConfigurationError: If required configuration or credentials are missing.
    """
    agent_config = config or AgentConfig.from_settings()
    agent_config.validate()
    if tools is not None:
        runtime_tools = list(tools)
    elif enable_rag:
        runtime_tools = [echo_diagnostic_tool, search_project_knowledge]
    else:
        runtime_tools = [echo_diagnostic_tool]

    return AgentRuntime(
        config=agent_config,
        tools=runtime_tools,
    )


def execute_prompt(
    runtime: AgentRuntime,
    prompt: str,
    project_id: str = DEFAULT_DEVELOPMENT_PROJECT_ID,
) -> AgentRunResponse:
    """Execute a single prompt against the Agent runtime within the development project context.

    Args:
        runtime: The active AgentRuntime instance.
        prompt: The user prompt string.
        project_id: Explicit project identifier for project isolation.

    Returns:
        AgentRunResponse: The normalized response from the Agent.
    """
    context = AgentContext(project_id=project_id)
    return runtime.execute(prompt, context=context)


def interactive_loop(
    runtime: AgentRuntime,
    project_id: str = DEFAULT_DEVELOPMENT_PROJECT_ID,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> None:
    """Run an interactive prompt-response loop with the developer.

    Args:
        runtime: The active AgentRuntime instance.
        project_id: Explicit project identifier for project isolation.
        input_func: Callable for reading user input (injected for testing).
        print_func: Callable for printing output (injected for testing).
    """
    print_func("========================================")
    print_func(" Agent MVP - Development Runner")
    print_func("========================================")
    print_func(f"Model:      {runtime.config.model}")
    print_func(f"Project ID: {project_id}")
    tool_names = [t.name for t in runtime.tools]
    print_func(f"Tools:      {', '.join(tool_names) if tool_names else 'None'}")
    print_func("")
    print_func("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print_func("----------------------------------------\n")

    while True:
        try:
            print_func("Enter your prompt:")
            user_input = input_func("> ")
        except (KeyboardInterrupt, EOFError):
            print_func("\n\nExiting development runner. Goodbye!")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ("exit", "quit", "q"):
            print_func("\nExiting development runner. Goodbye!")
            break

        print_func("\nExecuting request...")
        try:
            response = execute_prompt(runtime, cleaned_input, project_id=project_id)

            if response.tool_calls:
                for tc in response.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    print_func(f"  [Tool Invocation: {name}({args})]")

            print_func("\nAgent:")
            print_func(response.output_text)
            print_func("\n" + "-" * 40 + "\n")
        except AgentError as exc:
            print_func(f"\n[Agent Error]: {exc.message}\n")
        except Exception as exc:
            print_func(f"\n[Execution Error]: {exc}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the development runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent MVP Development Runner")
    parser.add_argument(
        "--project-id",
        "-p",
        default=DEFAULT_DEVELOPMENT_PROJECT_ID,
        help=f"Project ID for execution context (default: {DEFAULT_DEVELOPMENT_PROJECT_ID})",
    )
    parser.add_argument(
        "--prompt",
        help="Execute a single prompt and exit without entering interactive loop",
    )
    if argv is None:
        raw_args = sys.argv[1:] if __name__ == "__main__" else []
    else:
        raw_args = list(argv)
    args = parser.parse_args(raw_args)

    try:
        runtime = build_developer_runtime(enable_rag=True)
    except AgentConfigurationError as exc:
        print("\n[Configuration Error]: Unable to start Agent runner.")
        print(f"Details: {exc.message}")
        print("Please check your .env configuration (e.g., OPENROUTER_API_KEY / OPENAI_API_KEY).\n")
        return 1
    except Exception as exc:
        print(f"\n[Initialization Error]: {exc}\n")
        return 1

    if args.prompt:
        print(f"Executing prompt for project '{args.project_id}'...")
        try:
            response = execute_prompt(runtime, args.prompt, project_id=args.project_id)
            if response.tool_calls:
                for tc in response.tool_calls:
                    name = tc.get("name", "unknown")
                    tc_args = tc.get("args", {})
                    print(f"  [Tool Invocation: {name}({tc_args})]")
            print("\nAgent:")
            print(response.output_text)
            return 0
        except AgentError as exc:
            print(f"\n[Agent Error]: {exc.message}\n")
            return 1
        except Exception as exc:
            print(f"\n[Execution Error]: {exc}\n")
            return 1

    interactive_loop(runtime, project_id=args.project_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
