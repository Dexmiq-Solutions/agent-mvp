"""Developer manual Agent runner for interactive terminal testing and debugging.

Provides a lightweight, interactive developer-facing entry point to exercise the
Phase 2 DeepAgents + OpenRouter runtime foundation.

Usage:
    From the backend directory:
    uv run python scripts/run_agent.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Ensure src/ is on the Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.brd import BRDLeadAgent, create_brd_lead_agent
from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunResponse
from exceptions.agent import AgentConfigurationError, AgentError
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import search_project_knowledge

DEFAULT_DEVELOPMENT_PROJECT_ID = "7b155ced-728d-4569-b660-6c223167f295"


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


def build_brd_lead_agent(
    config: Optional[AgentConfig] = None,
    tools: Optional[Sequence[Any]] = None,
    enable_rag: bool = True,
) -> BRDLeadAgent:
    """Build and initialize the BRDLeadAgent equipped with developer diagnostic tools and RAG.

    Args:
        config: Optional AgentConfig instance. Defaults to loading from application settings.
        tools: Optional explicit sequence of tools to equip.
        enable_rag: If True, equips search_project_knowledge alongside echo_diagnostic_tool.
            Defaults to True for the BRD Lead Agent.

    Returns:
        Configured BRDLeadAgent instance.
    """
    runtime = build_developer_runtime(config=config, tools=tools, enable_rag=enable_rag)
    return runtime.create_brd_lead_agent()



def execute_prompt(
    agent: Any,
    prompt: str,
    project_id: str = DEFAULT_DEVELOPMENT_PROJECT_ID,
) -> AgentRunResponse:
    """Execute a single prompt synchronously against the Agent runtime or BRD Lead Agent.

    Args:
        agent: The active AgentRuntime or BRDLeadAgent instance.
        prompt: The user prompt string.
        project_id: Explicit project identifier for project isolation.

    Returns:
        AgentRunResponse: The normalized response from the Agent.
    """
    context = AgentContext(project_id=project_id)
    return agent.execute(prompt, context=context)


async def execute_prompt_async(
    agent: Any,
    prompt: str,
    project_id: str = DEFAULT_DEVELOPMENT_PROJECT_ID,
) -> AgentRunResponse:
    """Execute a single prompt asynchronously against the Agent runtime or BRD Lead Agent.

    Args:
        agent: The active AgentRuntime or BRDLeadAgent instance.
        prompt: The user prompt string.
        project_id: Explicit project identifier for project isolation.

    Returns:
        AgentRunResponse: The normalized response from the Agent.
    """
    context = AgentContext(project_id=project_id)
    return await agent.execute_async(prompt, context=context)


async def interactive_loop(
    agent: Any,
    project_id: str = DEFAULT_DEVELOPMENT_PROJECT_ID,
    input_func: Callable[[str], Any] = input,
    print_func: Callable[..., None] = print,
) -> None:
    """Run an interactive prompt-response loop asynchronously with the developer.

    Args:
        agent: The active AgentRuntime or BRDLeadAgent instance.
        project_id: Explicit project identifier for project isolation.
        input_func: Callable for reading user input (injected for testing).
        print_func: Callable for printing output (injected for testing).
    """
    agent_name = getattr(agent, "agent_name", "AgentRuntime")
    print_func("========================================")
    print_func(" Agent MVP - Development Runner")
    print_func("========================================")
    print_func(f"Agent:      {agent_name}")
    print_func(f"Model:      {agent.config.model}")
    print_func(f"Project ID: {project_id}")
    tool_names = [t.name for t in agent.tools]
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

        cleaned_input = str(user_input).strip() if user_input is not None else ""
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ("exit", "quit", "q"):
            print_func("\nExiting development runner. Goodbye!")
            break

        print_func("\nExecuting request...")
        try:
            response = await execute_prompt_async(agent, cleaned_input, project_id=project_id)

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


async def main_async(argv: Optional[Sequence[str]] = None) -> int:
    """Core asynchronous execution entry point for the development runner."""
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
    parser.add_argument(
        "--agent-type",
        choices=["brd", "runtime"],
        default="brd",
        help="Agent architecture type to run: 'brd' (BRD Lead Agent) or 'runtime' (bare AgentRuntime) (default: brd)",
    )
    if argv is None:
        raw_args = sys.argv[1:] if __name__ == "__main__" else []
    else:
        raw_args = list(argv)
    args = parser.parse_args(raw_args)

    try:
        runtime = build_developer_runtime(enable_rag=True)
        agent = runtime.create_brd_lead_agent() if args.agent_type == "brd" else runtime
    except AgentConfigurationError as exc:
        print("\n[Configuration Error]: Unable to start Agent runner.")
        print(f"Details: {exc.message}")
        print("Please check your .env configuration (e.g., OPENROUTER_API_KEY / OPENAI_API_KEY).\n")
        return 1
    except Exception as exc:
        print(f"\n[Initialization Error]: {exc}\n")
        return 1

    try:
        if args.prompt:
            print(f"Executing prompt for project '{args.project_id}' ({getattr(agent, 'agent_name', 'AgentRuntime')})...")
            try:
                response = await execute_prompt_async(agent, args.prompt, project_id=args.project_id)
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

        await interactive_loop(agent, project_id=args.project_id)
        return 0
    finally:
        from storage.vector.client import close_async_qdrant_client
        await close_async_qdrant_client()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the development runner."""
    try:
        return asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        print("\n\nExiting development runner. Goodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
