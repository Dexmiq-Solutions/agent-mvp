"""BRD Lead Agent implementation.

Establishes the BRD Lead Agent as the domain-specific Agent responsible for the
Business Requirements Document (BRD) generation objective.

Loads its foundational behavioral system instruction from the authoritative
external Markdown artifact (system_instruction.md).
"""

from pathlib import Path
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest, AgentRunResponse
from observability.logging import get_logger
from tools import get_default_tools

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"


def get_system_instruction_path() -> Path:
    """Resolve the absolute path to the BRD system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_system_instruction() -> str:
    """Load the authoritative BRD Lead Agent system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    instruction_path = get_system_instruction_path()
    if not instruction_path.is_file():
        raise FileNotFoundError(
            f"Required BRD Lead Agent system instruction file not found: {instruction_path}"
        )

    try:
        content = instruction_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read BRD system instruction from %s: %s", instruction_path, exc)
        raise

    if not content:
        raise ValueError(
            f"BRD Lead Agent system instruction file is empty: {instruction_path}"
        )

    return content


class _SystemInstructionDescriptor:
    """Descriptor providing access to system instruction on both instances and the class."""

    def __get__(self, instance: Optional["BRDLeadAgent"], owner: Optional[type] = None) -> str:
        if instance is not None and hasattr(instance, "_system_instruction"):
            return instance._system_instruction
        return load_system_instruction()


class BRDLeadAgent:
    """Domain-specific Agent responsible for accomplishing the BRD generation objective.

    The BRD Lead Agent is the primary agent for the BRD generation use case.
    It executes within the DeepAgents execution harness managed by AgentRuntime,
    operating within application-provided project boundaries and utilizing existing
    agent tools (including RAG knowledge retrieval).

    The Agent's foundational behavioral identity, objective, and evidence principles
    are defined in its authoritative external system instruction (system_instruction.md).
    """

    agent_name: str = "BRDLeadAgent"
    system_instruction: str = _SystemInstructionDescriptor()  # type: ignore[assignment]
    default_system_prompt: str = _SystemInstructionDescriptor()  # type: ignore[assignment]

    def __init__(
        self,
        runtime: Optional[AgentRuntime] = None,
        config: Optional[AgentConfig] = None,
        model: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the BRD Lead Agent.

        Args:
            runtime: Optional existing AgentRuntime harness. If provided, runtime execution
                is delegated to this instance.
            config: Optional AgentConfig instance. Used if runtime is not provided.
            model: Optional pre-configured BaseChatModel. Used if runtime is not provided.
            tools: Optional sequence of tools. Defaults to get_default_tools() if runtime is not provided.
            system_instruction: Optional system instruction override. If omitted, loaded from
                system_instruction.md.
            system_prompt: Deprecated alias for system_instruction for backward compatibility.
        """
        resolved_instruction = system_instruction or system_prompt or load_system_instruction()
        self._system_instruction = resolved_instruction

        if runtime is not None:
            self._runtime = runtime
            # Configure runtime system instruction if not already set
            if getattr(self._runtime, "_system_prompt", None) != resolved_instruction:
                self._runtime._system_prompt = resolved_instruction
        else:
            resolved_tools = list(tools) if tools is not None else get_default_tools()
            self._runtime = AgentRuntime(
                config=config,
                model=model,
                tools=resolved_tools,
                system_prompt=resolved_instruction,
            )

    @property
    def runtime(self) -> AgentRuntime:
        """Return the underlying AgentRuntime execution harness."""
        return self._runtime

    @property
    def config(self) -> AgentConfig:
        """Return the active Agent configuration."""
        return self._runtime.config

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._runtime.model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools."""
        return self._runtime.tools

    @property
    def graph(self) -> Any:
        """Return the compiled DeepAgents graph."""
        return self._runtime.graph

    def execute(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
    ) -> AgentRunResponse:
        """Execute a synchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.

        Returns:
            AgentRunResponse: Normalized response containing output text and execution artifacts.
        """
        return self._runtime.execute(request=request, context=context)

    async def execute_async(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
    ) -> AgentRunResponse:
        """Execute an asynchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.

        Returns:
            AgentRunResponse: Normalized response containing output text and execution artifacts.
        """
        return await self._runtime.execute_async(request=request, context=context)


def create_brd_lead_agent(
    runtime: Optional[AgentRuntime] = None,
    config: Optional[AgentConfig] = None,
    model: Optional[BaseChatModel] = None,
    tools: Optional[Sequence[Any]] = None,
    system_instruction: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> BRDLeadAgent:
    """Factory function to instantiate the BRD Lead Agent.

    Args:
        runtime: Optional existing AgentRuntime instance.
        config: Optional AgentConfig instance.
        model: Optional pre-configured BaseChatModel.
        tools: Optional sequence of tools.
        system_instruction: Optional system instruction override.
        system_prompt: Deprecated alias for system_instruction for backward compatibility.

    Returns:
        Configured BRDLeadAgent instance.
    """
    return BRDLeadAgent(
        runtime=runtime,
        config=config,
        model=model,
        tools=tools,
        system_instruction=system_instruction,
        system_prompt=system_prompt,
    )
