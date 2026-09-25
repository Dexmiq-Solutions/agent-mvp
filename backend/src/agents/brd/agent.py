"""BRD Lead Agent implementation.

Establishes the BRD Lead Agent as the domain-specific Agent responsible for the
Business Requirements Document (BRD) generation objective.

Loads its foundational behavioral system instruction from the authoritative
external Markdown artifact (system_instruction.md).
Loads its required BRD document structure from the authoritative
external Markdown artifact (brd_template.md).
Maintains its working context and progress via BRDAgentState.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from agents.brd.state import BRDAgentState
from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest, AgentRunResponse
from observability.logging import get_logger
from services.rag_service import RAGService
from tools import get_default_tools
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import create_search_project_knowledge_tool, search_project_knowledge

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"
BRD_TEMPLATE_FILE = "brd_template.md"


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


def get_brd_template_path() -> Path:
    """Resolve the absolute path to the authoritative BRD template Markdown file."""
    return Path(__file__).resolve().parent / BRD_TEMPLATE_FILE


def load_brd_template() -> str:
    """Load the authoritative BRD template from Markdown.

    Returns:
        str: Raw Markdown content of the BRD template.

    Raises:
        FileNotFoundError: If the BRD template Markdown file does not exist.
        ValueError: If the BRD template Markdown file is empty.
    """
    template_path = get_brd_template_path()
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Required BRD template file not found: {template_path}"
        )

    try:
        content = template_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read BRD template from %s: %s", template_path, exc)
        raise

    if not content:
        raise ValueError(
            f"BRD template file is empty: {template_path}"
        )

    return content


def extract_brd_sections(content: Optional[str] = None) -> list[str]:
    """Extract required BRD sections dynamically from the BRD template Markdown.

    The Markdown template is the single source of truth for the required BRD structure.
    Top-level section headings (level-2 markdown headers: '## <heading>') define the required
    sections in exact document order without duplicating them in Python code.

    Args:
        content: Optional raw Markdown string. If omitted, loaded from load_brd_template().

    Returns:
        list[str]: Ordered list of section headings extracted from the template.
    """
    raw_content = load_brd_template() if content is None else content
    matches = re.findall(r"^##\s+(.+)$", raw_content, re.MULTILINE)
    return [m.strip() for m in matches if m.strip()]


class _SystemInstructionDescriptor:
    """Descriptor providing access to system instruction on both instances and the class."""

    def __get__(self, instance: Optional["BRDLeadAgent"], owner: Optional[type] = None) -> str:
        if instance is not None and hasattr(instance, "_system_instruction"):
            return instance._system_instruction
        return load_system_instruction()


class _BRDTemplateDescriptor:
    """Descriptor providing access to BRD template on both instances and the class."""

    def __get__(self, instance: Optional["BRDLeadAgent"], owner: Optional[type] = None) -> str:
        if instance is not None and hasattr(instance, "_template"):
            return instance._template
        return load_brd_template()


class BRDLeadAgent:
    """Domain-specific Agent responsible for accomplishing the BRD generation objective.

    The BRD Lead Agent is the primary agent for the BRD generation use case.
    It operates across three foundational pillars:
    1. System Instruction (system_instruction.md): Defines how the Agent behaves.
    2. BRD Template (brd_template.md): Defines what the Agent must produce.
    3. Agent State (BRDAgentState): Defines what the Agent currently knows and remembers.

    It executes within the DeepAgents execution harness managed by AgentRuntime,
    operating within application-provided project boundaries and utilizing existing
    agent tools (including RAG knowledge retrieval).
    """

    agent_name: str = "BRDLeadAgent"
    system_instruction: str = _SystemInstructionDescriptor()  # type: ignore[assignment]
    default_system_prompt: str = _SystemInstructionDescriptor()  # type: ignore[assignment]
    brd_template: str = _BRDTemplateDescriptor()  # type: ignore[assignment]
    template: str = _BRDTemplateDescriptor()  # type: ignore[assignment]

    def __init__(
        self,
        runtime: Optional[AgentRuntime] = None,
        config: Optional[AgentConfig] = None,
        model: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        state: Optional[BRDAgentState] = None,
        rag_service: Optional[RAGService] = None,
        enable_rag: bool = True,
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
            template: Optional BRD template override. If omitted, loaded from brd_template.md.
            state: Optional BRDAgentState working state instance. If omitted, initialized
                from the template structure.
            rag_service: Optional pre-configured RAGService instance to inject for knowledge retrieval.
            enable_rag: Whether to equip the search_project_knowledge tool when constructing runtime.
                Defaults to True.
        """
        resolved_instruction = system_instruction or system_prompt or load_system_instruction()
        self._system_instruction = resolved_instruction

        resolved_template = template or load_brd_template()
        self._template = resolved_template

        # Initialize working state from template structure if not injected
        if state is not None:
            self._state = state
        else:
            self._state = BRDAgentState.initialize_from_template(sections=self.sections)

        if runtime is not None:
            self._runtime = runtime
            # Configure runtime system instruction if not already set
            if getattr(self._runtime, "_system_prompt", None) != resolved_instruction:
                self._runtime._system_prompt = resolved_instruction
        else:
            if tools is not None:
                resolved_tools = list(tools)
            elif not enable_rag:
                resolved_tools = [echo_diagnostic_tool]
            elif rag_service is not None:
                resolved_tools = [
                    echo_diagnostic_tool,
                    create_search_project_knowledge_tool(rag_service=rag_service),
                ]
            else:
                resolved_tools = get_default_tools()

            self._runtime = AgentRuntime(
                config=config,
                model=model,
                tools=resolved_tools,
                system_prompt=resolved_instruction,
            )

    @property
    def has_rag_capability(self) -> bool:
        """Return True if the search_project_knowledge tool is equipped on this agent."""
        return any(
            getattr(tool, "name", "") == "search_project_knowledge"
            for tool in self.tools
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

    @property
    def sections(self) -> list[str]:
        """Return ordered required BRD sections dynamically extracted from the template."""
        return extract_brd_sections(self._template)

    @property
    def state(self) -> BRDAgentState:
        """Return the active BRDAgentState working context."""
        return self._state

    def reset_state(self) -> None:
        """Reset the Agent working state to initial unstarted state based on template sections."""
        self._state = BRDAgentState.initialize_from_template(sections=self.sections)

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
            AgentRunResponse: Normalized response containing output text, artifacts, and state.
        """
        # Adopt or normalize incoming state if passed in request
        if isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

        # Preserve project context in state metadata
        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx and effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        response = self._runtime.execute(request=request, context=context)
        response.state = self._state
        return response

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
            AgentRunResponse: Normalized response containing output text, artifacts, and state.
        """
        if isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx and effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        response = await self._runtime.execute_async(request=request, context=context)
        response.state = self._state
        return response


def create_brd_lead_agent(
    runtime: Optional[AgentRuntime] = None,
    config: Optional[AgentConfig] = None,
    model: Optional[BaseChatModel] = None,
    tools: Optional[Sequence[Any]] = None,
    system_instruction: Optional[str] = None,
    system_prompt: Optional[str] = None,
    template: Optional[str] = None,
    state: Optional[BRDAgentState] = None,
    rag_service: Optional[RAGService] = None,
    enable_rag: bool = True,
) -> BRDLeadAgent:
    """Factory function to instantiate the BRD Lead Agent.

    Args:
        runtime: Optional existing AgentRuntime instance.
        config: Optional AgentConfig instance.
        model: Optional pre-configured BaseChatModel.
        tools: Optional sequence of tools.
        system_instruction: Optional system instruction override.
        system_prompt: Deprecated alias for system_instruction for backward compatibility.
        template: Optional BRD template override.
        state: Optional BRDAgentState working state instance.
        rag_service: Optional pre-configured RAGService instance to inject for knowledge retrieval.
        enable_rag: Whether to equip RAG knowledge retrieval tool (defaults to True).

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
        template=template,
        state=state,
        rag_service=rag_service,
        enable_rag=enable_rag,
    )
