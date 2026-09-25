"""BRD Lead Agent implementation.

Establishes the BRD Lead Agent as the domain-specific Agent responsible for the
Business Requirements Document (BRD) generation objective.
"""

from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest, AgentRunResponse
from observability.logging import get_logger
from tools import get_default_tools

logger = get_logger(__name__)


class BRDLeadAgent:
    """Domain-specific Agent responsible for accomplishing the BRD generation objective.

    The BRD Lead Agent is the primary agent for the BRD generation use case.
    It executes within the DeepAgents execution harness managed by AgentRuntime,
    operating within application-provided project boundaries and utilizing existing
    agent tools (including RAG knowledge retrieval).
    """

    agent_name: str = "BRDLeadAgent"
    default_system_prompt: str = (
        "You are the BRD Lead Agent, responsible for the Business Requirements Document generation objective."
    )

    def __init__(
        self,
        runtime: Optional[AgentRuntime] = None,
        config: Optional[AgentConfig] = None,
        model: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the BRD Lead Agent.

        Args:
            runtime: Optional existing AgentRuntime harness. If provided, runtime execution
                is delegated to this instance.
            config: Optional AgentConfig instance. Used if runtime is not provided.
            model: Optional pre-configured BaseChatModel. Used if runtime is not provided.
            tools: Optional sequence of tools. Defaults to get_default_tools() if runtime is not provided.
            system_prompt: Optional system prompt override. Defaults to default_system_prompt.
        """
        if runtime is not None:
            self._runtime = runtime
        else:
            resolved_tools = list(tools) if tools is not None else get_default_tools()
            resolved_prompt = system_prompt or self.default_system_prompt
            self._runtime = AgentRuntime(
                config=config,
                model=model,
                tools=resolved_tools,
                system_prompt=resolved_prompt,
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
    system_prompt: Optional[str] = None,
) -> BRDLeadAgent:
    """Factory function to instantiate the BRD Lead Agent.

    Args:
        runtime: Optional existing AgentRuntime instance.
        config: Optional AgentConfig instance.
        model: Optional pre-configured BaseChatModel.
        tools: Optional sequence of tools.
        system_prompt: Optional system prompt override.

    Returns:
        Configured BRDLeadAgent instance.
    """
    return BRDLeadAgent(
        runtime=runtime,
        config=config,
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
