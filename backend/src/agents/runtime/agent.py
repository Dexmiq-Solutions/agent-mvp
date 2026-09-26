"""Agent runtime harness encapsulating DeepAgents and model execution."""

import time
from typing import Any, Optional, Sequence

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agents.runtime.config import AgentConfig
from agents.runtime.model import create_agent_model
from agents.runtime.state import (
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
    reset_current_agent_context,
    set_current_agent_context,
)
from exceptions.agent import (
    AgentConfigurationError,
    AgentExecutionError,
    AgentInitializationError,
    AgentModelError,
)
from observability.logging import get_logger

logger = get_logger(__name__)


def create_runtime_agent(
    model: BaseChatModel,
    tools: Optional[Sequence[Any]] = None,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Initialize and compile a DeepAgents agent execution graph.

    Args:
        model: LangChain-compatible chat model with tool-binding capability.
        tools: Optional sequence of tools available to the agent.
        system_prompt: Optional base system instruction for the agent.
        **kwargs: Additional parameters forwarded to create_deep_agent.

    Returns:
        Compiled DeepAgents state graph runnable.

    Raises:
        AgentInitializationError: If graph creation or compilation fails.
    """
    try:
        agent_tools = list(tools) if tools else []
        agent = create_deep_agent(
            model=model,
            tools=agent_tools,
            system_prompt=system_prompt,
            **kwargs,
        )
        return agent
    except Exception as exc:
        logger.error("Failed to initialize DeepAgents runtime agent: %s", exc)
        raise AgentInitializationError(
            f"Failed to initialize DeepAgents runtime: {exc}",
            original_error=exc,
        ) from exc


class AgentRuntime:
    """Execution harness managing agent lifecycle, context boundaries, and tool orchestration.

    Encapsulates DeepAgents and provider-specific details behind a clean, application-facing boundary.
    Enforces project isolation by associating each execution with an AgentContext.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        model: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_prompt: Optional[str] = None,
        store: Optional[Any] = None,
        backend: Optional[Any] = None,
        memory: Optional[Sequence[str]] = None,
        permissions: Optional[Sequence[Any]] = None,
        middleware: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AgentRuntime.

        Args:
            config: Optional AgentConfig instance. If omitted, constructed from application Settings.
            model: Optional pre-configured BaseChatModel instance.
            tools: Optional sequence of tools to equip the agent with.
            system_prompt: Optional system prompt override.
            store: Optional LangGraph BaseStore instance for long-term memory.
            backend: Optional DeepAgents BackendProtocol instance (e.g. StoreBackend).
            memory: Optional sequence of memory file paths (e.g. ['/memory/project_context.md']).
            permissions: Optional sequence of FilesystemPermission rules (e.g. read-only enforcement).
            middleware: Optional sequence of custom DeepAgents/LangChain middleware.
            **kwargs: Additional parameters forwarded to create_runtime_agent / create_deep_agent.
        """
        self._config = config or AgentConfig.from_settings()
        self._tools = list(tools) if tools else []
        self._system_prompt = system_prompt or self._config.system_prompt
        self._store = store
        self._backend = backend
        self._memory = list(memory) if memory is not None else None
        self._permissions = list(permissions) if permissions is not None else None
        self._middleware = list(middleware) if middleware is not None else []

        # Initialize model if not injected
        self._model = model or create_agent_model(self._config)

        # Build DeepAgents execution harness
        runtime_kwargs: dict[str, Any] = {**kwargs}
        if self._store is not None:
            runtime_kwargs["store"] = self._store
        if self._backend is not None:
            runtime_kwargs["backend"] = self._backend
        if self._memory is not None:
            runtime_kwargs["memory"] = self._memory
        if self._permissions is not None:
            runtime_kwargs["permissions"] = self._permissions
        if self._middleware:
            runtime_kwargs["middleware"] = self._middleware

        self._graph = create_runtime_agent(
            model=self._model,
            tools=self._tools,
            system_prompt=self._system_prompt,
            **runtime_kwargs,
        )

    @property
    def config(self) -> AgentConfig:
        """Return the active Agent configuration."""
        return self._config

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools."""
        return list(self._tools)

    @property
    def graph(self) -> Any:
        """Return the compiled DeepAgents graph."""
        return self._graph

    @property
    def store(self) -> Optional[Any]:
        """Return the attached LangGraph BaseStore instance."""
        return self._store

    @property
    def backend(self) -> Optional[Any]:
        """Return the attached storage backend instance."""
        return self._backend

    @property
    def memory(self) -> Optional[list[str]]:
        """Return the configured memory sources if memory is enabled."""
        return list(self._memory) if self._memory is not None else None

    @property
    def permissions(self) -> Optional[list[Any]]:
        """Return the configured filesystem permission rules."""
        return list(self._permissions) if self._permissions is not None else None

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt/instruction configured for the runtime."""
        return self._system_prompt

    @property
    def system_instruction(self) -> str:
        """Return the active system instruction configured for the runtime."""
        return self._system_prompt

    def create_brd_lead_agent(
        self,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        state: Optional[Any] = None,
        store: Optional[Any] = None,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a domain-specific BRD Lead Agent backed by this runtime harness.

        Args:
            system_instruction: Optional system instruction override.
            system_prompt: Deprecated alias for system_instruction for backward compatibility.
            template: Optional BRD template override.
            state: Optional BRDAgentState working state instance.
            store: Optional LangGraph BaseStore instance.
            project_id: Optional project identifier for project-scoped memory.
            **kwargs: Additional parameters forwarded to BRDLeadAgent.

        Returns:
            BRDLeadAgent: Initialized BRD Lead Agent.
        """
        from agents.brd.agent import BRDLeadAgent

        return BRDLeadAgent(
            runtime=self,
            system_instruction=system_instruction,
            system_prompt=system_prompt,
            template=template,
            state=state,
            store=store or self._store,
            project_id=project_id,
            **kwargs,
        )

    def execute(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
    ) -> AgentRunResponse:
        """Execute a synchronous agent interaction cycle.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext (used if request is a string).

        Returns:
            AgentRunResponse: Normalized response containing output text and execution artifacts.

        Raises:
            AgentExecutionError: If execution fails.
        """
        run_req, ctx = self._normalize_request(request, context)
        start_time = time.perf_counter()
        token = set_current_agent_context(ctx)

        logger.info(
            "Agent execution started (model: %s, project_id: %s, prompt_len: %d)",
            self._config.model,
            ctx.project_id or "unspecified",
            len(run_req.input_text),
        )

        try:
            inputs = {"messages": [HumanMessage(content=run_req.input_text)]}
            result = self._graph.invoke(inputs)
            response = self._process_result(result, ctx, start_time)

            logger.info(
                "Agent execution completed (model: %s, duration: %.2fs, tool_calls: %d)",
                self._config.model,
                time.perf_counter() - start_time,
                len(response.tool_calls),
            )
            return response
        except (AgentConfigurationError, AgentInitializationError, AgentModelError):
            raise
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(
                "Agent execution failure after %.2fs: %s",
                duration,
                exc,
                exc_info=True,
            )
            raise AgentExecutionError(
                f"Agent execution failed: {exc}",
                original_error=exc,
            ) from exc
        finally:
            reset_current_agent_context(token)

    async def execute_async(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
    ) -> AgentRunResponse:
        """Execute an asynchronous agent interaction cycle.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext (used if request is a string).

        Returns:
            AgentRunResponse: Normalized response containing output text and execution artifacts.

        Raises:
            AgentExecutionError: If execution fails.
        """
        run_req, ctx = self._normalize_request(request, context)
        start_time = time.perf_counter()
        token = set_current_agent_context(ctx)

        logger.info(
            "Agent execution started [async] (model: %s, project_id: %s, prompt_len: %d)",
            self._config.model,
            ctx.project_id or "unspecified",
            len(run_req.input_text),
        )

        try:
            inputs = {"messages": [HumanMessage(content=run_req.input_text)]}
            result = await self._graph.ainvoke(inputs)
            response = self._process_result(result, ctx, start_time)

            logger.info(
                "Agent execution completed [async] (model: %s, duration: %.2fs, tool_calls: %d)",
                self._config.model,
                time.perf_counter() - start_time,
                len(response.tool_calls),
            )
            return response
        except (AgentConfigurationError, AgentInitializationError, AgentModelError):
            raise
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(
                "Agent execution failure [async] after %.2fs: %s",
                duration,
                exc,
                exc_info=True,
            )
            raise AgentExecutionError(
                f"Agent execution failed: {exc}",
                original_error=exc,
            ) from exc
        finally:
            reset_current_agent_context(token)

    def _normalize_request(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext],
    ) -> tuple[AgentRunRequest, AgentContext]:
        """Normalize raw input or AgentRunRequest into validated models."""
        if isinstance(request, str):
            ctx = context or AgentContext()
            run_req = AgentRunRequest(input_text=request, context=ctx)
        else:
            run_req = request
            ctx = run_req.context or context or AgentContext()
        return run_req, ctx

    def _process_result(
        self,
        result: dict[str, Any],
        context: AgentContext,
        start_time: float,
    ) -> AgentRunResponse:
        """Parse raw graph execution state into a normalized AgentRunResponse."""
        messages = result.get("messages", [])
        output_text = ""
        tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc)

        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", "")
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                # LangChain content parts
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                output_text = "".join(text_parts)
            else:
                output_text = str(content)

        return AgentRunResponse(
            output_text=output_text,
            messages=messages,
            tool_calls=tool_calls,
            context=context,
            model=self._config.model,
            success=True,
        )
