"""Agent execution context, state, and request/response models."""

import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional

_current_agent_context: contextvars.ContextVar[Optional["AgentContext"]] = (
    contextvars.ContextVar("current_agent_context", default=None)
)


def get_current_agent_context() -> Optional["AgentContext"]:
    """Retrieve the active AgentContext from context-local storage."""
    return _current_agent_context.get()


def set_current_agent_context(context: Optional["AgentContext"]) -> contextvars.Token:
    """Set the active AgentContext in context-local storage for the current execution context."""
    return _current_agent_context.set(context)


def reset_current_agent_context(token: contextvars.Token) -> None:
    """Reset the active AgentContext in context-local storage using the provided token."""
    _current_agent_context.reset(token)


@dataclass(frozen=True)
class AgentContext:
    """Execution context preserving tenancy and operational boundaries.

    Crucial for architectural integrity:
    Ensures that the LLM is never the authority for selecting tenant/project boundaries.
    The application supplies the project_id, which flows into the context and to subsequent tools.
    """

    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunRequest:
    """Request payload for an agent interaction cycle."""

    input_text: str
    context: AgentContext = field(default_factory=AgentContext)
    system_prompt: Optional[str] = None
    state: Optional[Any] = None


@dataclass
class AgentRunResponse:
    """Normalized response payload produced by the agent runtime."""

    output_text: str
    messages: list[Any] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    context: AgentContext = field(default_factory=AgentContext)
    model: str = ""
    success: bool = True
    error: Optional[str] = None
    state: Optional[Any] = None

    @property
    def is_direct_work(self) -> bool:
        """Return True if execution completed directly without external tool calls."""
        return len(self.tool_calls) == 0

    @property
    def used_rag(self) -> bool:
        """Return True if search_project_knowledge was invoked during execution."""
        return any(tc.get("name") == "search_project_knowledge" for tc in self.tool_calls)
