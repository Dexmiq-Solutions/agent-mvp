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


from enum import Enum


class ActionSource(str, Enum):
    """Source capability that produced an Action Result."""

    DIRECT_WORK = "direct_work"
    RAG = "rag"
    DELEGATION = "delegation"


@dataclass
class ActionResult:
    """Unified result boundary converging RAG, Direct Work, and Delegation executions.

    Downstream capabilities (such as future Evaluation) consume this common structure
    without needing bespoke handling per execution branch.

    Attributes:
        source: Origin of the action (DIRECT_WORK, RAG, or DELEGATION).
        content: Primary result content produced by the action.
        context: Execution context preserving project identity and boundaries.
        success: Whether the action executed successfully.
        error: Error description if execution failed.
        metadata: Relevant metadata (e.g., task_results for delegation, tool_calls, timing).
    """

    source: ActionSource | str
    content: str
    context: AgentContext = field(default_factory=AgentContext)
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_direct_work(self) -> bool:
        """Return True if the action originated from Direct Work."""
        src_val = self.source.value if isinstance(self.source, ActionSource) else str(self.source)
        return src_val == ActionSource.DIRECT_WORK.value

    @property
    def is_rag(self) -> bool:
        """Return True if the action originated from RAG knowledge retrieval."""
        src_val = self.source.value if isinstance(self.source, ActionSource) else str(self.source)
        return src_val == ActionSource.RAG.value

    @property
    def is_delegation(self) -> bool:
        """Return True if the action originated from Delegation."""
        src_val = self.source.value if isinstance(self.source, ActionSource) else str(self.source)
        return src_val == ActionSource.DELEGATION.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize ActionResult to a dictionary."""
        src_val = self.source.value if isinstance(self.source, ActionSource) else str(self.source)
        return {
            "source": src_val,
            "content": self.content,
            "context": {
                "project_id": self.context.project_id,
                "conversation_id": self.context.conversation_id,
                "user_id": self.context.user_id,
                "metadata": dict(self.context.metadata),
            },
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResult":
        """Deserialize a dictionary into an ActionResult instance."""
        ctx_data = data.get("context", {})
        ctx = AgentContext(
            project_id=ctx_data.get("project_id"),
            conversation_id=ctx_data.get("conversation_id"),
            user_id=ctx_data.get("user_id"),
            metadata=dict(ctx_data.get("metadata", {})),
        )
        raw_source = data.get("source", ActionSource.DIRECT_WORK.value)
        try:
            source = ActionSource(raw_source)
        except ValueError:
            source = raw_source
        return cls(
            source=source,
            content=data.get("content", ""),
            context=ctx,
            success=data.get("success", True),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )


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
    action_result: Optional[ActionResult] = None

    @property
    def is_direct_work(self) -> bool:
        """Return True if execution completed directly without external tool calls or delegation."""
        if self.action_result is not None:
            return self.action_result.is_direct_work
        return len(self.tool_calls) == 0

    @property
    def used_rag(self) -> bool:
        """Return True if search_project_knowledge was invoked during execution."""
        if self.action_result is not None and self.action_result.is_rag:
            return True
        return any(tc.get("name") == "search_project_knowledge" for tc in self.tool_calls)

    @property
    def is_delegation(self) -> bool:
        """Return True if execution was handled via Delegation."""
        if self.action_result is not None:
            return self.action_result.is_delegation
        return False

    def to_action_result(self) -> ActionResult:
        """Produce a normalized ActionResult representing this execution response."""
        if self.action_result is not None:
            return self.action_result

        source = ActionSource.RAG if self.used_rag else ActionSource.DIRECT_WORK
        return ActionResult(
            source=source,
            content=self.output_text,
            context=self.context,
            success=self.success,
            error=self.error,
            metadata={"tool_calls": self.tool_calls, "model": self.model},
        )
