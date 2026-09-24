"""Agent execution context, state, and request/response models."""

from dataclasses import dataclass, field
from typing import Any, Optional


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
