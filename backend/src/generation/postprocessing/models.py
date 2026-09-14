"""Domain models for post-processing generation results."""

from dataclasses import dataclass, field
from typing import Any

from generation.llm.models import LLMUsage


@dataclass(frozen=True)
class ProcessedResponse:
    """Application-level normalized response produced by the Post-processing stage.

    Represents the stable, provider-independent output of generation ready
    for downstream Groundedness / Safety Checks and API presentation.
    Faithfully encapsulates extracted content, normalized finish reasons,
    token usage metrics, and execution metadata without leaking vendor SDK structures.
    """

    content: str
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    model: str | None = None
    structured_output: Any | None = None
    raw_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Alias for content providing conceptual alignment with text completion consumers."""
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalized response to a standard dictionary representation."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
            "structured_output": self.structured_output,
            "raw_content": self.raw_content,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe debug representation preventing log dumps of sensitive or unbounded output."""
        content_preview = (
            (self.content[:40] + "...") if len(self.content) > 40 else self.content
        )
        tokens_info = f"tokens={self.usage.total_tokens}" if self.usage else "tokens=N/A"
        return (
            f"ProcessedResponse(model={self.model!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"{tokens_info}, "
            f"preview={content_preview!r})"
        )


PostProcessedResponse = ProcessedResponse
