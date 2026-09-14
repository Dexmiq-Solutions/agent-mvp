"""Domain models for LLM Interface results, usage, and streaming events."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMUsage:
    """Normalized token usage information returned by an LLM provider.

    Fields are None when the provider does not report token counts.
    Values are never fabricated.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        """Serialize usage metrics to a standard dictionary representation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResult:
    """Application-level normalized result of an LLM generation call.

    Exposes the raw model-generated content without semantic post-processing
    (no citations, summarizing, or BRD transformation). Encapsulates provider-specific
    responses inside the adapter layer.
    """

    content: str
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def generated_content(self) -> str:
        """Alias for content providing conceptual alignment with generation specs."""
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalized LLM result to dictionary representation."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe representation preventing logging of sensitive or unbounded output."""
        content_preview = (
            (self.content[:40] + "...") if len(self.content) > 40 else self.content
        )
        tokens_info = f"tokens={self.usage.total_tokens}" if self.usage else "tokens=N/A"
        return (
            f"LLMResult(model={self.metadata.get('model')!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"{tokens_info}, "
            f"preview={content_preview!r})"
        )


@dataclass(frozen=True)
class LLMStreamEvent:
    """Normalized event yielded during streaming token generation."""

    delta: str
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        """Return True if this event marks stream completion."""
        return self.finish_reason is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize stream event to standard dictionary representation."""
        return {
            "delta": self.delta,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "metadata": dict(self.metadata),
        }
