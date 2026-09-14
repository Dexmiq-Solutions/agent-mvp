"""Domain models for the Prompt Construction stage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConstructedPrompt:
    """Structured, provider-independent constructed prompt representation.

    Logically separates instructions, retrieved context, and the original user query.
    Maintains clean boundaries for downstream inference without coupling to any
    specific LLM provider or orchestration framework.
    """

    system_instruction: str
    user_query: str
    context_text: str
    user_prompt: str
    messages: tuple[dict[str, str], ...]
    project_id: str | None = None
    context_items_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_prompt(self) -> str:
        """Convenience property returning combined text prompt for raw completion models."""
        if self.system_instruction:
            return f"{self.system_instruction}\n\n{self.user_prompt}"
        return self.user_prompt

    def to_messages(self) -> list[dict[str, str]]:
        """Return chat messages list compatible with standard chat completion providers."""
        return [dict(msg) for msg in self.messages]

    def to_dict(self) -> dict[str, Any]:
        """Serialize constructed prompt to a standard dictionary representation."""
        return {
            "system_instruction": self.system_instruction,
            "user_query": self.user_query,
            "context_text": self.context_text,
            "user_prompt": self.user_prompt,
            "messages": list(self.messages),
            "project_id": self.project_id,
            "context_items_count": self.context_items_count,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe debug representation avoiding dumping huge context into logs."""
        query_preview = (
            (self.user_query[:40] + "...") if len(self.user_query) > 40 else self.user_query
        )
        return (
            f"ConstructedPrompt(project_id={self.project_id!r}, "
            f"context_items_count={self.context_items_count}, "
            f"query={query_preview!r})"
        )


GenerationPrompt = ConstructedPrompt
StructuredPrompt = ConstructedPrompt
