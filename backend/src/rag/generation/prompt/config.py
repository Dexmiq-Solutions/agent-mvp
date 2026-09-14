"""Configuration options for the Prompt Construction stage."""

from dataclasses import dataclass
from typing import Optional

from core.config import Settings, get_settings


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful and precise assistant. Answer the user's query using only the provided retrieved context. "
    "If the context does not contain sufficient information to answer the query, clearly state that you do not have enough information."
)


@dataclass(frozen=True)
class PromptConstructionConfig:
    """Configuration settings for the Prompt Construction stage.

    Attributes:
        default_system_instruction: Default system instructions for RAG generation.
        include_metadata: If True, includes document/chunk identifier metadata in context items.
        include_provenance: If True, includes structural provenance (headings, section path, source).
        context_header: Markdown header separating the retrieved context block.
        query_header: Markdown header separating the original user query block.
        empty_context_text: Text used when context items are empty.
        allow_empty_context: If True, constructs a valid prompt when context is empty; if False, raises validation error.
    """

    default_system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    include_metadata: bool = True
    include_provenance: bool = True
    context_header: str = "## Retrieved Context"
    query_header: str = "## User Query"
    empty_context_text: str = "[No retrieved context provided]"
    allow_empty_context: bool = True

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PromptConstructionConfig":
        """Construct PromptConstructionConfig from application Settings (Single Source of Truth)."""
        app_settings = settings or get_settings()
        sys_inst = getattr(
            app_settings,
            "PROMPT_CONSTRUCTION_DEFAULT_SYSTEM_INSTRUCTION",
            DEFAULT_SYSTEM_INSTRUCTION,
        ) or DEFAULT_SYSTEM_INSTRUCTION
        inc_meta = getattr(app_settings, "PROMPT_CONSTRUCTION_INCLUDE_METADATA", True)
        inc_prov = getattr(app_settings, "PROMPT_CONSTRUCTION_INCLUDE_PROVENANCE", True)
        allow_empty = getattr(app_settings, "PROMPT_CONSTRUCTION_ALLOW_EMPTY_CONTEXT", True)

        return cls(
            default_system_instruction=sys_inst,
            include_metadata=inc_meta,
            include_provenance=inc_prov,
            allow_empty_context=allow_empty,
        )

    @classmethod
    def from_env(cls) -> "PromptConstructionConfig":
        """Construct PromptConstructionConfig from environment settings."""
        return cls.from_settings()
