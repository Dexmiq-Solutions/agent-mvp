"""Configuration settings for the Context Formatting stage of the RAG retrieval pipeline."""

from dataclasses import dataclass
from typing import Optional

from core.config import Settings, get_settings

DEFAULT_CONTEXT_HEADER = "RETRIEVED CONTEXT"
DEFAULT_CONTEXT_FOOTER = "END RETRIEVED CONTEXT"
DEFAULT_EMPTY_CONTEXT_TEXT = "[No retrieved context provided]"
DEFAULT_ITEM_TEMPLATE = "[Context {index}]"


@dataclass(frozen=True)
class ContextFormattingConfig:
    """Configuration options governing Context Formatting behavior and representation.

    Attributes:
        strategy: Formatting strategy identifier (default: "text").
        context_header: Overall boundary header preceding the formatted context block.
        context_footer: Overall boundary footer concluding the formatted context block.
        item_label_template: Label template for individual context items (e.g. "[Context {index}]").
        include_header: If True, renders context_header before items.
        include_footer: If True, renders context_footer after items.
        include_metadata: If True, formats document and chunk identity coordinates.
        include_provenance: If True, formats section path, heading, and source details.
        empty_context_text: Placeholder text when context contains zero items.
        allow_empty_context: If True, formats empty context without raising an error.
            If False, raises ContextFormattingValidationError.
        strict_project_validation: If True, raises validation error on cross-tenant items.
            If False, discards cross-tenant items with an observability warning.
    """

    strategy: str = "text"
    context_header: str = DEFAULT_CONTEXT_HEADER
    context_footer: str = DEFAULT_CONTEXT_FOOTER
    item_label_template: str = DEFAULT_ITEM_TEMPLATE
    include_header: bool = True
    include_footer: bool = True
    include_metadata: bool = True
    include_provenance: bool = True
    empty_context_text: str = DEFAULT_EMPTY_CONTEXT_TEXT
    allow_empty_context: bool = True
    strict_project_validation: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "ContextFormattingConfig":
        """Construct ContextFormattingConfig from application Settings (Single Source of Truth)."""
        app_settings = settings or get_settings()

        strategy = getattr(app_settings, "CONTEXT_FORMATTING_STRATEGY", "text") or "text"
        header = getattr(
            app_settings,
            "CONTEXT_FORMATTING_HEADER",
            DEFAULT_CONTEXT_HEADER,
        ) or DEFAULT_CONTEXT_HEADER
        footer = getattr(
            app_settings,
            "CONTEXT_FORMATTING_FOOTER",
            DEFAULT_CONTEXT_FOOTER,
        ) or DEFAULT_CONTEXT_FOOTER
        item_tmpl = getattr(
            app_settings,
            "CONTEXT_FORMATTING_ITEM_TEMPLATE",
            DEFAULT_ITEM_TEMPLATE,
        ) or DEFAULT_ITEM_TEMPLATE
        inc_meta = getattr(app_settings, "CONTEXT_FORMATTING_INCLUDE_METADATA", True)
        inc_prov = getattr(app_settings, "CONTEXT_FORMATTING_INCLUDE_PROVENANCE", True)
        allow_empty = getattr(app_settings, "CONTEXT_FORMATTING_ALLOW_EMPTY_CONTEXT", True)
        empty_txt = getattr(
            app_settings,
            "CONTEXT_FORMATTING_EMPTY_TEXT",
            DEFAULT_EMPTY_CONTEXT_TEXT,
        ) or DEFAULT_EMPTY_CONTEXT_TEXT
        strict_project = getattr(app_settings, "CONTEXT_ASSEMBLY_STRICT_PROJECT_VALIDATION", False)

        return cls(
            strategy=strategy,
            context_header=header,
            context_footer=footer,
            item_label_template=item_tmpl,
            include_metadata=inc_meta,
            include_provenance=inc_prov,
            allow_empty_context=allow_empty,
            empty_context_text=empty_txt,
            strict_project_validation=strict_project,
        )

    @classmethod
    def from_env(cls) -> "ContextFormattingConfig":
        """Construct ContextFormattingConfig directly from environment settings."""
        return cls.from_settings()
