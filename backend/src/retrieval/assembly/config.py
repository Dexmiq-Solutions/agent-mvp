"""Configuration options for the Context Assembly stage."""

from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings
from exceptions.retrieval import ContextAssemblyValidationError


@dataclass(frozen=True)
class ContextAssemblyConfig:
    """Configuration settings for the Context Assembly stage.

    Attributes:
        max_context_items: Optional maximum number of context items to include in assembled context.
        use_contextual_enrichment: If True, prefers stored contextual_content over verbatim content.
        strict_project_validation: If True, raises ContextAssemblyValidationError on cross-tenant items.
    """

    max_context_items: Optional[int] = None
    use_contextual_enrichment: bool = True
    strict_project_validation: bool = False

    def __post_init__(self) -> None:
        """Validate context assembly configuration parameters upon instantiation."""
        if self.max_context_items is not None:
            if not isinstance(self.max_context_items, int) or self.max_context_items <= 0:
                raise ContextAssemblyValidationError(
                    f"max_context_items must be a positive integer or None, got {self.max_context_items}."
                )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "ContextAssemblyConfig":
        """Construct ContextAssemblyConfig from application Settings (Single Source of Truth)."""
        app_settings = settings or get_settings()
        return cls(
            max_context_items=getattr(app_settings, "CONTEXT_ASSEMBLY_MAX_ITEMS", None),
            use_contextual_enrichment=getattr(app_settings, "CONTEXT_ASSEMBLY_USE_CONTEXTUAL_ENRICHMENT", True),
            strict_project_validation=getattr(app_settings, "CONTEXT_ASSEMBLY_STRICT_PROJECT_VALIDATION", False),
        )

    @classmethod
    def from_env(cls) -> "ContextAssemblyConfig":
        """Construct ContextAssemblyConfig from environment settings."""
        return cls.from_settings()
