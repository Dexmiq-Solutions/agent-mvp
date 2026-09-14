"""Configuration settings for post-processing generation results."""

from dataclasses import dataclass
from typing import Optional

from core.config import Settings, get_settings


@dataclass(frozen=True)
class PostProcessingConfig:
    """Configuration governing deterministic post-processing transformations.

    Follows project guidelines:
    - Purely deterministic local processing parameters.
    - Sensible defaults for whitespace, line endings, finish reasons, and empty checks.
    - Single Source of Truth via integration with application Settings.
    """

    strip_whitespace: bool = True
    normalize_line_endings: bool = True
    normalize_finish_reason: bool = True
    allow_empty_content: bool = False
    parse_json: bool = False
    require_json: bool = False
    strip_code_fences_for_json: bool = True
    max_content_length: int | None = None

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PostProcessingConfig":
        """Construct PostProcessingConfig from application Settings (Single Source of Truth)."""
        app_settings = settings or get_settings()

        strip_ws = getattr(app_settings, "POST_PROCESSING_STRIP_WHITESPACE", True)
        norm_lines = getattr(app_settings, "POST_PROCESSING_NORMALIZE_LINE_ENDINGS", True)
        norm_finish = getattr(app_settings, "POST_PROCESSING_NORMALIZE_FINISH_REASON", True)
        allow_empty = getattr(app_settings, "POST_PROCESSING_ALLOW_EMPTY_CONTENT", False)
        parse_json = getattr(app_settings, "POST_PROCESSING_PARSE_JSON", False)

        return cls(
            strip_whitespace=bool(strip_ws),
            normalize_line_endings=bool(norm_lines),
            normalize_finish_reason=bool(norm_finish),
            allow_empty_content=bool(allow_empty),
            parse_json=bool(parse_json),
            require_json=False,
            strip_code_fences_for_json=True,
            max_content_length=None,
        )

    @classmethod
    def from_env(cls) -> "PostProcessingConfig":
        """Convenience factory constructing PostProcessingConfig from environment settings."""
        return cls.from_settings()

    def __repr__(self) -> str:
        """Safe representation of post-processing configuration parameters."""
        return (
            f"PostProcessingConfig("
            f"strip_whitespace={self.strip_whitespace}, "
            f"normalize_line_endings={self.normalize_line_endings}, "
            f"normalize_finish_reason={self.normalize_finish_reason}, "
            f"allow_empty_content={self.allow_empty_content}, "
            f"parse_json={self.parse_json})"
        )
