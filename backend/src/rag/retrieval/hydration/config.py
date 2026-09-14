"""Configuration options for the Chunk Fetching / Hydration stage."""

from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings
from exceptions.retrieval import ChunkHydrationValidationError


@dataclass(frozen=True)
class ChunkHydrationConfig:
    """Configuration options for chunk hydration from PostgreSQL Content Store."""

    fail_on_missing: bool = False
    verify_version_identity: bool = True
    max_batch_size: int = 500

    def __post_init__(self) -> None:
        """Validate hydration configuration parameters upon instantiation."""
        if self.max_batch_size <= 0:
            raise ChunkHydrationValidationError(
                f"max_batch_size must be a positive integer, got {self.max_batch_size}."
            )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "ChunkHydrationConfig":
        """Construct ChunkHydrationConfig from application Settings."""
        app_settings = settings or get_settings()
        return cls(
            fail_on_missing=getattr(app_settings, "CHUNK_HYDRATION_FAIL_ON_MISSING", False),
            verify_version_identity=getattr(app_settings, "CHUNK_HYDRATION_VERIFY_VERSION", True),
            max_batch_size=getattr(app_settings, "CHUNK_HYDRATION_MAX_BATCH_SIZE", 500),
        )

    @classmethod
    def from_env(cls) -> "ChunkHydrationConfig":
        """Construct ChunkHydrationConfig by reading environment variables via application Settings."""
        return cls.from_settings()
