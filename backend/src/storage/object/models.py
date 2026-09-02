"""Data models for object storage operations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StorageObjectMetadata:
    """Metadata representing a stored object in object storage."""

    name: str
    path: str
    bucket: str
    size_bytes: int | None = None
    content_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    etag: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
