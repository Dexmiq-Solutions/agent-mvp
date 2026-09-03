"""Domain models for the document ingestion layer."""

from dataclasses import dataclass, field
from typing import Any

from acquisition.formats import DocumentType


@dataclass(frozen=True)
class DocumentSourceReference:
    """Explicit reference identifying an acquired source document to be ingested.
    
    Provides location, tenant, and optional known metadata to initiate ingestion.
    """

    project_id: str
    storage_path: str
    storage_bucket: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    document_type: DocumentType | None = None
    document_version_id: str | None = None
    document_id: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize source reference to dictionary."""
        return {
            "project_id": self.project_id,
            "storage_path": self.storage_path,
            "storage_bucket": self.storage_bucket,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "document_type": self.document_type.value if self.document_type else None,
            "document_version_id": self.document_version_id,
            "document_id": self.document_id,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class IngestedDocument:
    """Standardized internal representation of a successfully ingested document.
    
    Represents the document fetched into the application boundary with its raw
    content and preserved source metadata, ready for subsequent parsing and extraction.
    
    IMPORTANT:
    This model explicitly does NOT contain parsed text, extracted content,
    chunks, embeddings, or vector database identifiers.
    """

    document_id: str
    project_id: str
    source_storage_path: str
    original_filename: str
    detected_document_type: DocumentType
    content_type: str
    raw_bytes: bytes
    source_metadata: dict[str, Any] = field(default_factory=dict)
    document_version_id: str | None = None
    size_bytes: int = 0

    def __post_init__(self) -> None:
        """Ensure size_bytes matches raw_bytes length if not explicitly set."""
        if not self.size_bytes and self.raw_bytes is not None:
            object.__setattr__(self, "size_bytes", len(self.raw_bytes))

    def __repr__(self) -> str:
        """Return safe string representation omitting raw byte contents.
        
        Prevents leaking file contents into application logs or stack traces.
        """
        return (
            f"IngestedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"original_filename={self.original_filename!r}, "
            f"detected_document_type={self.detected_document_type.value!r}, "
            f"content_type={self.content_type!r}, "
            f"size_bytes={self.size_bytes}, "
            f"document_version_id={self.document_version_id!r})"
        )

    def to_dict(self, include_bytes: bool = False) -> dict[str, Any]:
        """Serialize ingested document metadata to a dictionary.
        
        Args:
            include_bytes: Whether to include raw bytes payload. Defaults to False.
        """
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "document_version_id": self.document_version_id,
            "source_storage_path": self.source_storage_path,
            "original_filename": self.original_filename,
            "detected_document_type": self.detected_document_type.value,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "source_metadata": self.source_metadata,
        }
        if include_bytes:
            data["raw_bytes"] = self.raw_bytes
        return data
