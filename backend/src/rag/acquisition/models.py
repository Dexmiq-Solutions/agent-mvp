"""Domain models for the source document acquisition layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from rag.acquisition.formats import DocumentType


class AcquisitionStatus(str, Enum):
    """Execution status for an acquisition operation."""

    SUCCESS = "success"
    EMPTY = "empty"
    PARTIAL = "partial"


@dataclass(frozen=True)
class SourceReference:
    """Stable storage reference to a physical document in object storage.
    
    Provides the downstream ingestion pipeline with precise, provider-agnostic
    location details to download and process the file.
    """

    bucket: str
    path: str
    source_type: str = "supabase_storage"
    etag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize source reference to dictionary."""
        return {
            "bucket": self.bucket,
            "path": self.path,
            "source_type": self.source_type,
            "etag": self.etag,
        }


@dataclass(frozen=True)
class SourceDocument:
    """Acquired source document reference available for the ingestion stage.
    
    Represents metadata and storage coordinates for a verified, supported document
    without containing extracted text or parsed content.
    """

    source_id: str
    project_id: str
    filename: str
    storage_path: str
    storage_bucket: str
    document_type: DocumentType
    content_type: str
    extension: str
    size_bytes: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ref: SourceReference | None = None

    def to_source_reference(self) -> SourceReference:
        """Return or construct a SourceReference for this document."""
        if self.source_ref is not None:
            return self.source_ref
        etag = self.metadata.get("etag") or self.metadata.get("eTag")
        return SourceReference(
            bucket=self.storage_bucket,
            path=self.storage_path,
            source_type="supabase_storage",
            etag=etag,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize document metadata to a dictionary."""
        return {
            "source_id": self.source_id,
            "project_id": self.project_id,
            "filename": self.filename,
            "storage_path": self.storage_path,
            "storage_bucket": self.storage_bucket,
            "document_type": self.document_type.value,
            "content_type": self.content_type,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "source_ref": self.to_source_reference().to_dict(),
        }


@dataclass(frozen=True)
class AcquisitionResult:
    """Structured result of an acquisition run for a specific project.
    
    Contains all successfully discovered and validated source documents,
    along with accounting of any skipped or unsupported objects.
    """

    project_id: str
    documents: list[SourceDocument] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    total_discovered: int = 0
    status: AcquisitionStatus = AcquisitionStatus.SUCCESS

    def __iter__(self) -> Iterator[SourceDocument]:
        """Allow direct iteration over acquired documents."""
        return iter(self.documents)

    def __len__(self) -> int:
        """Return the number of acquired documents."""
        return len(self.documents)

    def __getitem__(self, index: int) -> SourceDocument:
        """Allow indexed access to acquired documents."""
        return self.documents[index]

    def to_dict(self) -> dict[str, Any]:
        """Serialize acquisition result to dictionary."""
        return {
            "project_id": self.project_id,
            "status": self.status.value,
            "document_count": len(self.documents),
            "skipped_count": len(self.skipped_paths),
            "total_discovered": self.total_discovered,
            "documents": [doc.to_dict() for doc in self.documents],
            "skipped_paths": list(self.skipped_paths),
        }
