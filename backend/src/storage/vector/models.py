"""Data models representing vector storage payloads, records, and search results."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class VectorPayload:
    """Retrieval-oriented metadata payload associated with an embedding vector.
    
    Contains stable relational references connecting Qdrant vector points back to
    PostgreSQL entities (project, document, version, chunk).
    """

    project_id: str
    document_id: str
    chunk_id: str
    document_version_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload into a dictionary for Qdrant storage."""
        data: dict[str, Any] = {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }
        if self.document_version_id is not None:
            data["document_version_id"] = self.document_version_id

        # Merge additional retrieval metadata without overwriting primary keys
        for k, v in self.metadata.items():
            if k not in data:
                data[k] = v
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorPayload":
        """Deserialize a dictionary payload retrieved from Qdrant."""
        payload_data = dict(data)
        project_id = str(payload_data.pop("project_id", ""))
        document_id = str(payload_data.pop("document_id", ""))
        chunk_id = str(payload_data.pop("chunk_id", ""))
        document_version_id = payload_data.pop("document_version_id", None)
        if document_version_id is not None:
            document_version_id = str(document_version_id)

        return cls(
            project_id=project_id,
            document_id=document_id,
            chunk_id=chunk_id,
            document_version_id=document_version_id,
            metadata=payload_data,
        )


@dataclass(frozen=True)
class VectorRecord:
    """Represents a single vector point to be stored or updated in vector storage."""

    id: str | int
    vector: list[float]
    payload: VectorPayload


@dataclass(frozen=True)
class VectorSearchResult:
    """Represents a scored vector search match returned from similarity search."""

    id: str | int
    score: float
    payload: VectorPayload
    vector: Optional[list[float]] = None
