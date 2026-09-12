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


import math


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation with sorted dimension indices and positive weights.

    Attributes:
        indices: Sorted sequence of non-negative integer dimension indices.
        values: Corresponding sequence of non-negative finite float weights.
    """

    indices: tuple[int, ...] = field(default_factory=tuple)
    values: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate sparse vector structure upon creation."""
        # Normalize list/iterable to tuple if passed
        if isinstance(self.indices, (list, set)):
            object.__setattr__(self, "indices", tuple(self.indices))
        if isinstance(self.values, (list, set)):
            object.__setattr__(self, "values", tuple(float(v) for v in self.values))

        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector indices and values length mismatch: "
                f"indices={len(self.indices)}, values={len(self.values)}."
            )
        for i, idx in enumerate(self.indices):
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"SparseVector index at position {i} must be a non-negative integer, got {idx}."
                )
            if i > 0 and idx <= self.indices[i - 1]:
                raise ValueError(
                    f"SparseVector indices must be strictly increasing and unique: "
                    f"index[{i-1}]={self.indices[i-1]}, index[{i}]={idx}."
                )
        for i, val in enumerate(self.values):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"SparseVector value at position {i} must be a finite float, got {val}."
                )

    @property
    def is_empty(self) -> bool:
        """Return True if sparse vector contains no non-zero dimensions."""
        return len(self.indices) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize sparse vector to a standard dictionary."""
        return {
            "indices": list(self.indices),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class VectorRecord:
    """Represents a single vector point to be stored or updated in vector storage."""

    id: str | int
    vector: list[float]
    payload: VectorPayload
    sparse_vector: Optional[SparseVector] = None


@dataclass(frozen=True)
class VectorSearchResult:
    """Represents a scored vector search match returned from similarity search."""

    id: str | int
    score: float
    payload: VectorPayload
    vector: Optional[list[float]] = None
