"""Domain models for the retrieval and query preprocessing layer."""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProcessedQuery:
    """Structured representation of a preprocessed retrieval query.

    Preserves the verbatim raw query alongside the conservatively normalized
    representation for downstream retrieval, tracing, and future transformation.
    """

    original_query: str
    processed_query: str

    @property
    def is_changed(self) -> bool:
        """Return True if preprocessing modified the query representation."""
        return self.original_query != self.processed_query

    def to_dict(self) -> dict[str, Any]:
        """Serialize the processed query to a standard dictionary."""
        return {
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "is_changed": self.is_changed,
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return (
            f"ProcessedQuery(original_query={self.original_query!r}, "
            f"processed_query={self.processed_query!r})"
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Decision produced by a Transformation Policy."""

    should_transform: bool
    reason: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy decision."""
        return {
            "should_transform": self.should_transform,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RetrievalQuerySet:
    """Stable output representation produced by Query Transformation for downstream retrieval.

    Guarantees that the original preprocessed query is always preserved as the primary
    authoritative representation. When transformation is performed, carries both the original
    and transformed queries additively. Downstream retrieval (dense/vector, sparse/keyword, fusion)
    operates directly on this query set without needing conditional branching.
    """

    original_query: str
    transformed_query: str | None = None
    is_transformed: bool = False
    strategy_used: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def queries(self) -> tuple[str, ...]:
        """Return all distinct retrieval query representations in priority order.

        Always guarantees original_query is at index 0. If transformation occurred
        and produced a non-empty transformed query distinct from original, returns
        (original_query, transformed_query). Otherwise returns (original_query,).
        """
        if (
            self.is_transformed
            and self.transformed_query
            and self.transformed_query != self.original_query
        ):
            return (self.original_query, self.transformed_query)
        return (self.original_query,)

    @property
    def has_transformed_query(self) -> bool:
        """Return True if a valid transformed query is present and distinct."""
        return bool(
            self.is_transformed
            and self.transformed_query
            and self.transformed_query != self.original_query
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retrieval query set to a standard dictionary."""
        return {
            "original_query": self.original_query,
            "transformed_query": self.transformed_query,
            "is_transformed": self.is_transformed,
            "strategy_used": self.strategy_used,
            "queries": list(self.queries),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return (
            f"RetrievalQuerySet(original_query={self.original_query!r}, "
            f"transformed_query={self.transformed_query!r}, "
            f"is_transformed={self.is_transformed}, "
            f"strategy_used={self.strategy_used!r})"
        )


@dataclass(frozen=True)
class EmbeddedQuery:
    """Vector embedding and identity for a single retrieval query representation."""

    query: str
    vector: list[float]
    query_type: str = "original"
    model: str = ""


@dataclass(frozen=True)
class EmbeddedQuerySet:
    """Minimal representation connecting a retrieval query set to embedding vectors."""

    original: EmbeddedQuery
    transformed: EmbeddedQuery | None = None

    @property
    def queries(self) -> tuple[EmbeddedQuery, ...]:
        """Return all embedded query representations in priority order.

        Always guarantees original is at index 0. If transformed is present,
        returns (original, transformed).
        """
        if self.transformed is not None:
            return (self.original, self.transformed)
        return (self.original,)


@dataclass(frozen=True)
class VectorSearchCandidate:
    """Represents a scored chunk candidate retrieved via vector similarity search.

    Preserves relational coordinates, similarity score from the vector index,
    and query representation provenance for downstream fusion and reranking.
    """

    chunk_id: str
    document_id: str
    project_id: str
    score: float
    query_type: str = "original"
    document_version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize search candidate to a standard dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "score": self.score,
            "query_type": self.query_type,
            "document_version_id": self.document_version_id,
        }

from storage.vector.models import SparseVector


@dataclass(frozen=True)
class KeywordSearchCandidate:
    """Represents a scored chunk candidate retrieved via keyword/sparse retrieval.

    Preserves relational coordinates, sparse retrieval score from the index,
    and query representation provenance for downstream fusion and reranking.
    """

    chunk_id: str
    document_id: str
    project_id: str
    score: float
    query_type: str = "original"
    document_version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize search candidate to a standard dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "score": self.score,
            "query_type": self.query_type,
            "document_version_id": self.document_version_id,
        }

