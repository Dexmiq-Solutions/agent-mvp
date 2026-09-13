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
    query representation provenance, and retrieval metadata for downstream fusion and filtering.
    """

    chunk_id: str
    document_id: str
    project_id: str
    score: float
    query_type: str = "original"
    document_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize search candidate to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "score": self.score,
            "query_type": self.query_type,
            "document_version_id": self.document_version_id,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

from storage.vector.models import SparseVector


@dataclass(frozen=True)
class KeywordSearchCandidate:
    """Represents a scored chunk candidate retrieved via keyword/sparse retrieval.

    Preserves relational coordinates, sparse retrieval score from the index,
    query representation provenance, and retrieval metadata for downstream fusion and filtering.
    """

    chunk_id: str
    document_id: str
    project_id: str
    score: float
    query_type: str = "original"
    document_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize search candidate to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "score": self.score,
            "query_type": self.query_type,
            "document_version_id": self.document_version_id,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class FusedCandidate:
    """Represents a scored chunk candidate produced by hybrid retrieval fusion (RRF).

    Preserves relational coordinates, unified fused ranking score, 1-based rank,
    individual retrieval branch provenance, and combined retrieval metadata.
    """

    chunk_id: str
    document_id: str
    project_id: str
    score: float
    rank: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    document_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fused_score(self) -> float:
        """Convenience alias for the unified fused score."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Serialize fused candidate to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "score": self.score,
            "fused_score": self.score,
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "document_version_id": self.document_version_id,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


FusedSearchCandidate = FusedCandidate


@dataclass(frozen=True)
class RerankedCandidate:
    """Represents a scored chunk candidate produced by cross-encoder reranking.

    Preserves relational coordinates, individual retrieval branch provenance,
    prior fused ranking score, updated 1-based rank, and attached cross-encoder
    relevance score.
    """

    chunk_id: str
    document_id: str
    project_id: str
    rerank_score: float
    rank: int
    fusion_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    document_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Convenience property returning the authoritative cross-encoder relevance score."""
        return self.rerank_score

    @classmethod
    def from_candidate(
        cls,
        candidate: Any,
        rerank_score: float,
        rank: int,
    ) -> "RerankedCandidate":
        """Construct a RerankedCandidate from an existing candidate model (e.g. FusedCandidate)."""
        fusion_score_val = getattr(candidate, "fusion_score", None)
        if fusion_score_val is None and hasattr(candidate, "score"):
            fusion_score_val = getattr(candidate, "score")

        cand_metadata = dict(getattr(candidate, "metadata", None) or {})

        return cls(
            chunk_id=getattr(candidate, "chunk_id"),
            document_id=getattr(candidate, "document_id"),
            project_id=getattr(candidate, "project_id"),
            rerank_score=rerank_score,
            rank=rank,
            fusion_score=fusion_score_val,
            dense_rank=getattr(candidate, "dense_rank", None),
            sparse_rank=getattr(candidate, "sparse_rank", None),
            dense_score=getattr(candidate, "dense_score", None),
            sparse_score=getattr(candidate, "sparse_score", None),
            document_version_id=getattr(candidate, "document_version_id", None),
            metadata=cand_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize reranked candidate to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "rerank_score": self.rerank_score,
            "score": self.rerank_score,
            "rank": self.rank,
            "fusion_score": self.fusion_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "document_version_id": self.document_version_id,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


RerankedSearchCandidate = RerankedCandidate


