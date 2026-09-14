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


@dataclass(frozen=True)
class HydratedCandidate:
    """Represents a fully hydrated chunk candidate resolved from the PostgreSQL Content Store.

    Preserves exact retrieval coordinates, cross-encoder rerank score, ranking position,
    first-stage retrieval provenance (dense/sparse ranks and scores, fusion score),
    and attaches the authoritative stored chunk text, contextual content, and
    structured hierarchy/metadata required for downstream Context Assembly.
    """

    chunk_id: str
    document_id: str
    project_id: str
    content: str
    rank: int
    rerank_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    document_version_id: str | None = None
    contextual_content: str | None = None
    chunk_index: int = 0
    heading: str | None = None
    heading_level: int | None = None
    section_path: tuple[str, ...] = ()
    parent_element_id: str | None = None
    parent_chunk_id: str | None = None
    element_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Convenience property returning authoritative chunk text."""
        return self.content

    @property
    def chunk_text(self) -> str:
        """Convenience property returning authoritative chunk text."""
        return self.content

    @property
    def index(self) -> int:
        """Convenience property returning chunk_index."""
        return self.chunk_index

    @property
    def score(self) -> float:
        """Convenience property returning primary relevance score."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fusion_score is not None:
            return self.fusion_score
        return 0.0

    @classmethod
    def from_candidate_and_record(
        cls,
        candidate: Any,
        record: Any,
    ) -> "HydratedCandidate":
        """Construct a HydratedCandidate by combining a ranked candidate and a stored record."""
        content = (
            getattr(record, "content", None)
            or getattr(record, "chunk_text", None)
            or getattr(record, "text", "")
            or ""
        )
        contextual = getattr(record, "contextual_content", None)

        rerank_score = getattr(candidate, "rerank_score", None)
        fusion_score = getattr(candidate, "fusion_score", None)
        if fusion_score is None and hasattr(candidate, "score") and rerank_score is None:
            fusion_score = getattr(candidate, "score")
        elif rerank_score is None and hasattr(candidate, "score"):
            rerank_score = getattr(candidate, "score")

        rank = getattr(candidate, "rank", 1)

        merged_meta: dict[str, Any] = {}
        record_meta = getattr(record, "chunk_metadata", None) or getattr(record, "meta", None)
        if not isinstance(record_meta, dict):
            raw_meta = getattr(record, "metadata", None)
            if isinstance(raw_meta, dict):
                record_meta = raw_meta
        if isinstance(record_meta, dict):
            merged_meta.update(record_meta)
        cand_meta = getattr(candidate, "metadata", None)
        if isinstance(cand_meta, dict):
            merged_meta.update(cand_meta)

        section_path = getattr(record, "section_path", ()) or ()
        if isinstance(section_path, (list, set)):
            section_path = tuple(section_path)

        element_types = getattr(record, "element_types", ()) or ()
        if isinstance(element_types, (list, set)):
            element_types = tuple(str(et) for et in element_types)

        doc_version = getattr(candidate, "document_version_id", None) or getattr(
            record, "document_version_id", None
        )

        return cls(
            chunk_id=getattr(candidate, "chunk_id"),
            document_id=getattr(candidate, "document_id"),
            project_id=getattr(candidate, "project_id"),
            content=str(content),
            rank=rank,
            rerank_score=rerank_score,
            fusion_score=fusion_score,
            dense_rank=getattr(candidate, "dense_rank", None),
            sparse_rank=getattr(candidate, "sparse_rank", None),
            dense_score=getattr(candidate, "dense_score", None),
            sparse_score=getattr(candidate, "sparse_score", None),
            document_version_id=doc_version,
            contextual_content=str(contextual) if contextual else None,
            chunk_index=int(getattr(record, "chunk_index", getattr(record, "index", 0)) or 0),
            heading=getattr(record, "heading", None),
            heading_level=getattr(record, "heading_level", None),
            section_path=section_path,
            parent_element_id=getattr(record, "parent_element_id", None),
            parent_chunk_id=getattr(record, "parent_chunk_id", None),
            element_types=element_types,
            metadata=merged_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize hydrated candidate to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "content": self.content,
            "chunk_text": self.content,
            "rank": self.rank,
            "rerank_score": self.rerank_score,
            "fusion_score": self.fusion_score,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "document_version_id": self.document_version_id,
            "contextual_content": self.contextual_content,
            "chunk_index": self.chunk_index,
            "index": self.chunk_index,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "section_path": list(self.section_path),
            "parent_element_id": self.parent_element_id,
            "parent_chunk_id": self.parent_chunk_id,
            "element_types": list(self.element_types),
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


HydratedSearchCandidate = HydratedCandidate
HydratedChunk = HydratedCandidate


@dataclass(frozen=True)
class AssembledContextItem:
    """Represents an individual structured context item in the assembled retrieval context.

    Preserves exact retrieval order, identity coordinates, source provenance,
    relational and structural hierarchy, scoring provenance, and the authoritative
    stored content representation for downstream relevance check and generation.
    """

    chunk_id: str
    document_id: str
    project_id: str
    content: str
    rank: int
    text: str
    document_version_id: str | None = None
    contextual_content: str | None = None
    chunk_index: int = 0
    heading: str | None = None
    heading_level: int | None = None
    section_path: tuple[str, ...] = ()
    parent_element_id: str | None = None
    parent_chunk_id: str | None = None
    element_types: tuple[str, ...] = ()
    rerank_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Convenience property returning primary relevance score."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fusion_score is not None:
            return self.fusion_score
        return 0.0

    @property
    def chunk_text(self) -> str:
        """Convenience alias returning authoritative chunk text."""
        return self.content

    @property
    def index(self) -> int:
        """Convenience property returning chunk_index."""
        return self.chunk_index

    @property
    def source(self) -> str | None:
        """Convenience property extracting source identifier from metadata."""
        return (
            self.metadata.get("source")
            or self.metadata.get("file_name")
            or self.metadata.get("title")
        )

    @classmethod
    def from_hydrated_candidate(
        cls,
        candidate: Any,
        use_contextual_enrichment: bool = True,
        rank_override: int | None = None,
    ) -> "AssembledContextItem":
        """Construct an AssembledContextItem from a HydratedCandidate or duck-typed record.

        Args:
            candidate: HydratedCandidate or compatible candidate object.
            use_contextual_enrichment: If True and contextual_content is present,
                uses contextual_content as the downstream `text`. Otherwise uses `content`.
            rank_override: Optional explicit rank override preserving pipeline ordering.

        Returns:
            AssembledContextItem: Structured context item.
        """
        raw_content = (
            getattr(candidate, "content", None)
            or getattr(candidate, "chunk_text", None)
            or getattr(candidate, "text", "")
            or ""
        )
        contextual = getattr(candidate, "contextual_content", None)

        if use_contextual_enrichment and contextual and str(contextual).strip():
            effective_text = str(contextual)
        else:
            effective_text = str(raw_content)

        rank = rank_override if rank_override is not None else getattr(candidate, "rank", 1)

        section_path = getattr(candidate, "section_path", ()) or ()
        if isinstance(section_path, (list, set)):
            section_path = tuple(section_path)

        element_types = getattr(candidate, "element_types", ()) or ()
        if isinstance(element_types, (list, set)):
            element_types = tuple(str(et) for et in element_types)

        cand_metadata = dict(getattr(candidate, "metadata", None) or {})

        return cls(
            chunk_id=str(getattr(candidate, "chunk_id")),
            document_id=str(getattr(candidate, "document_id")),
            project_id=str(getattr(candidate, "project_id")),
            content=str(raw_content),
            rank=int(rank),
            text=effective_text,
            document_version_id=getattr(candidate, "document_version_id", None),
            contextual_content=str(contextual) if contextual else None,
            chunk_index=int(getattr(candidate, "chunk_index", getattr(candidate, "index", 0)) or 0),
            heading=getattr(candidate, "heading", None),
            heading_level=getattr(candidate, "heading_level", None),
            section_path=section_path,
            parent_element_id=getattr(candidate, "parent_element_id", None),
            parent_chunk_id=getattr(candidate, "parent_chunk_id", None),
            element_types=element_types,
            rerank_score=getattr(candidate, "rerank_score", None),
            fusion_score=getattr(candidate, "fusion_score", None),
            dense_rank=getattr(candidate, "dense_rank", None),
            sparse_rank=getattr(candidate, "sparse_rank", None),
            dense_score=getattr(candidate, "dense_score", None),
            sparse_score=getattr(candidate, "sparse_score", None),
            metadata=cand_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize assembled context item to a standard dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "content": self.content,
            "chunk_text": self.content,
            "rank": self.rank,
            "text": self.text,
            "document_version_id": self.document_version_id,
            "contextual_content": self.contextual_content,
            "chunk_index": self.chunk_index,
            "index": self.chunk_index,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "section_path": list(self.section_path),
            "parent_element_id": self.parent_element_id,
            "parent_chunk_id": self.parent_chunk_id,
            "element_types": list(self.element_types),
            "rerank_score": self.rerank_score,
            "fusion_score": self.fusion_score,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
        }
        if self.source:
            data["source"] = self.source
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


ContextItem = AssembledContextItem


@dataclass(frozen=True)
class AssembledContext:
    """Structured retrieval context containing ordered context items and execution metadata.

    Provides the clean, structured retrieval context contract between Context Assembly
    and downstream Relevance Check / Fallback.
    """

    items: tuple[AssembledContextItem, ...]
    project_id: str
    query: str | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return True if the assembled context contains zero items."""
        return len(self.items) == 0

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return all chunk identifiers in preserved retrieval order."""
        return tuple(item.chunk_id for item in self.items)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Return distinct document identifiers in order of first encounter."""
        return tuple(dict.fromkeys(item.document_id for item in self.items))

    def __len__(self) -> int:
        """Return the number of assembled context items."""
        return len(self.items)

    def __iter__(self):
        """Iterate over assembled context items in preserved order."""
        return iter(self.items)

    def __getitem__(self, idx: int) -> AssembledContextItem:
        """Access context item by 0-based index."""
        return self.items[idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize assembled context to a dictionary."""
        return {
            "project_id": self.project_id,
            "query": self.query,
            "item_count": len(self.items),
            "total_tokens": self.total_tokens,
            "items": [item.to_dict() for item in self.items],
            "chunk_ids": list(self.chunk_ids),
            "document_ids": list(self.document_ids),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return (
            f"AssembledContext(project_id={self.project_id!r}, "
            f"items_count={len(self.items)}, "
            f"query={self.query!r})"
        )


RetrievalContext = AssembledContext
StructuredRetrievalContext = AssembledContext


@dataclass(frozen=True)
class RetrievedChunk:
    """Application-facing representation of a retrieved, hydrated, and reranked chunk.

    Provides a clean, provider-agnostic domain model exposing authoritative stored
    content, relevance score, ranking, provenance, and hierarchy coordinates.
    """

    chunk_id: str
    document_id: str
    project_id: str
    content: str
    rank: int
    score: float
    document_version_id: str | None = None
    heading: str | None = None
    section_path: tuple[str, ...] = ()
    contextual_content: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hydrated_candidate(cls, candidate: Any) -> "RetrievedChunk":
        """Construct a RetrievedChunk from a HydratedCandidate or compatible candidate."""
        sec_path = getattr(candidate, "section_path", ()) or ()
        if isinstance(sec_path, (list, set)):
            sec_path = tuple(sec_path)

        cand_meta = dict(getattr(candidate, "metadata", None) or {})

        return cls(
            chunk_id=str(getattr(candidate, "chunk_id")),
            document_id=str(getattr(candidate, "document_id")),
            project_id=str(getattr(candidate, "project_id")),
            content=str(getattr(candidate, "content", getattr(candidate, "text", ""))),
            rank=int(getattr(candidate, "rank", 1)),
            score=float(getattr(candidate, "score", getattr(candidate, "rerank_score", getattr(candidate, "fusion_score", 0.0))) or 0.0),
            document_version_id=getattr(candidate, "document_version_id", None),
            heading=getattr(candidate, "heading", None),
            section_path=sec_path,
            contextual_content=getattr(candidate, "contextual_content", None),
            chunk_index=int(getattr(candidate, "chunk_index", getattr(candidate, "index", 0)) or 0),
            metadata=cand_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrieved chunk to a dictionary."""
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "content": self.content,
            "rank": self.rank,
            "score": self.score,
            "document_version_id": self.document_version_id,
            "heading": self.heading,
            "section_path": list(self.section_path),
            "contextual_content": self.contextual_content,
            "chunk_index": self.chunk_index,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class RetrievalAttemptMetadata:
    """Detailed observability and metrics for an individual retrieval attempt."""

    attempt: int
    query: str
    is_transformed: bool = False
    transformed_query: str | None = None
    strategy_used: str | None = None
    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    fused_candidates_count: int = 0
    filtered_candidates_count: int = 0
    reranked_candidates_count: int = 0
    hydrated_candidates_count: int = 0
    is_relevant: bool | None = None
    relevance_reason: str | None = None
    latency_ms: float = 0.0
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize attempt metadata to a dictionary."""
        return {
            "attempt": self.attempt,
            "query": self.query,
            "is_transformed": self.is_transformed,
            "transformed_query": self.transformed_query,
            "strategy_used": self.strategy_used,
            "dense_candidates_count": self.dense_candidates_count,
            "sparse_candidates_count": self.sparse_candidates_count,
            "fused_candidates_count": self.fused_candidates_count,
            "filtered_candidates_count": self.filtered_candidates_count,
            "reranked_candidates_count": self.reranked_candidates_count,
            "hydrated_candidates_count": self.hydrated_candidates_count,
            "is_relevant": self.is_relevant,
            "relevance_reason": self.relevance_reason,
            "latency_ms": self.latency_ms,
            "stage_latencies_ms": dict(self.stage_latencies_ms),
        }


@dataclass(frozen=True)
class RetrievalExecutionMetadata:
    """Execution telemetry and observability summary for the complete retrieval operation."""

    total_duration_ms: float
    attempts_count: int = 1
    fallback_triggered: bool = False
    attempts: tuple[RetrievalAttemptMetadata, ...] = ()
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution metadata to a dictionary."""
        data: dict[str, Any] = {
            "total_duration_ms": self.total_duration_ms,
            "attempts_count": self.attempts_count,
            "fallback_triggered": self.fallback_triggered,
            "attempts": [a.to_dict() for a in self.attempts],
            "stage_latencies_ms": dict(self.stage_latencies_ms),
        }
        if self.additional_metadata:
            data["additional_metadata"] = dict(self.additional_metadata)
        return data


@dataclass(frozen=True)
class RetrievalResult:
    """Comprehensive, application-facing result produced by RAG retrieval orchestration.

    Contains the authoritative retrieved chunks, the assembled and formatted context ready
    for downstream generation, execution telemetry, and query representations.
    """

    project_id: str
    original_query: str
    retrieval_query: str
    chunks: tuple[RetrievedChunk, ...]
    assembled_context: AssembledContext
    formatted_context: Any
    execution_metadata: RetrievalExecutionMetadata

    @property
    def is_empty(self) -> bool:
        """Return True if zero chunks were retrieved."""
        return len(self.chunks) == 0

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return distinct chunk IDs in retrieved order."""
        return tuple(chunk.chunk_id for chunk in self.chunks)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Return distinct document IDs in order of first appearance."""
        return tuple(dict.fromkeys(chunk.document_id for chunk in self.chunks))

    @property
    def formatted_text(self) -> str:
        """Return string representation of formatted context."""
        if hasattr(self.formatted_context, "text"):
            return str(self.formatted_context.text)
        return str(self.formatted_context or "")

    @property
    def total_tokens(self) -> int | None:
        """Return total tokens calculated during context formatting or assembly if available."""
        if hasattr(self.formatted_context, "total_tokens"):
            return getattr(self.formatted_context, "total_tokens")
        if self.assembled_context is not None:
            return self.assembled_context.total_tokens
        return None

    def __len__(self) -> int:
        """Return number of retrieved chunks."""
        return len(self.chunks)

    def __iter__(self):
        """Iterate over retrieved chunks."""
        return iter(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrieval result to a standard dictionary."""
        formatted_dict = (
            self.formatted_context.to_dict()
            if hasattr(self.formatted_context, "to_dict")
            else {"text": str(self.formatted_context or "")}
        )
        return {
            "project_id": self.project_id,
            "original_query": self.original_query,
            "retrieval_query": self.retrieval_query,
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
            "assembled_context": self.assembled_context.to_dict(),
            "formatted_context": formatted_dict,
            "execution_metadata": self.execution_metadata.to_dict(),
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return (
            f"RetrievalResult(project_id={self.project_id!r}, "
            f"chunks_count={len(self.chunks)}, "
            f"query={self.original_query!r})"
        )



