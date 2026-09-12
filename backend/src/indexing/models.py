"""Domain models, configuration, and audit reports for vector indexing and storage."""

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import Settings, get_settings
from chunking.models import DocumentChunk
from exceptions.indexing import IndexingConfigurationError, InvalidIndexingInputError
from indexing.identity import generate_point_id
from retrieval.models import SparseVector
from storage.vector.models import VectorPayload, VectorRecord


@dataclass(frozen=True)
class IndexingConfig:
    """Configuration options for the vector indexing and storage stage."""

    batch_size: int = 64
    max_retries: int = 3
    retry_delay: float = 0.5
    retry_backoff: float = 2.0
    collection_name: Optional[str] = None
    expected_vector_size: Optional[int] = None
    distance: str = "Cosine"
    ensure_collection: bool = True
    sparse_indexing_enabled: bool = True
    sparse_encoder_strategy: str = "technical_hash"
    sparse_encoder_version: str = "1.0"
    sparse_vector_name: str = "sparse"

    def __post_init__(self) -> None:
        """Validate indexing configuration parameters upon instantiation."""
        if self.batch_size <= 0:
            raise IndexingConfigurationError(
                f"batch_size must be a positive integer, got {self.batch_size}."
            )
        if self.max_retries < 0:
            raise IndexingConfigurationError(
                f"max_retries cannot be negative, got {self.max_retries}."
            )
        if self.retry_delay < 0:
            raise IndexingConfigurationError(
                f"retry_delay cannot be negative, got {self.retry_delay}."
            )
        if self.retry_backoff < 1.0:
            raise IndexingConfigurationError(
                f"retry_backoff must be >= 1.0, got {self.retry_backoff}."
            )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "IndexingConfig":
        """Construct IndexingConfig from application Settings."""
        app_settings = settings or get_settings()
        return cls(
            batch_size=getattr(app_settings, "QDRANT_BATCH_SIZE", 64),
            max_retries=getattr(app_settings, "QDRANT_MAX_RETRIES", 3),
            retry_delay=getattr(app_settings, "QDRANT_RETRY_DELAY", 0.5),
            retry_backoff=getattr(app_settings, "QDRANT_RETRY_BACKOFF", 2.0),
            collection_name=app_settings.QDRANT_COLLECTION_NAME,
            expected_vector_size=app_settings.QDRANT_VECTOR_SIZE,
            distance=getattr(app_settings, "QDRANT_DISTANCE", "Cosine"),
            ensure_collection=True,
            sparse_indexing_enabled=getattr(app_settings, "SPARSE_INDEXING_ENABLED", True),
            sparse_encoder_strategy=getattr(app_settings, "SPARSE_ENCODER_STRATEGY", "technical_hash"),
            sparse_encoder_version=getattr(app_settings, "SPARSE_ENCODER_VERSION", "1.0"),
            sparse_vector_name=getattr(app_settings, "SPARSE_VECTOR_NAME", "sparse"),
        )

    @classmethod
    def from_env(cls) -> "IndexingConfig":
        """Construct IndexingConfig by reading environment variables via Settings."""
        return cls.from_settings()


@dataclass(frozen=True)
class IndexableRecord:
    """Represents a validated, indexable record ready to be converted into a Qdrant point.

    Binds the generated embedding vector, optional sparse vector, stable entity coordinates,
    and retrieval metadata while preserving representation configuration provenance.
    """

    chunk_id: str
    document_id: str
    project_id: str
    vector: list[float]
    document_version_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    embedding_model: Optional[str] = None
    embedding_provider: str = "voyage"
    embedding_dimension: Optional[int] = None
    point_id_override: Optional[str] = None
    sparse_vector: Optional[SparseVector] = None
    sparse_encoder_strategy: Optional[str] = None
    sparse_encoder_version: Optional[str] = None

    @property
    def point_id(self) -> str:
        """Deterministic, stable UUID point ID."""
        if self.point_id_override:
            return self.point_id_override
        return generate_point_id(
            project_id=self.project_id,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            document_version_id=self.document_version_id,
        )

    def to_vector_record(self) -> VectorRecord:
        """Construct a VectorRecord instance compatible with the vector storage abstraction."""
        # Build metadata payload preserving retrieval attributes and representation configuration
        merged_metadata: dict[str, Any] = dict(self.payload)

        # Ensure embedding configuration identity is attached
        if self.embedding_model is not None:
            merged_metadata["embedding_model"] = self.embedding_model
        if self.embedding_provider is not None:
            merged_metadata["embedding_provider"] = self.embedding_provider
        dim = self.embedding_dimension or len(self.vector)
        merged_metadata["embedding_dimension"] = dim

        # Ensure sparse encoder configuration identity is attached
        if self.sparse_encoder_strategy is not None:
            merged_metadata["sparse_encoder_strategy"] = self.sparse_encoder_strategy
        if self.sparse_encoder_version is not None:
            merged_metadata["sparse_encoder_version"] = self.sparse_encoder_version

        vector_payload = VectorPayload(
            project_id=self.project_id,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            document_version_id=self.document_version_id,
            metadata=merged_metadata,
        )

        return VectorRecord(
            id=self.point_id,
            vector=list(self.vector),
            payload=vector_payload,
            sparse_vector=self.sparse_vector,
        )

    @classmethod
    def from_chunk_and_vector(
        cls,
        chunk: DocumentChunk,
        vector: list[float],
        embedding_model: Optional[str] = None,
        embedding_provider: str = "voyage",
        embedding_dimension: Optional[int] = None,
        sparse_vector: Optional[SparseVector] = None,
        sparse_encoder_strategy: Optional[str] = None,
        sparse_encoder_version: Optional[str] = None,
    ) -> "IndexableRecord":
        """Factory method to construct an IndexableRecord from a DocumentChunk and embedding vector."""
        if not isinstance(chunk, DocumentChunk):
            raise InvalidIndexingInputError(
                f"chunk must be an instance of DocumentChunk, got {type(chunk).__name__}."
            )

        # Extract retrieval payload from chunk if available
        if hasattr(chunk, "to_vector_payload") and callable(chunk.to_vector_payload):
            chunk_payload = chunk.to_vector_payload()
        else:
            chunk_payload = {
                "project_id": chunk.project_id,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "document_version_id": chunk.document_version_id,
                "index": chunk.index,
                "section_path": list(chunk.section_path),
                "heading": chunk.heading,
            }

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            project_id=chunk.project_id,
            document_version_id=chunk.document_version_id,
            vector=vector,
            payload=chunk_payload,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension or len(vector),
            sparse_vector=sparse_vector,
            sparse_encoder_strategy=sparse_encoder_strategy,
            sparse_encoder_version=sparse_encoder_version,
        )


@dataclass(frozen=True)
class IndexingBatchResult:
    """Details and metrics for an individual batched upsert execution."""

    batch_index: int
    points_count: int
    point_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    status: str  # 'success' | 'failed'
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch result to a dictionary."""
        return {
            "batch_index": self.batch_index,
            "points_count": self.points_count,
            "point_ids": list(self.point_ids),
            "chunk_ids": list(self.chunk_ids),
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class IndexingReport:
    """Summary report detailing the results, counts, and metrics of an indexing operation.

    Provides complete explainability, auditability, and partial failure detection without
    exposing raw vectors or sensitive text payloads.
    """

    total_attempted: int = 0
    total_indexed: int = 0
    total_failed: int = 0
    batches_count: int = 0
    failed_batches_count: int = 0
    point_ids: tuple[str, ...] = ()
    failed_chunk_ids: tuple[str, ...] = ()
    collection_name: str = ""
    model: Optional[str] = None
    elapsed_ms: float = 0.0
    batches: tuple[IndexingBatchResult, ...] = ()

    @property
    def is_success(self) -> bool:
        """Return True if all attempted points were successfully indexed."""
        return self.total_failed == 0 and self.total_attempted > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize indexing report to a dictionary."""
        return {
            "total_attempted": self.total_attempted,
            "total_indexed": self.total_indexed,
            "total_failed": self.total_failed,
            "batches_count": self.batches_count,
            "failed_batches_count": self.failed_batches_count,
            "point_ids": list(self.point_ids),
            "failed_chunk_ids": list(self.failed_chunk_ids),
            "collection_name": self.collection_name,
            "model": self.model,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "is_success": self.is_success,
            "batches": [b.to_dict() for b in self.batches],
        }

    def __repr__(self) -> str:
        """Safe representation omitting raw payloads."""
        return (
            f"IndexingReport(total_attempted={self.total_attempted}, "
            f"total_indexed={self.total_indexed}, "
            f"total_failed={self.total_failed}, "
            f"batches={self.batches_count}, "
            f"failed_batches={self.failed_batches_count}, "
            f"collection={self.collection_name!r}, "
            f"model={self.model!r}, "
            f"elapsed_ms={self.elapsed_ms:.2f}ms, "
            f"is_success={self.is_success})"
        )
