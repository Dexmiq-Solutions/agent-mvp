"""Document indexing service orchestrating batched, idempotent vector storage in Qdrant."""

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any, Optional

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.chunking.models import ChunkedDocument, DocumentChunk
from rag.embeddings.models import EmbeddingBatchResult, EmbeddingResult
from exceptions.indexing import (
    IndexingConfigurationError,
    IndexingConnectionError,
    IndexingError,
    IndexingOperationError,
    IndexingPartialFailureError,
    InvalidIndexingInputError,
)
from exceptions.vector import (
    CollectionConfigurationError,
    CollectionNotFoundError,
    VectorInputValidationError,
    VectorStoreAuthenticationError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from rag.indexing.models import (
    IndexableRecord,
    IndexingBatchResult,
    IndexingConfig,
    IndexingReport,
)
from rag.retrieval.keyword.encoder.base import BaseSparseEncoder
from rag.retrieval.models import SparseVector
from storage.vector import BaseVectorStore, get_vector_store
from storage.vector.models import VectorRecord

logger = get_logger(__name__)


class DocumentIndexingService:
    """Service orchestrating the Indexing / Storage stage of the RAG pipeline.

    Responsible for taking successful embedding results and chunks, validating records,
    ensuring tenant isolation, generating stable deterministic point IDs, batching writes,
    executing idempotent upserts in Qdrant with bounded retries, and recording audit reports.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        config: Optional[IndexingConfig] = None,
        sparse_encoder: Optional[BaseSparseEncoder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize DocumentIndexingService.

        Args:
            vector_store: Pre-configured BaseVectorStore instance. Defaults to default vector store.
            config: Indexing configuration. Defaults to IndexingConfig loaded from settings.
            sparse_encoder: Optional BaseSparseEncoder instance override.
            settings: Application settings. Defaults to cached app settings.
        """
        self._settings = settings or get_settings()
        self._config = config or IndexingConfig.from_settings(self._settings)
        self._vector_store = vector_store or get_vector_store(
            collection_name=self._config.collection_name,
            settings=self._settings,
        )
        self._sparse_encoder = sparse_encoder
        self._collection_verified = False

    def _get_sparse_encoder(self) -> BaseSparseEncoder:
        """Resolve active BaseSparseEncoder instance."""
        if self._sparse_encoder is not None:
            return self._sparse_encoder
        from rag.retrieval.keyword.encoder import get_sparse_encoder
        return get_sparse_encoder(
            strategy=self._config.sparse_encoder_strategy,
            settings=self._settings,
        )

    @property
    def config(self) -> IndexingConfig:
        """Return the active indexing configuration."""
        return self._config

    @property
    def vector_store(self) -> BaseVectorStore:
        """Return the active vector store repository."""
        return self._vector_store

    # --------------------------------------------------------------------------
    # Validation Helpers
    # --------------------------------------------------------------------------

    def _validate_single_record(
        self,
        record: IndexableRecord,
        expected_project_id: Optional[str] = None,
        expected_dim: Optional[int] = None,
    ) -> None:
        """Validate a single indexable record for tenant boundaries, vector integrity, and dimensions."""
        if not isinstance(record, IndexableRecord):
            raise InvalidIndexingInputError(
                f"Expected IndexableRecord instance, got '{type(record).__name__}'."
            )

        if not record.project_id or not record.project_id.strip():
            raise InvalidIndexingInputError(
                f"Record '{record.chunk_id}' has missing or empty project_id."
            )
        if not record.document_id or not record.document_id.strip():
            raise InvalidIndexingInputError(
                f"Record '{record.chunk_id}' has missing or empty document_id."
            )
        if not record.chunk_id or not record.chunk_id.strip():
            raise InvalidIndexingInputError("Record has missing or empty chunk_id.")

        # Enforce project isolation invariant
        if expected_project_id is not None and record.project_id != expected_project_id:
            raise InvalidIndexingInputError(
                f"Project isolation violation: Record '{record.chunk_id}' has project_id "
                f"'{record.project_id}', expected '{expected_project_id}'."
            )

        # Validate vector structure
        if not isinstance(record.vector, (list, tuple)) or len(record.vector) == 0:
            raise InvalidIndexingInputError(
                f"Record '{record.chunk_id}' has an empty or non-sequence vector."
            )

        # Validate vector components (must be finite floats/ints)
        for val in record.vector:
            if not isinstance(val, (float, int)) or math.isnan(val) or math.isinf(val):
                raise InvalidIndexingInputError(
                    f"Record '{record.chunk_id}' contains invalid non-finite or non-numeric vector values."
                )

        # Validate dimensionality against expected dimension
        actual_dim = len(record.vector)
        if expected_dim is not None and actual_dim != expected_dim:
            raise InvalidIndexingInputError(
                f"Record '{record.chunk_id}' vector dimension mismatch: expected {expected_dim}, got {actual_dim}."
            )

        # Validate sparse vector if present
        if record.sparse_vector is not None:
            if not isinstance(record.sparse_vector, SparseVector) and not (
                hasattr(record.sparse_vector, "indices") and hasattr(record.sparse_vector, "values")
            ):
                raise InvalidIndexingInputError(
                    f"Record '{record.chunk_id}' sparse_vector must be a SparseVector instance, "
                    f"got '{type(record.sparse_vector).__name__}'."
                )

    def _validate_records_batch(
        self, records: list[IndexableRecord]
    ) -> tuple[str, int]:
        """Validate an entire collection of records, verifying project isolation and dimension consistency."""
        if not isinstance(records, (list, tuple)):
            raise InvalidIndexingInputError(
                f"Records must be a list or tuple of IndexableRecord instances, got '{type(records).__name__}'."
            )
        if len(records) == 0:
            raise InvalidIndexingInputError("Cannot index an empty list of records.")

        first_record = records[0]
        self._validate_single_record(first_record)
        common_project_id = first_record.project_id
        common_dim = len(first_record.vector)

        # Check configuration expected vector size if set
        if self._config.expected_vector_size is not None and common_dim != self._config.expected_vector_size:
            raise InvalidIndexingInputError(
                f"Vector dimension {common_dim} does not match configured vector size {self._config.expected_vector_size}."
            )

        seen_chunk_ids: set[str] = set()
        for idx, rec in enumerate(records):
            self._validate_single_record(
                rec,
                expected_project_id=common_project_id,
                expected_dim=common_dim,
            )
            if rec.chunk_id in seen_chunk_ids:
                raise InvalidIndexingInputError(
                    f"Duplicate chunk_id '{rec.chunk_id}' detected at index {idx} in indexing batch."
                )
            seen_chunk_ids.add(rec.chunk_id)

        return common_project_id, common_dim

    # --------------------------------------------------------------------------
    # Collection Verification
    # --------------------------------------------------------------------------

    async def ensure_collection(self, vector_size: int) -> None:
        """Verify or create the Qdrant collection idempotently."""
        if self._collection_verified:
            return

        try:
            logger.info(
                "Ensuring Qdrant collection '%s' exists with vector_size=%d, distance=%s",
                self._vector_store.collection_name,
                vector_size,
                self._config.distance,
            )
            await self._vector_store.ensure_collection_exists(
                vector_size=vector_size,
                distance=self._config.distance,
            )
            self._collection_verified = True
        except CollectionConfigurationError as exc:
            logger.error("Qdrant collection configuration mismatch: %s", exc)
            raise IndexingConfigurationError(
                f"Collection configuration mismatch for '{self._vector_store.collection_name}': {exc}",
                original_error=exc,
            ) from exc
        except VectorStoreAuthenticationError as exc:
            logger.error("Qdrant authentication failed during collection verification")
            raise IndexingOperationError(
                f"Authentication failed with Qdrant: {exc}", original_error=exc
            ) from exc
        except VectorStoreConnectionError as exc:
            logger.error("Qdrant connection error during collection verification: %s", exc)
            raise IndexingConnectionError(
                f"Failed to connect to Qdrant: {exc}", original_error=exc
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during collection verification: %s", exc)
            raise IndexingOperationError(
                f"Failed to verify collection '{self._vector_store.collection_name}': {exc}",
                original_error=exc,
            ) from exc

    # --------------------------------------------------------------------------
    # Batched Upsert with Bounded Retries
    # --------------------------------------------------------------------------

    async def _upsert_batch_with_retries(
        self,
        batch_records: list[VectorRecord],
        batch_index: int,
    ) -> int:
        """Upsert a single batch of VectorRecords with bounded retries for transient failures."""
        attempts = 0
        delay = self._config.retry_delay

        while True:
            attempts += 1
            try:
                logger.debug(
                    "Upserting batch %d (%d points) to '%s' (attempt %d/%d)",
                    batch_index,
                    len(batch_records),
                    self._vector_store.collection_name,
                    attempts,
                    self._config.max_retries + 1,
                )
                upserted = await self._vector_store.upsert(batch_records)
                return upserted
            except (VectorInputValidationError, VectorStoreAuthenticationError, CollectionConfigurationError) as exc:
                # Permanent failures: do not retry
                logger.error("Permanent vector store error on batch %d: %s", batch_index, exc)
                raise IndexingOperationError(
                    f"Permanent vector store error on batch {batch_index}: {exc}",
                    original_error=exc,
                ) from exc
            except (VectorStoreConnectionError, VectorUpsertError, VectorStoreError) as exc:
                if attempts > self._config.max_retries:
                    logger.error(
                        "Exceeded max retries (%d) on batch %d: %s",
                        self._config.max_retries,
                        batch_index,
                        exc,
                    )
                    raise IndexingConnectionError(
                        f"Failed to upsert batch {batch_index} after {attempts} attempts: {exc}",
                        original_error=exc,
                    ) from exc

                logger.warning(
                    "Transient error on batch %d (attempt %d/%d): %s. Retrying in %.2fs...",
                    batch_index,
                    attempts,
                    self._config.max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
            except Exception as exc:
                # Unexpected exceptions
                logger.error("Unexpected error on batch %d: %s", batch_index, exc)
                raise IndexingOperationError(
                    f"Unexpected error upserting batch {batch_index}: {exc}",
                    original_error=exc,
                ) from exc

    # --------------------------------------------------------------------------
    # Main Indexing Entrypoints
    # --------------------------------------------------------------------------

    async def index_records(
        self,
        records: list[IndexableRecord],
        strict: bool = True,
    ) -> IndexingReport:
        """Index a list of pre-constructed IndexableRecords into Qdrant in bounded batches.

        Args:
            records: List of IndexableRecord instances to index.
            strict: If True, raises IndexingPartialFailureError if any batch fails.
                   If False, returns the report detailing succeeded and failed batches.

        Returns:
            IndexingReport: Detailed execution and audit report.

        Raises:
            InvalidIndexingInputError: If inputs, vectors, dimensions, or tenant isolation are invalid.
            IndexingPartialFailureError: If strict is True and one or more batches fail.
            IndexingOperationError: If permanent or critical infrastructure errors occur.
        """
        start_time = time.perf_counter()
        project_id, vector_dim = self._validate_records_batch(records)

        if self._config.ensure_collection:
            await self.ensure_collection(vector_dim)

        batch_size = self._config.batch_size
        total_attempted = len(records)
        total_indexed = 0
        total_failed = 0
        batch_results: list[IndexingBatchResult] = []
        all_point_ids: list[str] = []
        failed_chunk_ids: list[str] = []

        logger.info(
            "Starting vector indexing for project '%s' (%d records, batch_size=%d, collection='%s')",
            project_id,
            total_attempted,
            batch_size,
            self._vector_store.collection_name,
        )

        # Slice records into bounded batches
        for i in range(0, total_attempted, batch_size):
            batch_records = records[i : i + batch_size]
            batch_idx = (i // batch_size) + 1
            batch_chunk_ids = tuple(r.chunk_id for r in batch_records)
            batch_point_ids = tuple(r.point_id for r in batch_records)

            # Map to VectorRecords
            vector_records = [r.to_vector_record() for r in batch_records]

            try:
                upserted = await self._upsert_batch_with_retries(vector_records, batch_idx)
                total_indexed += upserted
                all_point_ids.extend(batch_point_ids)
                batch_results.append(
                    IndexingBatchResult(
                        batch_index=batch_idx,
                        points_count=upserted,
                        point_ids=batch_point_ids,
                        chunk_ids=batch_chunk_ids,
                        status="success",
                        error=None,
                    )
                )
            except Exception as exc:
                failed_count = len(batch_records)
                total_failed += failed_count
                failed_chunk_ids.extend(batch_chunk_ids)
                error_msg = str(exc)
                logger.error(
                    "Batch %d failed to index (%d records): %s",
                    batch_idx,
                    failed_count,
                    error_msg,
                )
                batch_results.append(
                    IndexingBatchResult(
                        batch_index=batch_idx,
                        points_count=0,
                        point_ids=(),
                        chunk_ids=batch_chunk_ids,
                        status="failed",
                        error=error_msg,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        model_name = records[0].embedding_model if records else None
        failed_batches_count = sum(1 for b in batch_results if b.status == "failed")

        report = IndexingReport(
            total_attempted=total_attempted,
            total_indexed=total_indexed,
            total_failed=total_failed,
            batches_count=len(batch_results),
            failed_batches_count=failed_batches_count,
            point_ids=tuple(all_point_ids),
            failed_chunk_ids=tuple(failed_chunk_ids),
            collection_name=self._vector_store.collection_name,
            model=model_name,
            elapsed_ms=elapsed_ms,
            batches=tuple(batch_results),
        )

        logger.info(
            "Indexing finished for project '%s' in %.2fms (attempted=%d, indexed=%d, failed=%d)",
            project_id,
            elapsed_ms,
            total_attempted,
            total_indexed,
            total_failed,
        )

        if strict and total_failed > 0:
            raise IndexingPartialFailureError(
                f"Indexing partially failed: {total_failed} of {total_attempted} points could not be indexed.",
                report=report,
            )

        return report

    async def index_document(
        self,
        document: Any,
        embeddings: Any,
        sparse_vectors: Optional[Sequence[SparseVector]] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: str = "voyage",
        strict: bool = True,
    ) -> IndexingReport:
        """Index a document and its corresponding generated representations into Qdrant.

        Integrates with the output of Contextual Enrichment, Embedding Generation,
        and Sparse Representation Generation.

        Args:
            document: Document object carrying `chunks` and `project_id` (e.g. EnrichedDocument).
            embeddings: EmbeddingBatchResult, list of EmbeddingResults, or list of float vectors.
            sparse_vectors: Optional sequence of SparseVector representations corresponding 1-to-1
                with document chunks. Required when sparse indexing is enabled.
            embedding_model: Optional model name override.
            embedding_provider: Embedding provider identifier (default: 'voyage').
            strict: If True, raises IndexingPartialFailureError on partial failure.

        Returns:
            IndexingReport: Detailed execution and audit report.

        Raises:
            InvalidIndexingInputError: If document, embeddings, or sparse_vectors are mismatched,
                invalid, or missing when required.
        """
        chunks = getattr(document, "chunks", None)
        if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
            raise InvalidIndexingInputError(
                "Document must contain a non-empty list of chunks under 'chunks'."
            )

        # Extract vectors and model name from rag.embeddings container
        vectors: list[list[float]]
        model: Optional[str] = embedding_model

        if isinstance(embeddings, EmbeddingBatchResult):
            vectors = list(embeddings.embeddings)
            model = model or embeddings.model
        elif isinstance(embeddings, (list, tuple)):
            vectors = []
            for item in embeddings:
                if isinstance(item, EmbeddingResult):
                    vectors.append(item.vector)
                    if model is None:
                        model = item.model
                elif isinstance(item, (list, tuple)):
                    vectors.append(list(item))
                else:
                    raise InvalidIndexingInputError(
                        f"Expected embedding vector or EmbeddingResult, got '{type(item).__name__}'."
                    )
        else:
            raise InvalidIndexingInputError(
                f"Unsupported embeddings format '{type(embeddings).__name__}'. Expected EmbeddingBatchResult or list of vectors."
            )

        if len(chunks) != len(vectors):
            raise InvalidIndexingInputError(
                f"Mismatched chunk and embedding counts: document has {len(chunks)} chunks, "
                f"but received {len(vectors)} embedding vectors."
            )

        # Validate sparse representations when sparse indexing is enabled
        validated_sparse_vectors: Optional[Sequence[Optional[SparseVector]]] = None
        if self._config.sparse_indexing_enabled:
            if sparse_vectors is None:
                raise InvalidIndexingInputError(
                    "sparse_vectors must be provided when sparse indexing is enabled."
                )
            if not isinstance(sparse_vectors, (list, tuple)):
                raise InvalidIndexingInputError(
                    f"sparse_vectors must be a sequence of SparseVector instances, got '{type(sparse_vectors).__name__}'."
                )
            if len(chunks) != len(sparse_vectors):
                raise InvalidIndexingInputError(
                    f"Mismatched chunk and sparse vector counts: document has {len(chunks)} chunks, "
                    f"but received {len(sparse_vectors)} sparse vectors."
                )
            for idx, sv in enumerate(sparse_vectors):
                if sv is not None and not isinstance(sv, SparseVector) and not (
                    hasattr(sv, "indices") and hasattr(sv, "values")
                ):
                    raise InvalidIndexingInputError(
                        f"Item at index {idx} in sparse_vectors must be a SparseVector instance, "
                        f"got '{type(sv).__name__}'."
                    )
            validated_sparse_vectors = sparse_vectors

        # Construct IndexableRecords preserving chunk metadata and representation config
        records: list[IndexableRecord] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            sparse_vec = (
                validated_sparse_vectors[idx]
                if validated_sparse_vectors is not None
                else None
            )
            rec = IndexableRecord.from_chunk_and_vector(
                chunk=chunk,
                vector=vector,
                embedding_model=model,
                embedding_provider=embedding_provider,
                sparse_vector=sparse_vec,
                sparse_encoder_strategy=self._config.sparse_encoder_strategy if sparse_vec is not None else None,
                sparse_encoder_version=self._config.sparse_encoder_version if sparse_vec is not None else None,
            )
            records.append(rec)

        return await self.index_records(records, strict=strict)

    # --------------------------------------------------------------------------
    # Document Lifecycle Operations
    # --------------------------------------------------------------------------

    async def delete_document(self, project_id: str, document_id: str) -> bool:
        """Idempotently delete all searchable vectors associated with a document.

        Enforces project isolation so deletions never affect other tenants.

        Args:
            project_id: Mandatory project/tenant ID.
            document_id: Document ID to remove from index.

        Returns:
            bool: True if deletion succeeded.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise InvalidIndexingInputError("project_id must be a non-empty string.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidIndexingInputError("document_id must be a non-empty string.")

        try:
            logger.info(
                "Deleting all vectors for document '%s' in project '%s'",
                document_id,
                project_id,
            )
            return await self._vector_store.delete_by_filter(
                project_id=project_id.strip(),
                document_id=document_id.strip(),
            )
        except VectorInputValidationError as exc:
            raise InvalidIndexingInputError(str(exc), original_error=exc) from exc
        except Exception as exc:
            logger.error(
                "Failed to delete document '%s' in project '%s': %s",
                document_id,
                project_id,
                exc,
            )
            raise IndexingOperationError(
                f"Failed to delete vectors for document '{document_id}': {exc}",
                original_error=exc,
            ) from exc

    async def delete_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
    ) -> bool:
        """Idempotently delete all searchable vectors for a specific document version.

        Args:
            project_id: Mandatory project/tenant ID.
            document_id: Document ID.
            document_version_id: Specific document version ID to delete.

        Returns:
            bool: True if deletion succeeded.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise InvalidIndexingInputError("project_id must be a non-empty string.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidIndexingInputError("document_id must be a non-empty string.")
        if not isinstance(document_version_id, str) or not document_version_id.strip():
            raise InvalidIndexingInputError("document_version_id must be a non-empty string.")

        try:
            logger.info(
                "Deleting vectors for document '%s' version '%s' in project '%s'",
                document_id,
                document_version_id,
                project_id,
            )
            return await self._vector_store.delete_by_filter(
                project_id=project_id.strip(),
                document_id=document_id.strip(),
                document_version_id=document_version_id.strip(),
            )
        except VectorInputValidationError as exc:
            raise InvalidIndexingInputError(str(exc), original_error=exc) from exc
        except Exception as exc:
            logger.error(
                "Failed to delete document '%s' version '%s': %s",
                document_id,
                document_version_id,
                exc,
            )
            raise IndexingOperationError(
                f"Failed to delete vectors for document version: {exc}",
                original_error=exc,
            ) from exc

    async def delete_chunks(self, project_id: str, point_ids: list[str]) -> int:
        """Delete specific vector points by their stable point IDs.

        Args:
            project_id: Mandatory project ID for logging and validation.
            point_ids: List of point ID UUID strings.

        Returns:
            int: Number of point IDs requested for deletion.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise InvalidIndexingInputError("project_id must be a non-empty string.")
        if not isinstance(point_ids, (list, tuple)) or len(point_ids) == 0:
            raise InvalidIndexingInputError("point_ids list cannot be empty.")

        try:
            logger.info(
                "Deleting %d specific points for project '%s'",
                len(point_ids),
                project_id,
            )
            return await self._vector_store.delete(point_ids=list(point_ids))
        except VectorInputValidationError as exc:
            raise InvalidIndexingInputError(str(exc), original_error=exc) from exc
        except Exception as exc:
            raise IndexingOperationError(
                f"Failed to delete points: {exc}", original_error=exc
            ) from exc


_default_indexing_service: Optional[DocumentIndexingService] = None


def get_indexing_service(
    vector_store: Optional[BaseVectorStore] = None,
    config: Optional[IndexingConfig] = None,
    settings: Optional[Settings] = None,
) -> DocumentIndexingService:
    """Get or create the singleton DocumentIndexingService instance."""
    global _default_indexing_service

    if vector_store is not None or config is not None:
        return DocumentIndexingService(
            vector_store=vector_store,
            config=config,
            settings=settings,
        )

    if _default_indexing_service is None:
        _default_indexing_service = DocumentIndexingService(settings=settings)

    return _default_indexing_service


def reset_indexing_service() -> None:
    """Reset the cached default indexing service instance. Useful for tests."""
    global _default_indexing_service
    _default_indexing_service = None

