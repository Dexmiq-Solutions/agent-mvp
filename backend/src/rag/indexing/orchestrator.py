"""End-to-End Indexing Service orchestrating the complete RAG indexing pipeline."""

from dataclasses import dataclass, field
import time
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import Settings, get_settings
from observability.logging import get_logger
from db.session import get_async_session_maker
from exceptions.indexing import (
    IndexingConfigurationError,
    IndexingError,
    IndexingOperationError,
    InvalidIndexingInputError,
)
from models.chunk import ChunkModel
from models.document import DocumentVersionModel, DocumentVersionStatus
from rag.acquisition import BaseAcquisitionService, SourceDocument, get_acquisition_service
from rag.chunking import DocumentChunkingService, get_chunking_service
from rag.chunking.models import ChunkedDocument
from rag.cleaning import DocumentCleaningService, get_cleaning_service
from rag.cleaning.models import CleanedDocument
from rag.contextual_enrichment import (
    DocumentContextualEnrichmentService,
    get_contextual_enrichment_service,
)
from rag.contextual_enrichment.models import ContextuallyEnrichedDocument
from rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from rag.indexing.models import IndexingReport
from rag.indexing.persistence import (
    ChunkPersistenceService,
    get_chunk_persistence_service,
)
from rag.indexing.representations import (
    extract_representation_texts,
    generate_sparse_representations,
)
from rag.indexing.service import DocumentIndexingService, get_indexing_service
from rag.ingestion import BaseIngestionService, DocumentSourceReference, IngestedDocument, get_ingestion_service
from rag.metadata_enrichment import (
    DocumentMetadataEnrichmentService,
    get_metadata_enrichment_service,
)
from rag.metadata_enrichment.models import EnrichedDocument
from rag.normalization import DocumentNormalizationService, get_normalization_service
from rag.normalization.models import NormalizedDocument
from rag.parsing import DocumentParsingService, get_parsing_service
from rag.parsing.models import ParsedDocument
from storage.object import BaseObjectStorage, get_object_storage

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexingPipelineReport:
    """Comprehensive execution report of an end-to-end indexing operation."""

    status: str
    project_id: str
    document_id: str
    document_version_id: str
    total_chunks: int
    persisted_chunks: int
    indexed_points: int
    dense_model: str
    sparse_enabled: bool
    contextual_enrichment_enabled: bool
    elapsed_ms: float
    indexing_report: IndexingReport
    stale_versions_pruned: int = 0
    stage_reports: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary."""
        return {
            "status": self.status,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "total_chunks": self.total_chunks,
            "persisted_chunks": self.persisted_chunks,
            "indexed_points": self.indexed_points,
            "dense_model": self.dense_model,
            "sparse_enabled": self.sparse_enabled,
            "contextual_enrichment_enabled": self.contextual_enrichment_enabled,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "stale_versions_pruned": self.stale_versions_pruned,
            "indexing_report": self.indexing_report.to_dict(),
            "stage_reports": self.stage_reports,
        }


class EndToEndIndexingService:
    """Orchestrator executing the complete 10-stage RAG indexing pipeline.

    Takes a verified DocumentVersion through:
      1. Acquisition
      2. Ingestion
      3. Parsing & Extraction
      4. Cleaning
      5. Normalization
      6. Chunking
      7. Metadata Enrichment
      8. Optional Contextual Enrichment
      9. Dense Embedding + Sparse Representation
      10. PostgreSQL Chunk Persistence
      11. Qdrant Vector Indexing
      12. Stale Version Pruning in Vector Store
    """

    def __init__(
        self,
        acquisition_service: Optional[BaseAcquisitionService] = None,
        ingestion_service: Optional[BaseIngestionService] = None,
        parsing_service: Optional[DocumentParsingService] = None,
        cleaning_service: Optional[DocumentCleaningService] = None,
        normalization_service: Optional[DocumentNormalizationService] = None,
        chunking_service: Optional[DocumentChunkingService] = None,
        metadata_enrichment_service: Optional[DocumentMetadataEnrichmentService] = None,
        contextual_enrichment_service: Optional[DocumentContextualEnrichmentService] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunk_persistence_service: Optional[ChunkPersistenceService] = None,
        indexing_service: Optional[DocumentIndexingService] = None,
        storage: Optional[BaseObjectStorage] = None,
        session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
        cleanup_stale_versions: bool = True,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize EndToEndIndexingService.

        Injects or lazily initializes all required indexing stage services.
        """
        self._settings = settings or get_settings()
        self._storage = storage or get_object_storage(settings=self._settings)
        self._session_maker = session_maker
        self._cleanup_stale_versions = cleanup_stale_versions

        self._acquisition_service = acquisition_service or get_acquisition_service(
            storage=self._storage, settings=self._settings
        )
        self._ingestion_service = ingestion_service or get_ingestion_service(
            storage=self._storage, settings=self._settings
        )
        self._parsing_service = parsing_service or get_parsing_service()
        self._cleaning_service = cleaning_service or get_cleaning_service()
        self._normalization_service = normalization_service or get_normalization_service()
        self._chunking_service = chunking_service or get_chunking_service()
        self._metadata_enrichment_service = (
            metadata_enrichment_service or get_metadata_enrichment_service()
        )
        self._contextual_enrichment_service = (
            contextual_enrichment_service or get_contextual_enrichment_service()
        )
        self._embedding_provider = embedding_provider or get_embedding_provider(
            settings=self._settings
        )
        self._chunk_persistence_service = (
            chunk_persistence_service
            or get_chunk_persistence_service(
                session_maker=self._session_maker, settings=self._settings
            )
        )
        self._indexing_service = indexing_service or get_indexing_service(
            settings=self._settings
        )

    async def _prune_stale_qdrant_versions(
        self,
        project_id: str,
        document_id: str,
        current_version_id: str,
    ) -> int:
        """Prune prior READY version vector points from Qdrant upon new version indexing success."""
        if not self._cleanup_stale_versions:
            return 0

        maker = self._session_maker or get_async_session_maker(settings=self._settings)
        pruned_count = 0

        try:
            async with maker() as session:
                stmt = select(DocumentVersionModel.id).where(
                    DocumentVersionModel.document_id == document_id,
                    DocumentVersionModel.project_id == project_id,
                    DocumentVersionModel.id != current_version_id,
                    DocumentVersionModel.status == DocumentVersionStatus.READY.value,
                )
                res = await session.execute(stmt)
                prior_version_ids = res.scalars().all()

            for old_ver_id in prior_version_ids:
                logger.info(
                    "Pruning superseded version '%s' of doc '%s' from Qdrant (project: '%s')",
                    old_ver_id,
                    document_id,
                    project_id,
                )
                await self._indexing_service.delete_document_version(
                    project_id=project_id,
                    document_id=document_id,
                    document_version_id=old_ver_id,
                )
                pruned_count += 1

            return pruned_count
        except Exception as exc:
            logger.warning(
                "Stale version pruning encountered non-fatal error for doc '%s': %s",
                document_id,
                exc,
            )
            return pruned_count

    async def index_document_version(
        self,
        project_id: str,
        document_id: str,
        document_version_id: str,
        storage_bucket: str,
        storage_path: str,
        original_filename: str,
        content_type: Optional[str] = None,
    ) -> IndexingPipelineReport:
        """Execute the end-to-end RAG indexing pipeline for a specific document version.

        Args:
            project_id: Verified project ID enforcing isolation.
            document_id: Verified logical document ID.
            document_version_id: Verified physical document version ID.
            storage_bucket: Bucket containing source binary.
            storage_path: Relative storage path of file.
            original_filename: Name of the uploaded file.
            content_type: Optional MIME content type.

        Returns:
            IndexingPipelineReport: Detailed summary report of the indexing run.

        Raises:
            Exception: If any pipeline stage fails.
        """
        start_time = time.perf_counter()
        logger.info(
            "End-to-End Indexing started for version '%s' (project: '%s', doc: '%s', path: '%s')",
            document_version_id,
            project_id,
            document_id,
            storage_path,
        )

        stage_reports: dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # Stage 1: Acquisition
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        source_doc: SourceDocument = await self._acquisition_service.acquire_document(
            project_id=project_id,
            storage_path=storage_path,
        )
        stage_reports["acquisition"] = {
            "source_id": source_doc.source_id,
            "document_type": source_doc.document_type.value,
            "size_bytes": source_doc.size_bytes,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 1 (Acquisition) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 2: Ingestion
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        doc_source_ref = DocumentSourceReference(
            project_id=project_id,
            storage_path=source_doc.storage_path,
            storage_bucket=storage_bucket,
            original_filename=original_filename,
            content_type=content_type or source_doc.content_type,
            document_type=source_doc.document_type,
            document_id=document_id,
            document_version_id=document_version_id,
            source_metadata=dict(source_doc.metadata),
        )
        ingested_doc: IngestedDocument = await self._ingestion_service.ingest(
            source=doc_source_ref,
            project_id=project_id,
        )
        stage_reports["ingestion"] = {
            "detected_type": ingested_doc.detected_document_type.value,
            "size_bytes": ingested_doc.size_bytes,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 2 (Ingestion) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 3: Parsing & Extraction
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        parsed_doc: Any = await self._parsing_service.parse(ingested_doc)
        stage_reports["parsing"] = {
            "total_elements": parsed_doc.total_elements,
            "headings_count": len(parsed_doc.headings),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 3 (Parsing) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 4: Cleaning
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        cleaned_doc: CleanedDocument = await self._cleaning_service.clean(parsed_doc)
        stage_reports["cleaning"] = {
            "removed_elements": cleaned_doc.cleaning_report.removed_count,
            "preserved_elements": cleaned_doc.cleaning_report.preserved_count,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 4 (Cleaning) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 5: Normalization
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        normalized_doc: NormalizedDocument = await self._normalization_service.normalize(cleaned_doc)
        stage_reports["normalization"] = {
            "normalized_elements": normalized_doc.normalization_report.normalized_count,
            "unchanged_elements": normalized_doc.normalization_report.unchanged_count,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 5 (Normalization) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 6: Chunking
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        chunked_doc: ChunkedDocument = await self._chunking_service.chunk(normalized_doc)
        stage_reports["chunking"] = {
            "total_chunks": chunked_doc.total_chunks,
            "recursively_split": chunked_doc.chunking_report.recursively_split_count,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 6 (Chunking) complete for version '%s' (%d chunks)", document_version_id, chunked_doc.total_chunks)

        if chunked_doc.total_chunks == 0:
            raise IndexingOperationError(
                f"Document version '{document_version_id}' yielded 0 chunks after chunking."
            )

        # ----------------------------------------------------------------------
        # Stage 7: Metadata Enrichment
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        enriched_doc: EnrichedDocument = await self._metadata_enrichment_service.enrich(chunked_doc)
        stage_reports["metadata_enrichment"] = {
            "enriched_chunks": len(enriched_doc.chunks),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 7 (Metadata Enrichment) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 8: Optional Contextual Enrichment
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        contextual_doc: ContextuallyEnrichedDocument = (
            await self._contextual_enrichment_service.enrich(enriched_doc)
        )
        contextual_report = getattr(
            contextual_doc,
            "contextual_enrichment_report",
            getattr(contextual_doc, "contextual_report", None),
        )
        is_enabled = contextual_report.enabled if contextual_report else False
        stage_reports["contextual_enrichment"] = {
            "enabled": is_enabled,
            "strategy": contextual_report.strategy_used if contextual_report else "none",
            "enriched_count": contextual_report.enriched_chunks_count if contextual_report else 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug(
            "Stage 8 (Contextual Enrichment) complete for version '%s' (enabled=%s)",
            document_version_id,
            is_enabled,
        )

        # ----------------------------------------------------------------------
        # Stage 9: Representation Generation (Dense Embeddings + Sparse Vectors)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        representation_texts = extract_representation_texts(contextual_doc)

        # Dense Embedding generation (batched)
        embedding_result = await self._embedding_provider.embed_batch(
            texts=representation_texts,
            input_type="document",
        )

        # Sparse Representation generation (if enabled)
        sparse_indexing_enabled = self._indexing_service.config.sparse_indexing_enabled
        sparse_vectors = None
        if sparse_indexing_enabled:
            sparse_vectors = generate_sparse_representations(
                document_or_chunks=contextual_doc,
                settings=self._settings,
            )

        stage_reports["representations"] = {
            "dense_model": self._embedding_provider.model_name,
            "dense_count": len(embedding_result.embeddings),
            "sparse_enabled": sparse_indexing_enabled,
            "sparse_count": len(sparse_vectors) if sparse_vectors else 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 9 (Representation Generation) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 10: PostgreSQL Chunk Persistence (Atomic Short Transaction)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        persisted_chunks = await self._chunk_persistence_service.persist_chunks(
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
            chunks=contextual_doc.chunks,
        )
        stage_reports["postgres_persistence"] = {
            "persisted_count": len(persisted_chunks),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 10 (PostgreSQL Persistence) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 11: Qdrant Vector Indexing (Batched Upsert with Retries)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        indexing_report = await self._indexing_service.index_document(
            document=contextual_doc,
            embeddings=embedding_result,
            sparse_vectors=sparse_vectors,
            embedding_model=self._embedding_provider.model_name,
            strict=True,
        )
        stage_reports["qdrant_indexing"] = {
            "total_indexed": indexing_report.total_indexed,
            "total_failed": indexing_report.total_failed,
            "collection": indexing_report.collection_name,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        logger.debug("Stage 11 (Qdrant Vector Indexing) complete for version '%s'", document_version_id)

        # ----------------------------------------------------------------------
        # Stage 12: Stale Version Pruning (Qdrant)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        stale_pruned = await self._prune_stale_qdrant_versions(
            project_id=project_id,
            document_id=document_id,
            current_version_id=document_version_id,
        )
        stage_reports["stale_version_pruning"] = {
            "pruned_versions_count": stale_pruned,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        pipeline_report = IndexingPipelineReport(
            status="success",
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
            total_chunks=chunked_doc.total_chunks,
            persisted_chunks=len(persisted_chunks),
            indexed_points=indexing_report.total_indexed,
            dense_model=self._embedding_provider.model_name,
            sparse_enabled=sparse_indexing_enabled,
            contextual_enrichment_enabled=contextual_report.enabled,
            elapsed_ms=total_elapsed,
            indexing_report=indexing_report,
            stale_versions_pruned=stale_pruned,
            stage_reports=stage_reports,
        )

        logger.info(
            "End-to-End Indexing successfully completed for version '%s' in %.2fms (chunks: %d, indexed: %d)",
            document_version_id,
            total_elapsed,
            pipeline_report.total_chunks,
            pipeline_report.indexed_points,
        )
        return pipeline_report

    # Interoperability alias
    process_document_version = index_document_version


_default_end_to_end_indexing_service: Optional[EndToEndIndexingService] = None


def get_end_to_end_indexing_service(
    storage: Optional[BaseObjectStorage] = None,
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> EndToEndIndexingService:
    """Get or create the singleton EndToEndIndexingService instance."""
    global _default_end_to_end_indexing_service
    if storage is not None or session_maker is not None or kwargs:
        return EndToEndIndexingService(
            storage=storage,
            session_maker=session_maker,
            settings=settings,
            **kwargs,
        )

    if _default_end_to_end_indexing_service is None:
        _default_end_to_end_indexing_service = EndToEndIndexingService(settings=settings)
    return _default_end_to_end_indexing_service


def reset_end_to_end_indexing_service() -> None:
    """Reset the cached default end-to-end indexing service instance. Useful for tests."""
    global _default_end_to_end_indexing_service
    _default_end_to_end_indexing_service = None
