"""Document chunking service orchestrating structure-aware chunking and recursive splitting."""

import asyncio
import time

from app.core.logging import get_logger
from chunking.models import (
    ChunkedDocument,
    ChunkingConfig,
    ChunkingReport,
    DocumentChunk,
)
from chunking.sizer import BaseChunkSizer
from chunking.splitter import StructureAwareChunker
from exceptions.chunking import (
    ChunkingConfigurationError,
    ChunkingProcessingError,
    InvalidChunkingInputError,
)
from normalization.models import NormalizedDocument

logger = get_logger(__name__)


class DocumentChunkingService:
    """Service orchestrating structure-aware document chunking for the RAG pipeline.

    Consumes a NormalizedDocument, analyzes document structure and hierarchy in O(N) time,
    identifies meaningful structural units, groups content up to max_chunk_size, and
    recursively splits oversized units using natural boundaries while preserving complete
    provenance and hierarchical context.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        sizer: BaseChunkSizer | None = None,
    ) -> None:
        if config is not None and sizer is not None:
            # If both provided, override sizer on config
            self._config = ChunkingConfig(
                max_chunk_size=config.max_chunk_size,
                min_chunk_size=config.min_chunk_size,
                chunk_overlap=config.chunk_overlap,
                preserve_hierarchy=config.preserve_hierarchy,
                sizer=sizer,
            )
        elif config is not None:
            self._config = config
        elif sizer is not None:
            self._config = ChunkingConfig(sizer=sizer)
        else:
            self._config = ChunkingConfig()

    @property
    def config(self) -> ChunkingConfig:
        """Return the active chunking configuration."""
        return self._config

    def chunk_sync(self, document: NormalizedDocument) -> ChunkedDocument:
        """Chunk a NormalizedDocument synchronously.

        Strictly enforces that the input document has passed through Normalization.
        Extracts structural sections, generates chunks within configured constraints,
        and returns a ChunkedDocument.

        Args:
            document: NormalizedDocument from the Normalization stage.

        Returns:
            ChunkedDocument containing ordered chunks, reports, and metadata.

        Raises:
            InvalidChunkingInputError: If document is not a NormalizedDocument instance.
            ChunkingProcessingError: If chunking fails unexpectedly.
        """
        if not isinstance(document, NormalizedDocument):
            raise InvalidChunkingInputError(
                f"Expected NormalizedDocument, got '{type(document).__name__}'."
            )

        start_time = time.perf_counter()
        doc_id = document.document_id
        project_id = document.project_id
        doc_type_val = document.document_type.value

        logger.info(
            "Starting chunking for document '%s' (project: '%s', type: '%s', elements: %d)",
            doc_id,
            project_id,
            doc_type_val,
            document.total_elements,
        )

        try:
            # Preserve existing reports from previous stages
            cleaning_report = getattr(document, "cleaning_report", None)
            normalization_report = getattr(document, "normalization_report", None)

            # Handle empty document edge case
            if document.total_elements == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                report = ChunkingReport(
                    total_input_elements=0,
                    total_chunks=0,
                    recursively_split_count=0,
                    structural_chunks_count=0,
                    fallback_chunks_count=0,
                    elapsed_ms=elapsed_ms,
                )
                return ChunkedDocument(
                    document_id=doc_id,
                    project_id=project_id,
                    document_type=document.document_type,
                    chunks=[],
                    document_version_id=document.document_version_id,
                    source_metadata=dict(document.source_metadata),
                    parser_metadata=dict(document.parser_metadata),
                    cleaning_report=cleaning_report,
                    normalization_report=normalization_report,
                    chunking_report=report,
                )

            # Execute structure-aware chunking
            chunker = StructureAwareChunker(config=self._config)
            chunks = chunker.chunk_document(
                document_id=doc_id,
                project_id=project_id,
                elements=document.elements,
                document_version_id=document.document_version_id,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = ChunkingReport(
                total_input_elements=document.total_elements,
                total_chunks=len(chunks),
                recursively_split_count=chunker.recursively_split_count,
                structural_chunks_count=chunker.structural_chunks_count,
                fallback_chunks_count=chunker.fallback_chunks_count,
                elapsed_ms=elapsed_ms,
            )

            logger.info(
                "Successfully chunked document '%s' in %.2fms "
                "(elements: %d, chunks: %d, recursively split: %d, structural: %d, fallback: %d)",
                doc_id,
                elapsed_ms,
                document.total_elements,
                len(chunks),
                chunker.recursively_split_count,
                chunker.structural_chunks_count,
                chunker.fallback_chunks_count,
            )

            return ChunkedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                chunks=chunks,
                document_version_id=document.document_version_id,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                cleaning_report=cleaning_report,
                normalization_report=normalization_report,
                chunking_report=report,
            )

        except InvalidChunkingInputError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure chunking document '%s': %s",
                doc_id,
                str(exc),
            )
            raise ChunkingProcessingError(
                f"Unexpected failure chunking document '{doc_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    async def chunk(self, document: NormalizedDocument) -> ChunkedDocument:
        """Chunk a NormalizedDocument asynchronously.

        Offloads CPU-bound chunking operations to a worker thread via asyncio.to_thread.

        Args:
            document: Structured NormalizedDocument.

        Returns:
            ChunkedDocument.
        """
        return await asyncio.to_thread(self.chunk_sync, document)

    async def chunk_batch(
        self,
        documents: list[NormalizedDocument],
    ) -> list[ChunkedDocument]:
        """Chunk multiple NormalizedDocuments concurrently.

        Args:
            documents: List of NormalizedDocument instances.

        Returns:
            List of ChunkedDocument results preserving input document order.
        """
        tasks = [self.chunk(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_chunking_service: DocumentChunkingService | None = None


def get_chunking_service(
    config: ChunkingConfig | None = None,
    sizer: BaseChunkSizer | None = None,
) -> DocumentChunkingService:
    """Get or create the singleton DocumentChunkingService instance."""
    global _default_chunking_service
    if config is not None or sizer is not None:
        return DocumentChunkingService(config=config, sizer=sizer)
    if _default_chunking_service is None:
        _default_chunking_service = DocumentChunkingService()
    return _default_chunking_service


def reset_chunking_service() -> None:
    """Reset the singleton chunking service instance. Useful for test isolation."""
    global _default_chunking_service
    _default_chunking_service = None
