"""Document contextual enrichment service orchestrating document-level enrichment."""

import asyncio
import time
from typing import Any

from observability.logging import get_logger
from rag.contextual_enrichment.base import BaseContextProvider, DocumentContext
from rag.contextual_enrichment.models import (
    ContextualEnrichmentConfig,
    ContextualEnrichmentReport,
    ContextuallyEnrichedChunk,
    ContextuallyEnrichedDocument,
)
from rag.contextual_enrichment.providers.structured import StructuredContextProvider
from exceptions.contextual_enrichment import (
    ContextualEnrichmentConfigurationError,
    ContextualEnrichmentError,
    ContextualEnrichmentProcessingError,
    ContextualEnrichmentProviderError,
    InvalidContextualEnrichmentInputError,
)
from rag.metadata_enrichment.models import EnrichedChunk, EnrichedDocument

logger = get_logger(__name__)


class DocumentContextualEnrichmentService:
    """Service orchestrating document-level contextual enrichment for RAG indexing.

    Applies enrichment at document/indexing-run scope before embedding generation.
    When disabled, chunks pass through with verbatim content and 0ms model overhead.
    When enabled, adds situational context to the representation passed to embeddings
    while strictly preserving original chunk content, metadata, and tenant isolation.
    """

    def __init__(self, config: ContextualEnrichmentConfig | None = None) -> None:
        self._config = config or ContextualEnrichmentConfig.from_settings()
        self._provider: BaseContextProvider

        if self._config.provider is not None:
            self._provider = self._config.provider
        else:
            self._provider = StructuredContextProvider()

    @property
    def config(self) -> ContextualEnrichmentConfig:
        """Return the active configuration."""
        return self._config

    @property
    def provider(self) -> BaseContextProvider:
        """Return the active context provider."""
        return self._provider

    def _validate_document(self, document: EnrichedDocument) -> None:
        """Validate input document type and project isolation across chunks."""
        if not isinstance(document, EnrichedDocument):
            raise InvalidContextualEnrichmentInputError(
                f"Expected EnrichedDocument, got '{type(document).__name__}'."
            )

        project_id = document.project_id
        for chunk in document.chunks:
            if not isinstance(chunk, EnrichedChunk):
                raise InvalidContextualEnrichmentInputError(
                    f"Expected EnrichedChunk in document chunks list, got '{type(chunk).__name__}'."
                )
            if chunk.project_id != project_id:
                raise ContextualEnrichmentProcessingError(
                    f"Project isolation violation: Chunk '{chunk.chunk_id}' has project_id "
                    f"'{chunk.project_id}', which does not match document project_id '{project_id}'."
                )

    def _create_disabled_result(
        self,
        document: EnrichedDocument,
        elapsed_ms: float,
    ) -> ContextuallyEnrichedDocument:
        """Construct ContextuallyEnrichedDocument when enrichment is disabled."""
        chunks: list[ContextuallyEnrichedChunk] = []
        for chunk in document.chunks:
            enriched_chunk = ContextuallyEnrichedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                project_id=chunk.project_id,
                content=chunk.content,  # Verbatim preservation
                index=chunk.index,
                document_version_id=chunk.document_version_id,
                section_path=chunk.section_path,
                heading=chunk.heading,
                heading_level=chunk.heading_level,
                parent_element_id=chunk.parent_element_id,
                parent_chunk_id=chunk.parent_chunk_id,
                source_element_ids=chunk.source_element_ids,
                element_types=chunk.element_types,
                metadata=dict(chunk.metadata),
                enriched_metadata=chunk.enriched_metadata,
                context_text=None,
                is_contextually_enriched=False,
                context_strategy=None,
            )
            chunks.append(enriched_chunk)

        report = ContextualEnrichmentReport(
            enabled=False,
            total_chunks=len(document.chunks),
            enriched_chunks_count=0,
            strategy_used="disabled",
            provider_name=None,
            elapsed_ms=elapsed_ms,
        )

        return ContextuallyEnrichedDocument(
            document_id=document.document_id,
            project_id=document.project_id,
            document_type=document.document_type,
            chunks=chunks,
            document_version_id=document.document_version_id,
            source_metadata=dict(document.source_metadata),
            parser_metadata=dict(document.parser_metadata),
            cleaning_report=document.cleaning_report,
            normalization_report=document.normalization_report,
            chunking_report=document.chunking_report,
            enrichment_report=document.enrichment_report,
            contextual_enrichment_report=report,
        )

    async def enrich(self, document: EnrichedDocument) -> ContextuallyEnrichedDocument:
        """Enrich an EnrichedDocument asynchronously at document scope.

        When CONTEXTUAL_ENRICHMENT_ENABLED is False, skips enrichment and preserves
        verbatim content with zero external overhead.
        When True, applies the configured provider strategy to generate situational context.

        Args:
            document: EnrichedDocument from Metadata Enrichment stage.

        Returns:
            ContextuallyEnrichedDocument containing ContextuallyEnrichedChunks.

        Raises:
            InvalidContextualEnrichmentInputError: If document is not an EnrichedDocument.
            ContextualEnrichmentProcessingError: If tenant isolation fails or processing errors.
            ContextualEnrichmentProviderError: If external model/provider fails.
        """
        self._validate_document(document)

        start_time = time.perf_counter()
        doc_id = document.document_id
        project_id = document.project_id
        total_chunks = len(document.chunks)

        # 1. Document-Level Disabled Scope Check
        if not self._config.enabled:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Contextual enrichment disabled via configuration; skipping document '%s' (project: '%s', chunks: %d)",
                doc_id,
                project_id,
                total_chunks,
            )
            return self._create_disabled_result(document, elapsed_ms)

        # 2. Handle empty document
        if total_chunks == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = ContextualEnrichmentReport(
                enabled=True,
                total_chunks=0,
                enriched_chunks_count=0,
                strategy_used=self._provider.provider_name,
                provider_name=self._provider.provider_name,
                elapsed_ms=elapsed_ms,
            )
            return ContextuallyEnrichedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                chunks=[],
                document_version_id=document.document_version_id,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                cleaning_report=document.cleaning_report,
                normalization_report=document.normalization_report,
                chunking_report=document.chunking_report,
                enrichment_report=document.enrichment_report,
                contextual_enrichment_report=report,
            )

        logger.info(
            "Starting contextual enrichment for document '%s' (project: '%s', chunks: %d, strategy: '%s')",
            doc_id,
            project_id,
            total_chunks,
            self._provider.provider_name,
        )

        try:
            # 3. Construct DocumentContext
            headings: list[str] = []
            for c in document.chunks:
                if c.heading and c.heading not in headings:
                    headings.append(c.heading)

            orig_filename = document.source_metadata.get("original_filename")
            storage_path = document.source_metadata.get("source_storage_path")

            context = DocumentContext(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                total_chunks=total_chunks,
                document_version_id=document.document_version_id,
                original_filename=orig_filename,
                source_storage_path=storage_path,
                source_metadata=dict(document.source_metadata),
                headings_hierarchy=tuple(headings),
            )

            # 4. Generate context descriptions via provider
            context_results = await self._provider.generate_context_batch(
                document.chunks,
                context,
            )

            # 5. Assemble ContextuallyEnrichedChunks strictly preserving original content
            enriched_chunks: list[ContextuallyEnrichedChunk] = []
            enriched_count = 0

            for chunk, ctx_text in zip(document.chunks, context_results, strict=True):
                is_enriched = bool(ctx_text and ctx_text.strip())
                if is_enriched:
                    enriched_count += 1

                combined_metadata = dict(chunk.metadata)
                if is_enriched:
                    combined_metadata["context_text"] = ctx_text
                    combined_metadata["is_contextually_enriched"] = True
                    combined_metadata["context_strategy"] = self._provider.provider_name

                enriched_chunk = ContextuallyEnrichedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    project_id=chunk.project_id,
                    content=chunk.content,  # Verbatim original content strictly preserved
                    index=chunk.index,
                    document_version_id=chunk.document_version_id,
                    section_path=chunk.section_path,
                    heading=chunk.heading,
                    heading_level=chunk.heading_level,
                    parent_element_id=chunk.parent_element_id,
                    parent_chunk_id=chunk.parent_chunk_id,
                    source_element_ids=chunk.source_element_ids,
                    element_types=chunk.element_types,
                    metadata=combined_metadata,
                    enriched_metadata=chunk.enriched_metadata,
                    context_text=ctx_text if is_enriched else None,
                    is_contextually_enriched=is_enriched,
                    context_strategy=self._provider.provider_name if is_enriched else None,
                )
                enriched_chunks.append(enriched_chunk)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            report = ContextualEnrichmentReport(
                enabled=True,
                total_chunks=total_chunks,
                enriched_chunks_count=enriched_count,
                strategy_used=self._provider.provider_name,
                provider_name=self._provider.provider_name,
                elapsed_ms=elapsed_ms,
            )

            logger.info(
                "Successfully enriched document '%s' in %.2fms (enriched chunks: %d/%d, strategy: '%s')",
                doc_id,
                elapsed_ms,
                enriched_count,
                total_chunks,
                self._provider.provider_name,
            )

            return ContextuallyEnrichedDocument(
                document_id=doc_id,
                project_id=project_id,
                document_type=document.document_type,
                chunks=enriched_chunks,
                document_version_id=document.document_version_id,
                source_metadata=dict(document.source_metadata),
                parser_metadata=dict(document.parser_metadata),
                cleaning_report=document.cleaning_report,
                normalization_report=document.normalization_report,
                chunking_report=document.chunking_report,
                enrichment_report=document.enrichment_report,
                contextual_enrichment_report=report,
            )

        except (
            InvalidContextualEnrichmentInputError,
            ContextualEnrichmentProcessingError,
            ContextualEnrichmentProviderError,
        ):
            raise
        except Exception as exc:
            logger.error(
                "Unexpected failure during contextual enrichment for document '%s': %s",
                doc_id,
                str(exc),
            )
            raise ContextualEnrichmentProcessingError(
                f"Unexpected failure enriching document '{doc_id}': {str(exc)}",
                original_error=exc,
            ) from exc

    def enrich_sync(self, document: EnrichedDocument) -> ContextuallyEnrichedDocument:
        """Enrich an EnrichedDocument synchronously.

        Useful for synchronous test fixtures or pipelines. When using async providers,
        safely bridges the async event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside a running event loop (e.g. jupyter or test harness)
            # Create a separate thread to run the coroutine synchronously
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.enrich(document))
                return future.result()
        else:
            return asyncio.run(self.enrich(document))

    async def enrich_batch(
        self,
        documents: list[EnrichedDocument],
    ) -> list[ContextuallyEnrichedDocument]:
        """Enrich multiple EnrichedDocuments concurrently while preserving tenant isolation.

        Args:
            documents: List of EnrichedDocument instances.

        Returns:
            List of ContextuallyEnrichedDocument results preserving input order.
        """
        tasks = [self.enrich(doc) for doc in documents]
        return await asyncio.gather(*tasks)


_default_contextual_enrichment_service: DocumentContextualEnrichmentService | None = None


def get_contextual_enrichment_service(
    config: ContextualEnrichmentConfig | None = None,
) -> DocumentContextualEnrichmentService:
    """Get or create DocumentContextualEnrichmentService instance."""
    global _default_contextual_enrichment_service
    if config is not None:
        return DocumentContextualEnrichmentService(config=config)
    if _default_contextual_enrichment_service is None:
        _default_contextual_enrichment_service = DocumentContextualEnrichmentService()
    return _default_contextual_enrichment_service


def reset_contextual_enrichment_service() -> None:
    """Reset the singleton instance for test isolation."""
    global _default_contextual_enrichment_service
    _default_contextual_enrichment_service = None
