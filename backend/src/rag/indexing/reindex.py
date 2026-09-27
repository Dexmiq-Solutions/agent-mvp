"""Lightweight re-indexing mechanism for migrating dense vectors to Cohere Embed v4."""

import asyncio
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import Settings, get_settings
from db.session import get_async_session_maker
from exceptions.indexing import IndexingOperationError, InvalidIndexingInputError
from models.chunk import ChunkModel
from models.document import DocumentVersionModel, DocumentVersionStatus
from observability.logging import get_logger
from rag.chunking.models import DocumentChunk
from rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from rag.indexing.models import IndexableRecord, IndexingReport
from rag.indexing.service import DocumentIndexingService, get_indexing_service
from rag.retrieval.keyword.encoder import get_sparse_encoder
from storage.vector.models import SparseVector

logger = get_logger(__name__)


async def reindex_document_version(
    project_id: str,
    document_id: str,
    document_version_id: str,
    session: Optional[AsyncSession] = None,
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    indexing_service: Optional[DocumentIndexingService] = None,
    settings: Optional[Settings] = None,
) -> IndexingReport:
    """Re-index an existing document version using the active embedding provider.

    Reads existing stored chunks from PostgreSQL, computes fresh dense embeddings
    using the active provider (e.g. Cohere Embed v4), removes prior vector points
    from Qdrant to prevent vector mixing, and upserts fresh 1024-dimensional vectors.

    Args:
        project_id: Project/tenant identifier.
        document_id: Document identifier.
        document_version_id: Specific version identifier to re-index.
        session: Optional active AsyncSession.
        session_maker: Optional session maker.
        embedding_provider: Active BaseEmbeddingProvider (defaults to configured Cohere provider).
        indexing_service: DocumentIndexingService instance.
        settings: Application settings.

    Returns:
        IndexingReport: Execution report from Qdrant vector indexing.
    """
    app_settings = settings or get_settings()
    active_provider = embedding_provider or get_embedding_provider(settings=app_settings)
    active_indexer = indexing_service or get_indexing_service(settings=app_settings)

    # 1. Fetch chunks from PostgreSQL
    async def _fetch_chunks(sess: AsyncSession) -> Sequence[ChunkModel]:
        stmt = (
            select(ChunkModel)
            .where(
                ChunkModel.project_id == project_id,
                ChunkModel.document_id == document_id,
                ChunkModel.document_version_id == document_version_id,
            )
            .order_by(ChunkModel.chunk_index.asc())
        )
        res = await sess.execute(stmt)
        return res.scalars().all()

    if session is not None:
        chunk_models = await _fetch_chunks(session)
    else:
        maker = session_maker or get_async_session_maker(settings=app_settings)
        async with maker() as sess:
            chunk_models = await _fetch_chunks(sess)

    if not chunk_models:
        raise InvalidIndexingInputError(
            f"No chunks found in PostgreSQL for version '{document_version_id}' "
            f"(doc: '{document_id}', project: '{project_id}')."
        )

    logger.info(
        "Re-indexing %d chunks for version '%s' using provider '%s' (model: '%s')",
        len(chunk_models),
        document_version_id,
        getattr(active_provider, "provider_name", "unknown"),
        active_provider.model_name,
    )

    # 2. Extract representation text (contextual content preferred if enriched, else content)
    representation_texts = [
        c.contextual_content if c.contextual_content and c.contextual_content.strip() else c.content
        for c in chunk_models
    ]

    # 3. Generate fresh dense embeddings (batched with input_type='document')
    embedding_result = await active_provider.embed_batch(
        texts=representation_texts,
        input_type="document",
    )

    # 4. Generate sparse representations if sparse indexing enabled
    sparse_vectors: Optional[list[SparseVector]] = None
    if active_indexer.config.sparse_indexing_enabled:
        encoder = get_sparse_encoder(settings=app_settings)
        sparse_vectors = encoder.encode_documents(representation_texts)

    # 5. Build IndexableRecords
    provider_name = getattr(active_provider, "provider_name", "cohere")
    records: list[IndexableRecord] = []
    for idx, (chunk_m, vec) in enumerate(zip(chunk_models, embedding_result.embeddings)):
        doc_chunk = DocumentChunk(
            chunk_id=chunk_m.chunk_id,
            document_id=chunk_m.document_id,
            project_id=chunk_m.project_id,
            content=chunk_m.content,
            index=chunk_m.chunk_index,
            document_version_id=chunk_m.document_version_id,
            section_path=tuple(chunk_m.section_path or []),
            heading=chunk_m.heading,
            heading_level=chunk_m.heading_level,
            parent_element_id=chunk_m.parent_element_id,
            parent_chunk_id=chunk_m.parent_chunk_id,
            metadata=dict(chunk_m.chunk_metadata or {}),
        )

        sparse_vec = sparse_vectors[idx] if sparse_vectors is not None else None
        rec = IndexableRecord.from_chunk_and_vector(
            chunk=doc_chunk,
            vector=vec,
            embedding_model=active_provider.model_name,
            embedding_provider=provider_name,
            sparse_vector=sparse_vec,
            sparse_encoder_strategy=active_indexer.config.sparse_encoder_strategy if sparse_vec else None,
            sparse_encoder_version=active_indexer.config.sparse_encoder_version if sparse_vec else None,
        )
        records.append(rec)

    # 6. Ensure old points for this version are removed before upserting new vectors
    try:
        await active_indexer.delete_document_version(
            project_id=project_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )
    except Exception as exc:
        logger.warning(
            "Could not delete old points for version '%s' before re-indexing: %s",
            document_version_id,
            exc,
        )

    # 7. Upsert new vectors into Qdrant
    report = await active_indexer.index_records(records, strict=True)
    logger.info(
        "Successfully re-indexed version '%s': %d points indexed in Qdrant",
        document_version_id,
        report.total_indexed,
    )
    return report


async def reindex_project_documents(
    project_id: str,
    session_maker: Optional[async_sessionmaker[AsyncSession]] = None,
    settings: Optional[Settings] = None,
) -> list[IndexingReport]:
    """Re-index all READY document versions belonging to a given project."""
    app_settings = settings or get_settings()
    maker = session_maker or get_async_session_maker(settings=app_settings)

    async with maker() as sess:
        stmt = (
            select(DocumentVersionModel)
            .where(
                DocumentVersionModel.project_id == project_id,
                DocumentVersionModel.status == DocumentVersionStatus.READY.value,
            )
        )
        res = await sess.execute(stmt)
        versions = res.scalars().all()

    reports: list[IndexingReport] = []
    for ver in versions:
        report = await reindex_document_version(
            project_id=ver.project_id,
            document_id=ver.document_id,
            document_version_id=ver.id,
            session_maker=maker,
            settings=app_settings,
        )
        reports.append(report)

    return reports
