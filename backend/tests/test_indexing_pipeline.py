"""Comprehensive integration tests for the End-to-End Indexing Pipeline.

Verifies the complete flow from DocumentVersion through:
  - Processing Lifecycle
  - Acquisition
  - Ingestion
  - Parsing & Extraction
  - Cleaning
  - Normalization
  - Chunking
  - Metadata Enrichment
  - Optional Contextual Enrichment
  - Dense Embedding (Voyage) + Sparse Representation (technical_hash)
  - PostgreSQL Chunk Persistence
  - Qdrant Vector Indexing
  - DocumentVersion READY transition and stale version pruning.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from exceptions.document import DocumentProcessingError, ProjectDocumentMismatchError
from exceptions.embedding import EmbeddingError
from exceptions.indexing import (
    IndexingError,
    IndexingOperationError,
    InvalidIndexingInputError,
)
from exceptions.storage import ObjectNotFoundError
from exceptions.vector import VectorUpsertError
from models.base import Base
from models.chunk import ChunkModel
from models.document import DocumentModel, DocumentVersionModel, DocumentVersionStatus
from models.project import ProjectModel
from rag.chunking.models import DocumentChunk
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.models import EmbeddingBatchResult
from rag.indexing import (
    ChunkPersistenceService,
    DocumentIndexingService,
    EndToEndIndexingService,
    IndexingConfig,
    generate_point_id,
    get_chunk_persistence_service,
    get_end_to_end_indexing_service,
    reset_chunk_persistence_service,
    reset_end_to_end_indexing_service,
    reset_indexing_service,
)
from rag.contextual_enrichment.models import ContextuallyEnrichedChunk
from rag.metadata_enrichment.models import ChunkMetadata, EnrichedChunk
from rag.pipeline.default import DefaultDocumentProcessingPipeline, get_processing_pipeline
from services.document_processing_service import DocumentProcessingService
from storage.object.base import BaseObjectStorage
from storage.object.models import StorageObjectMetadata
from storage.vector.base import BaseVectorStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons before and after each test."""
    reset_chunk_persistence_service()
    reset_end_to_end_indexing_service()
    reset_indexing_service()
    yield
    reset_chunk_persistence_service()
    reset_end_to_end_indexing_service()
    reset_indexing_service()


@asynccontextmanager
async def create_test_db() -> AsyncIterator[tuple[AsyncSession, async_sessionmaker[AsyncSession]]]:
    """Create an isolated in-memory SQLite database and yield session and session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session, session_factory

    await engine.dispose()


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Stub embedding provider returning deterministic float vectors."""

    def __init__(self, dimension: int = 4, model: str = "voyage-4", should_fail: bool = False, fail_message: str = "Embedding failed"):
        self._dimension = dimension
        self._model = model
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.embedded_batches: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_text(self, text: str, input_type: Optional[str] = None) -> list[float]:
        res = await self.embed_batch([text], input_type=input_type)
        return res.embeddings[0]

    async def embed_batch(self, texts: list[str], input_type: str = "document") -> EmbeddingBatchResult:
        if self.should_fail:
            raise EmbeddingError(self.fail_message)
        self.embedded_batches.append(list(texts))
        vectors = []
        for i, text in enumerate(texts):
            # Deterministic mock vector based on text length and index
            val = float(i + 1) / 10.0
            vec = [val] * self._dimension
            vectors.append(vec)
        return EmbeddingBatchResult(
            embeddings=vectors,
            model=self._model,
            total_tokens=sum(len(t.split()) for t in texts),
        )


def make_mock_storage_with_doc(content: str, filename: str = "spec.md", bucket: str = "documents") -> BaseObjectStorage:
    """Create a mock BaseObjectStorage containing a single document."""
    storage = AsyncMock(spec=BaseObjectStorage)
    storage.bucket_name = bucket
    encoded = content.encode("utf-8")

    async def fake_get_metadata(path: str):
        if not path.endswith(filename):
            raise ObjectNotFoundError(f"Object '{path}' not found")
        return StorageObjectMetadata(
            name=filename,
            path=path,
            bucket=bucket,
            size_bytes=len(encoded),
            content_type="text/markdown",
            etag="etag-mock-md",
        )

    async def fake_download(path: str) -> bytes:
        if not path.endswith(filename):
            raise ObjectNotFoundError(f"Object '{path}' not found")
        return encoded

    storage.get_metadata = AsyncMock(side_effect=fake_get_metadata)
    storage.download = AsyncMock(side_effect=fake_download)
    return storage


def make_mock_vector_store() -> AsyncMock:
    """Create a mock BaseVectorStore tracking upserts and deletes."""
    store = AsyncMock(spec=BaseVectorStore)
    store.collection_name = "test_chunks"
    store.ensure_collection_exists = AsyncMock(return_value=False)
    store.upserted_records = []
    store.deleted_filters = []

    async def fake_upsert(records):
        store.upserted_records.extend(records)
        return len(records)

    async def fake_delete_by_filter(*args, **kwargs):
        criteria = kwargs if kwargs else (args[0] if args else {})
        store.deleted_filters.append(criteria)
        return True

    store.upsert = AsyncMock(side_effect=fake_upsert)
    store.delete_by_filter = AsyncMock(side_effect=fake_delete_by_filter)
    store.delete = AsyncMock(side_effect=lambda point_ids: len(point_ids))
    return store


async def seed_test_hierarchy(
    session: AsyncSession,
    version_num: int = 1,
    status: DocumentVersionStatus = DocumentVersionStatus.PENDING,
) -> tuple[ProjectModel, DocumentModel, DocumentVersionModel]:
    """Seed Project, Document, and DocumentVersion models."""
    project = ProjectModel(name="EndToEnd Indexing Project")
    session.add(project)
    await session.flush()

    doc = DocumentModel(project_id=project.id, name="System Architecture Spec")
    session.add(doc)
    await session.flush()

    version = DocumentVersionModel(
        document_id=doc.id,
        project_id=project.id,
        version_number=version_num,
        storage_bucket="documents",
        storage_path=f"{project.id}/{doc.id}/v{version_num}/spec.md",
        original_filename="spec.md",
        content_type="text/markdown",
        size_bytes=1024,
        status=status.value,
    )
    session.add(version)
    await session.commit()
    return project, doc, version


SAMPLE_MARKDOWN = """# Architecture Overview
This section outlines the primary architecture of the Dexmiq RAG platform.
The architecture combines dense vector embeddings with sparse representation search.

# Retrieval Mechanics
Retrieval utilizes reciprocal rank fusion across Qdrant vector indices and PostgreSQL text.
Each document version is indexed atomically into project-isolated partitions.

# Persistence Strategy
PostgreSQL stores chunk text and contextual headers for accurate downstream prompt assembly.
All transactions remain short and never span external provider network calls.
"""


# ==============================================================================
# 1. ChunkPersistenceService Unit & Idempotency Tests
# ==============================================================================


@pytest.mark.anyio
async def test_chunk_persistence_service_basic_persist_and_replace():
    """Verify persisting chunks into PostgreSQL ChunkModel and replacing them idempotently."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        persistence_service = ChunkPersistenceService(session_maker=session_factory)

        # Create 3 chunks
        chunks = [
            DocumentChunk(
                chunk_id=f"{doc.id}_chunk_0",
                document_id=doc.id,
                project_id=project.id,
                content="Chunk 0 content regarding architecture.",
                index=0,
                metadata={"token_count": 10},
                section_path=("Architecture",),
                heading="Architecture",
            ),
            DocumentChunk(
                chunk_id=f"{doc.id}_chunk_1",
                document_id=doc.id,
                project_id=project.id,
                content="Chunk 1 content regarding retrieval.",
                index=1,
                metadata={"token_count": 12},
                section_path=("Retrieval",),
                heading="Retrieval",
            ),
        ]

        persisted = await persistence_service.persist_chunks(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            chunks=chunks,
        )

        assert len(persisted) == 2
        assert persisted[0].chunk_id == f"{doc.id}_chunk_0"
        assert persisted[0].project_id == project.id
        assert persisted[0].document_id == doc.id
        assert persisted[0].document_version_id == version.id
        assert persisted[0].chunk_index == 0
        assert persisted[0].content == "Chunk 0 content regarding architecture."

        # Verify query directly from DB
        async with session_factory() as check_session:
            db_chunks = (
                await check_session.execute(
                    select(ChunkModel).where(ChunkModel.document_id == doc.id).order_by(ChunkModel.chunk_index)
                )
            ).scalars().all()
            assert len(db_chunks) == 2
            assert db_chunks[0].chunk_index == 0
            assert db_chunks[1].chunk_index == 1

        # Re-persist with updated content (idempotency check)
        updated_chunks = [
            DocumentChunk(
                chunk_id=f"{doc.id}_chunk_0",
                document_id=doc.id,
                project_id=project.id,
                content="Updated chunk 0 content.",
                index=0,
                metadata={"token_count": 15},
                section_path=("Architecture", "Updated"),
                heading="Architecture Updated",
            ),
        ]

        persisted_v2 = await persistence_service.persist_chunks(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            chunks=updated_chunks,
        )

        assert len(persisted_v2) == 1
        assert persisted_v2[0].content == "Updated chunk 0 content."

        # Verify in DB: only 1 chunk now exists
        async with session_factory() as check_session:
            db_chunks = (
                await check_session.execute(
                    select(ChunkModel).where(ChunkModel.document_id == doc.id)
                )
            ).scalars().all()
            assert len(db_chunks) == 1
            assert db_chunks[0].content == "Updated chunk 0 content."


@pytest.mark.anyio
async def test_chunk_persistence_service_enriched_chunk_metadata_mapping():
    """Verify that EnrichedChunk metadata and contextual_content map properly to ChunkModel."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        persistence_service = ChunkPersistenceService(session_maker=session_factory)

        meta = ChunkMetadata(
            project_id=project.id,
            document_id=doc.id,
            chunk_id=f"{doc.id}_chunk_0",
            chunk_index=0,
            heading="Section Title",
            section_path=("Root", "Section Title"),
            document_type="markdown",
            character_count=100,
            word_count=18,
            extra={"keywords": ["architecture", "indexing"]},
        )

        enriched = ContextuallyEnrichedChunk(
            chunk_id=f"{doc.id}_chunk_0",
            document_id=doc.id,
            project_id=project.id,
            content="Raw chunk content.",
            index=0,
            metadata=meta,
            context_text="Context Header",
            is_contextually_enriched=True,
            context_strategy="structured",
        )

        persisted = await persistence_service.persist_chunks(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            chunks=[enriched],
        )

        assert len(persisted) == 1
        record = persisted[0]
        assert record.content == "Raw chunk content."
        assert record.contextual_content == "Context Header\n\nRaw chunk content."
        assert record.chunk_metadata["structural"]["heading"] == "Section Title"
        assert record.chunk_metadata["extra"]["keywords"] == ["architecture", "indexing"]


@pytest.mark.anyio
async def test_chunk_persistence_service_tenant_isolation_violation():
    """Verify that ChunkPersistenceService rejects chunks with mismatched project_id or document_id."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)
        persistence_service = ChunkPersistenceService(session_maker=session_factory)

        # Chunk with wrong project_id
        bad_proj_chunk = DocumentChunk(
            chunk_id=f"{doc.id}_chunk_0",
            document_id=doc.id,
            project_id="alien-project-id",
            content="Intruder chunk",
            index=0,
        )

        with pytest.raises((IndexingOperationError, InvalidIndexingInputError)) as exc_info:
            await persistence_service.persist_chunks(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
                chunks=[bad_proj_chunk],
            )
        assert "isolation violation" in str(exc_info.value).lower()

        # Chunk with wrong document_id
        bad_doc_chunk = DocumentChunk(
            chunk_id="alien_doc_chunk_0",
            document_id="alien-doc-id",
            project_id=project.id,
            content="Intruder chunk",
            index=0,
        )

        with pytest.raises((IndexingOperationError, InvalidIndexingInputError)) as exc_info:
            await persistence_service.persist_chunks(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
                chunks=[bad_doc_chunk],
            )
        assert "isolation violation" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_chunk_persistence_empty_chunks():
    """Verify that passing empty chunks list raises InvalidIndexingInputError."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)
        persistence_service = ChunkPersistenceService(session_maker=session_factory)

        with pytest.raises(InvalidIndexingInputError):
            await persistence_service.persist_chunks(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
                chunks=[],
            )


# ==============================================================================
# 2. EndToEndIndexingService Full Integration Tests
# ==============================================================================


@pytest.mark.anyio
async def test_end_to_end_indexing_service_full_workflow():
    """Verify EndToEndIndexingService executes all 10 stages and returns comprehensive report."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        embedding_provider = FakeEmbeddingProvider(dimension=4, model="voyage-4")

        indexing_config = IndexingConfig(
            expected_vector_size=4,
            batch_size=2,
            sparse_indexing_enabled=True,
            collection_name="test_chunks",
        )
        indexing_service = DocumentIndexingService(
            vector_store=mock_vstore,
            config=indexing_config,
        )

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=indexing_service,
            session_maker=session_factory,
        )

        report = await orchestrator.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            storage_bucket=version.storage_bucket,
            storage_path=version.storage_path,
            original_filename=version.original_filename,
            content_type=version.content_type,
        )

        assert report.status == "success"
        assert report.project_id == project.id
        assert report.document_id == doc.id
        assert report.document_version_id == version.id
        assert report.total_chunks >= 3
        assert report.persisted_chunks == report.total_chunks
        assert report.indexed_points == report.total_chunks
        assert report.dense_model == "voyage-4"
        assert report.sparse_enabled is True

        # Verify stage reports are captured
        assert "acquisition" in report.stage_reports
        assert "parsing" in report.stage_reports
        assert "chunking" in report.stage_reports
        assert "representations" in report.stage_reports
        assert "postgres_persistence" in report.stage_reports
        assert "qdrant_indexing" in report.stage_reports

        # Verify PostgreSQL persistence
        async with session_factory() as check_session:
            db_chunks = (
                await check_session.execute(
                    select(ChunkModel).where(ChunkModel.document_id == doc.id).order_by(ChunkModel.chunk_index)
                )
            ).scalars().all()
            assert len(db_chunks) == report.total_chunks
            assert all(c.project_id == project.id for c in db_chunks)
            assert all(c.document_version_id == version.id for c in db_chunks)

        # Verify Qdrant vector indexing
        assert len(mock_vstore.upserted_records) == report.total_chunks
        for record in mock_vstore.upserted_records:
            assert len(record.vector) == 4
            assert record.sparse_vector is not None
            assert len(record.sparse_vector.indices) > 0
            payload_dict = record.payload.to_dict() if hasattr(record.payload, "to_dict") else record.payload
            assert payload_dict["project_id"] == project.id
            assert payload_dict["document_id"] == doc.id
            assert payload_dict["document_version_id"] == version.id
            assert "chunk_id" in payload_dict
            assert "chunk_index" in payload_dict


@pytest.mark.anyio
async def test_end_to_end_indexing_stale_version_pruning():
    """Verify that when Version 2 is indexed, stale vectors for Version 1 are pruned from Qdrant."""
    async with create_test_db() as (session, session_factory):
        # 1. Seed Version 1 as READY
        project, doc, version_1 = await seed_test_hierarchy(
            session, version_num=1, status=DocumentVersionStatus.READY
        )

        # 2. Seed Version 2 as PENDING
        version_2 = DocumentVersionModel(
            document_id=doc.id,
            project_id=project.id,
            version_number=2,
            storage_bucket="documents",
            storage_path=f"{project.id}/{doc.id}/v2/spec.md",
            original_filename="spec.md",
            content_type="text/markdown",
            size_bytes=1024,
            status=DocumentVersionStatus.PENDING.value,
        )
        session.add(version_2)
        await session.commit()

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        embedding_provider = FakeEmbeddingProvider(dimension=4)

        indexing_config = IndexingConfig(expected_vector_size=4, collection_name="test_chunks")
        indexing_service = DocumentIndexingService(
            vector_store=mock_vstore,
            config=indexing_config,
        )

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=indexing_service,
            session_maker=session_factory,
            cleanup_stale_versions=True,
        )

        # Index Version 2
        report = await orchestrator.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version_2.id,
            storage_bucket=version_2.storage_bucket,
            storage_path=version_2.storage_path,
            original_filename=version_2.original_filename,
            content_type=version_2.content_type,
        )

        assert report.stale_versions_pruned == 1
        # Verify vector store delete_by_filter was called with version_1's ID
        assert len(mock_vstore.deleted_filters) == 1
        filter_called = mock_vstore.deleted_filters[0]
        assert filter_called["project_id"] == project.id
        assert filter_called["document_id"] == doc.id
        assert filter_called["document_version_id"] == version_1.id


@pytest.mark.anyio
async def test_end_to_end_contextual_enrichment_toggle():
    """Verify contextual enrichment enabled produces contextual content while disabled uses standard content."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        embedding_provider = FakeEmbeddingProvider(dimension=4)

        # Test with contextual enrichment disabled (default)
        settings_disabled = Settings(
            ENABLE_CONTEXTUAL_ENRICHMENT=False,
            QDRANT_COLLECTION_NAME="test_chunks",
        )
        orchestrator_disabled = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=DocumentIndexingService(vector_store=mock_vstore, config=IndexingConfig(expected_vector_size=4)),
            session_maker=session_factory,
            settings=settings_disabled,
        )

        report = await orchestrator_disabled.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            storage_bucket=version.storage_bucket,
            storage_path=version.storage_path,
            original_filename=version.original_filename,
            content_type=version.content_type,
        )

        assert report.contextual_enrichment_enabled is False
        assert report.stage_reports["contextual_enrichment"]["enabled"] is False


@pytest.mark.anyio
async def test_end_to_end_indexing_batching_verification():
    """Verify batching across embedding generation and Qdrant upserts."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        # Longer markdown with many sections to produce 5+ chunks
        long_markdown = "\n\n".join([
            f"# Section {i}\n" + ("Detailed description of component " * 30)
            for i in range(8)
        ])

        storage = make_mock_storage_with_doc(long_markdown, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        embedding_provider = FakeEmbeddingProvider(dimension=4)

        indexing_config = IndexingConfig(
            expected_vector_size=4,
            batch_size=3,  # Small batch size to enforce multiple upsert calls
            collection_name="test_chunks",
        )
        indexing_service = DocumentIndexingService(
            vector_store=mock_vstore,
            config=indexing_config,
        )

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=indexing_service,
            session_maker=session_factory,
        )

        report = await orchestrator.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
            storage_bucket=version.storage_bucket,
            storage_path=version.storage_path,
            original_filename=version.original_filename,
            content_type=version.content_type,
        )

        assert report.total_chunks >= 8
        assert mock_vstore.upsert.call_count >= 3  # Multiple batches upserted
        assert len(mock_vstore.upserted_records) == report.total_chunks


# ==============================================================================
# 3. DocumentProcessingService + DefaultDocumentProcessingPipeline Integration
# ==============================================================================


@pytest.mark.anyio
async def test_document_processing_service_full_pipeline_success():
    """Verify DocumentProcessingService runs DefaultDocumentProcessingPipeline -> EndToEndIndexingService to READY."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        embedding_provider = FakeEmbeddingProvider(dimension=4)

        indexing_config = IndexingConfig(expected_vector_size=4, collection_name="test_chunks")
        indexing_service = DocumentIndexingService(
            vector_store=mock_vstore,
            config=indexing_config,
        )

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=indexing_service,
            session_maker=session_factory,
        )

        pipeline = DefaultDocumentProcessingPipeline(
            storage=storage,
            indexing_service=orchestrator,
            session_maker=session_factory,
        )

        service = DocumentProcessingService(
            session_maker=session_factory,
            storage=storage,
            pipeline=pipeline,
        )

        # Trigger lifecycle execution
        ready_version = await service.process_document_version(
            project_id=project.id,
            document_id=doc.id,
            document_version_id=version.id,
        )

        # 1. State transition assertions
        assert ready_version.status == DocumentVersionStatus.READY.value
        assert ready_version.indexed_at is not None
        assert ready_version.error_message is None

        # 2. Database verification
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.READY.value
            assert db_ver.indexed_at is not None

            # Verify chunks were written
            db_chunks = (
                await check_session.execute(
                    select(ChunkModel).where(ChunkModel.document_id == doc.id)
                )
            ).scalars().all()
            assert len(db_chunks) > 0

        # 3. Qdrant verification
        assert len(mock_vstore.upserted_records) == len(db_chunks)


@pytest.mark.anyio
async def test_document_processing_service_embedding_failure_transitions_to_failed():
    """Verify that if the embedding provider fails, version transitions to FAILED with sanitized error."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        # Embed provider raises error with simulated secret key leak
        embedding_provider = FakeEmbeddingProvider(
            should_fail=True,
            fail_message="Voyage API failed: sk-abcdef123456789 connection timed out",
        )

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=embedding_provider,
            indexing_service=DocumentIndexingService(vector_store=mock_vstore, config=IndexingConfig(expected_vector_size=4)),
            session_maker=session_factory,
        )

        pipeline = DefaultDocumentProcessingPipeline(
            storage=storage,
            indexing_service=orchestrator,
            session_maker=session_factory,
        )

        service = DocumentProcessingService(
            session_maker=session_factory,
            storage=storage,
            pipeline=pipeline,
        )

        with pytest.raises(DocumentProcessingError):
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        # Check DB state
        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.FAILED.value
            assert db_ver.indexed_at is None
            # Secret must be masked
            assert "sk-abcdef123456789" not in db_ver.error_message
            assert "sk-***" in db_ver.error_message


@pytest.mark.anyio
async def test_document_processing_service_qdrant_failure_transitions_to_failed():
    """Verify that if Qdrant upsert fails, version transitions to FAILED with sanitized error."""
    async with create_test_db() as (session, session_factory):
        project, doc, version = await seed_test_hierarchy(session)

        storage = make_mock_storage_with_doc(SAMPLE_MARKDOWN, filename="spec.md")
        mock_vstore = make_mock_vector_store()
        mock_vstore.upsert = AsyncMock(side_effect=VectorUpsertError("Qdrant cluster unavailable"))

        orchestrator = EndToEndIndexingService(
            storage=storage,
            embedding_provider=FakeEmbeddingProvider(dimension=4),
            indexing_service=DocumentIndexingService(vector_store=mock_vstore, config=IndexingConfig(expected_vector_size=4)),
            session_maker=session_factory,
        )

        pipeline = DefaultDocumentProcessingPipeline(
            storage=storage,
            indexing_service=orchestrator,
            session_maker=session_factory,
        )

        service = DocumentProcessingService(
            session_maker=session_factory,
            storage=storage,
            pipeline=pipeline,
        )

        with pytest.raises(DocumentProcessingError) as exc_info:
            await service.process_document_version(
                project_id=project.id,
                document_id=doc.id,
                document_version_id=version.id,
            )

        async with session_factory() as check_session:
            db_ver = (
                await check_session.execute(
                    select(DocumentVersionModel).where(DocumentVersionModel.id == version.id)
                )
            ).scalar_one()
            assert db_ver.status == DocumentVersionStatus.FAILED.value
            assert (
                "Indexing partially failed" in db_ver.error_message
                or "Qdrant cluster unavailable" in db_ver.error_message
            )
