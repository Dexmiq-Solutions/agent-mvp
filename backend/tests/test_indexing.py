"""Unit tests for the Indexing / Storage stage (DocumentIndexingService, models, identity)."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from acquisition.formats import DocumentType
from chunking.models import DocumentChunk
from embeddings.models import EmbeddingBatchResult, EmbeddingResult
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
    VectorInputValidationError,
    VectorStoreAuthenticationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorUpsertError,
)
from indexing import (
    DocumentIndexingService,
    IndexableRecord,
    IndexingBatchResult,
    IndexingConfig,
    IndexingReport,
    generate_point_id,
    get_indexing_service,
    reset_indexing_service,
)
from metadata_enrichment.models import ChunkMetadata, EnrichedChunk, EnrichedDocument
from storage.vector import BaseVectorStore
from storage.vector.models import VectorRecord


@pytest.fixture(autouse=True)
def cleanup_indexing_service():
    """Reset cached singleton service before and after each test."""
    reset_indexing_service()
    yield
    reset_indexing_service()


@pytest.fixture
def mock_indexing_settings():
    """Settings instance configured for indexing tests."""
    return Settings(
        QDRANT_URL="https://mock-cluster.qdrant.tech:6333",
        QDRANT_API_KEY="mock-api-key",
        QDRANT_COLLECTION_NAME="test_chunks",
        QDRANT_TIMEOUT=30,
        QDRANT_VECTOR_SIZE=4,
        QDRANT_BATCH_SIZE=3,
        QDRANT_DISTANCE="Cosine",
        QDRANT_MAX_RETRIES=2,
        QDRANT_RETRY_DELAY=0.01,  # Fast retries for testing
        QDRANT_RETRY_BACKOFF=1.5,
    )


@pytest.fixture
def mock_vector_store():
    """Mock BaseVectorStore instance."""
    store = AsyncMock(spec=BaseVectorStore)
    store.collection_name = "test_chunks"
    store.ensure_collection_exists = AsyncMock(return_value=False)
    store.upsert = AsyncMock(side_effect=lambda records: len(records))
    store.delete = AsyncMock(side_effect=lambda point_ids: len(point_ids))
    store.delete_by_filter = AsyncMock(return_value=True)
    return store


@pytest.fixture
def sample_chunk():
    """Sample DocumentChunk."""
    return DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-123",
        project_id="proj-abc",
        content="This is functional requirement chunk 1.",
        index=0,
        document_version_id="v1",
        section_path=("Requirements", "Functional"),
        heading="Functional Requirements",
    )


@pytest.fixture
def sample_enriched_chunk():
    """Sample EnrichedChunk with ChunkMetadata."""
    meta = ChunkMetadata(
        project_id="proj-abc",
        document_id="doc-123",
        chunk_id="chunk-001",
        document_version_id="v1",
        section_path=("Requirements", "Functional"),
        heading="Functional Requirements",
        heading_path_str="Requirements > Functional",
        chunk_index=0,
        total_chunks=1,
        relative_position=0.0,
        character_count=40,
        word_count=5,
        content_type="prose",
        has_code=False,
        has_table=False,
    )
    return EnrichedChunk(
        chunk_id="chunk-001",
        document_id="doc-123",
        project_id="proj-abc",
        content="This is functional requirement chunk 1.",
        index=0,
        document_version_id="v1",
        section_path=("Requirements", "Functional"),
        heading="Functional Requirements",
        enriched_metadata=meta,
    )


# ==============================================================================
# 1. Deterministic Point Identity & Model Tests
# ==============================================================================


def test_generate_point_id_determinism():
    """Verify generate_point_id produces identical RFC 4122 UUIDs for identical inputs."""
    id1 = generate_point_id("proj-1", "doc-1", "chunk-1", "v1")
    id2 = generate_point_id("proj-1", "doc-1", "chunk-1", "v1")

    assert id1 == id2
    assert len(id1) == 36
    assert id1.count("-") == 4


def test_generate_point_id_distinct_versions():
    """Verify different document versions map to distinct point IDs."""
    id_v1 = generate_point_id("proj-1", "doc-1", "chunk-1", "v1")
    id_v2 = generate_point_id("proj-1", "doc-1", "chunk-1", "v2")

    assert id_v1 != id_v2


def test_generate_point_id_distinct_chunks_and_projects():
    """Verify different chunks and projects map to distinct point IDs."""
    id_c1 = generate_point_id("proj-1", "doc-1", "chunk-1")
    id_c2 = generate_point_id("proj-1", "doc-1", "chunk-2")
    id_p2 = generate_point_id("proj-2", "doc-1", "chunk-1")

    assert id_c1 != id_c2
    assert id_c1 != id_p2


def test_generate_point_id_validation():
    """Verify generate_point_id raises InvalidIndexingInputError on empty or invalid inputs."""
    with pytest.raises(InvalidIndexingInputError):
        generate_point_id("", "doc-1", "chunk-1")

    with pytest.raises(InvalidIndexingInputError):
        generate_point_id("proj-1", "", "chunk-1")

    with pytest.raises(InvalidIndexingInputError):
        generate_point_id("proj-1", "doc-1", "")


def test_indexable_record_to_vector_record(sample_enriched_chunk):
    """Verify IndexableRecord converts to VectorRecord with correct ID, vector, and retrieval payload."""
    vector = [0.1, 0.2, 0.3, 0.4]
    rec = IndexableRecord.from_chunk_and_vector(
        chunk=sample_enriched_chunk,
        vector=vector,
        embedding_model="voyage-3-large",
        embedding_provider="voyage",
    )

    vec_record = rec.to_vector_record()
    assert isinstance(vec_record, VectorRecord)
    assert vec_record.id == rec.point_id
    assert vec_record.vector == vector

    payload = vec_record.payload
    assert payload.project_id == "proj-abc"
    assert payload.document_id == "doc-123"
    assert payload.chunk_id == "chunk-001"
    assert payload.document_version_id == "v1"

    # Verify retrieval metadata and embedding configuration attached
    p_dict = payload.to_dict()
    assert p_dict["embedding_model"] == "voyage-3-large"
    assert p_dict["embedding_provider"] == "voyage"
    assert p_dict["embedding_dimension"] == 4
    assert p_dict["heading_path_str"] == "Requirements > Functional"
    assert p_dict["content_type"] == "prose"


def test_indexing_config_validation():
    """Verify IndexingConfig validates parameters on instantiation."""
    with pytest.raises(IndexingConfigurationError):
        IndexingConfig(batch_size=0)

    with pytest.raises(IndexingConfigurationError):
        IndexingConfig(max_retries=-1)

    with pytest.raises(IndexingConfigurationError):
        IndexingConfig(retry_delay=-0.1)

    with pytest.raises(IndexingConfigurationError):
        IndexingConfig(retry_backoff=0.5)


# ==============================================================================
# 2. Input Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_validate_missing_ids(mock_vector_store, mock_indexing_settings):
    """Verify records with missing project, document, or chunk IDs are rejected."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="", document_id="doc-1", project_id="proj-1", vector=[0.1, 0.2, 0.3, 0.4])
        ])
    assert "chunk_id" in str(exc_info.value)

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="", project_id="proj-1", vector=[0.1, 0.2, 0.3, 0.4])
        ])
    assert "document_id" in str(exc_info.value)

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="doc-1", project_id="", vector=[0.1, 0.2, 0.3, 0.4])
        ])
    assert "project_id" in str(exc_info.value)


@pytest.mark.anyio
async def test_validate_vector_empty_and_non_numeric(mock_vector_store, mock_indexing_settings):
    """Verify empty or non-numeric/NaN vectors are rejected."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(InvalidIndexingInputError):
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[])
        ])

    with pytest.raises(InvalidIndexingInputError):
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.1, float("nan"), 0.3, 0.4])
        ])

    with pytest.raises(InvalidIndexingInputError):
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.1, float("inf"), 0.3, 0.4])
        ])


@pytest.mark.anyio
async def test_validate_vector_dimension_mismatch(mock_vector_store, mock_indexing_settings):
    """Verify dimension mismatch against configuration or between records is caught."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    # Config expects 4 dimensions; passing 3 should fail
    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.1, 0.2, 0.3])
        ])
    assert "dimension" in str(exc_info.value).lower()

    # Mismatched dimension between first and second record
    custom_config = IndexingConfig(expected_vector_size=None)
    service_no_size = DocumentIndexingService(
        vector_store=mock_vector_store, config=custom_config, settings=mock_indexing_settings
    )
    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service_no_size.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.1, 0.2]),
            IndexableRecord(chunk_id="c2", document_id="d1", project_id="p1", vector=[0.1, 0.2, 0.3]),
        ])
    assert "dimension mismatch" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_validate_project_isolation_violation(mock_vector_store, mock_indexing_settings):
    """Verify mismatched project IDs in the same indexing batch are rejected."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="proj-A", vector=[0.1, 0.2, 0.3, 0.4]),
            IndexableRecord(chunk_id="c2", document_id="d1", project_id="proj-B", vector=[0.5, 0.6, 0.7, 0.8]),
        ])
    assert "Project isolation violation" in str(exc_info.value)


@pytest.mark.anyio
async def test_validate_duplicate_chunk_ids(mock_vector_store, mock_indexing_settings):
    """Verify duplicate chunk IDs in the same batch are rejected."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_records([
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4]),
            IndexableRecord(chunk_id="c1", document_id="d1", project_id="p1", vector=[0.5, 0.6, 0.7, 0.8]),
        ])
    assert "Duplicate chunk_id" in str(exc_info.value)


# ==============================================================================
# 3. Batching & Idempotency Tests
# ==============================================================================


@pytest.mark.anyio
async def test_index_records_batching(mock_vector_store, mock_indexing_settings):
    """Verify records are split into configurable bounded batches and upserted."""
    # Batch size is 3 in mock_indexing_settings; 7 records should yield 3 batches (3, 3, 1)
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    records = [
        IndexableRecord(chunk_id=f"c-{i}", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
        for i in range(7)
    ]

    report = await service.index_records(records)

    assert report.total_attempted == 7
    assert report.total_indexed == 7
    assert report.total_failed == 0
    assert report.batches_count == 3
    assert report.failed_batches_count == 0
    assert report.is_success is True
    assert len(report.point_ids) == 7

    assert mock_vector_store.upsert.await_count == 3


@pytest.mark.anyio
async def test_reindexing_is_idempotent(mock_vector_store, mock_indexing_settings):
    """Verify re-indexing the same records twice generates identical point IDs and upserts idempotently."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    records = [
        IndexableRecord(chunk_id=f"c-{i}", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
        for i in range(2)
    ]

    report1 = await service.index_records(records)
    report2 = await service.index_records(records)

    # Identical point IDs generated on repeated runs
    assert report1.point_ids == report2.point_ids
    assert mock_vector_store.upsert.await_count == 2


# ==============================================================================
# 4. Document Indexing Integration Tests
# ==============================================================================


@pytest.mark.anyio
async def test_index_document_success(mock_vector_store, mock_indexing_settings, sample_enriched_chunk):
    """Verify index_document consumes an EnrichedDocument and EmbeddingBatchResult."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    doc = EnrichedDocument(
        document_id="doc-123",
        project_id="proj-abc",
        document_type=DocumentType.MARKDOWN,
        chunks=[sample_enriched_chunk],
    )
    embeddings = EmbeddingBatchResult(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        model="voyage-3-large",
        total_tokens=10,
    )

    report = await service.index_document(doc, embeddings)

    assert report.total_attempted == 1
    assert report.total_indexed == 1
    assert report.is_success is True
    assert report.model == "voyage-3-large"
    mock_vector_store.upsert.assert_awaited_once()


@pytest.mark.anyio
async def test_index_document_mismatched_counts(mock_vector_store, mock_indexing_settings, sample_enriched_chunk):
    """Verify index_document raises InvalidIndexingInputError when chunk and vector counts mismatch."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    doc = EnrichedDocument(
        document_id="doc-123",
        project_id="proj-abc",
        document_type=DocumentType.MARKDOWN,
        chunks=[sample_enriched_chunk],
    )
    # 2 vectors for 1 chunk -> error
    embeddings = EmbeddingBatchResult(
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        model="voyage-3-large",
    )

    with pytest.raises(InvalidIndexingInputError) as exc_info:
        await service.index_document(doc, embeddings)
    assert "Mismatched chunk and embedding counts" in str(exc_info.value)


# ==============================================================================
# 5. Retry Handling & Error Translation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_transient_failure_retry_and_recovery(mock_vector_store, mock_indexing_settings):
    """Verify transient VectorStoreConnectionError is retried and recovers on next attempt."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    # Fail on first call, succeed on second
    mock_vector_store.upsert.side_effect = [
        VectorStoreConnectionError("Transient network drop"),
        2,
    ]

    records = [
        IndexableRecord(chunk_id=f"c-{i}", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
        for i in range(2)
    ]

    report = await service.index_records(records)
    assert report.total_indexed == 2
    assert report.is_success is True
    assert mock_vector_store.upsert.await_count == 2


@pytest.mark.anyio
async def test_permanent_failure_no_excessive_retries(mock_vector_store, mock_indexing_settings):
    """Verify permanent VectorStoreAuthenticationError fails immediately without retry loop."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    mock_vector_store.upsert.side_effect = VectorStoreAuthenticationError("Invalid Qdrant API key")

    records = [
        IndexableRecord(chunk_id="c1", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
    ]

    with pytest.raises(IndexingPartialFailureError) as exc_info:
        await service.index_records(records, strict=True)

    report = exc_info.value.report
    assert report.total_failed == 1
    # Only 1 call because permanent failures are not retried
    assert mock_vector_store.upsert.await_count == 1


@pytest.mark.anyio
async def test_transient_failure_exhaustion(mock_vector_store, mock_indexing_settings):
    """Verify exhausting max retries fails cleanly and reports failed chunks."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    mock_vector_store.upsert.side_effect = VectorStoreConnectionError("Persistent network outage")

    records = [
        IndexableRecord(chunk_id="c1", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
    ]

    # strict=False returns report instead of raising
    report = await service.index_records(records, strict=False)

    assert report.total_failed == 1
    assert report.total_indexed == 0
    assert report.failed_chunk_ids == ("c1",)
    assert report.is_success is False
    # Initial attempt + 2 retries = 3 attempts
    assert mock_vector_store.upsert.await_count == 3


# ==============================================================================
# 6. Partial Failure Handling Tests
# ==============================================================================


@pytest.mark.anyio
async def test_partial_failure_reporting(mock_vector_store, mock_indexing_settings):
    """Verify multi-batch indexing with partial failure reports exact succeeded and failed batches."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    # Batch 1 (3 items) succeeds, Batch 2 (3 items) fails
    mock_vector_store.upsert.side_effect = [
        3,
        VectorStoreError("Disk full on vector node"),
        VectorStoreError("Disk full on vector node"),
        VectorStoreError("Disk full on vector node"),
    ]

    records = [
        IndexableRecord(chunk_id=f"c-{i}", document_id="doc-1", project_id="p1", vector=[0.1, 0.2, 0.3, 0.4])
        for i in range(6)
    ]

    with pytest.raises(IndexingPartialFailureError) as exc_info:
        await service.index_records(records, strict=True)

    report = exc_info.value.report
    assert report.total_attempted == 6
    assert report.total_indexed == 3
    assert report.total_failed == 3
    assert report.batches_count == 2
    assert report.failed_batches_count == 1
    assert report.failed_chunk_ids == ("c-3", "c-4", "c-5")
    assert report.is_success is False


# ==============================================================================
# 7. Lifecycle Operations Tests
# ==============================================================================


@pytest.mark.anyio
async def test_delete_document(mock_vector_store, mock_indexing_settings):
    """Verify delete_document enforces project isolation and calls delete_by_filter."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    res = await service.delete_document(project_id="proj-1", document_id="doc-1")
    assert res is True
    mock_vector_store.delete_by_filter.assert_awaited_once_with(project_id="proj-1", document_id="doc-1")


@pytest.mark.anyio
async def test_delete_document_version(mock_vector_store, mock_indexing_settings):
    """Verify delete_document_version scopes deletion to specific document version."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    res = await service.delete_document_version(
        project_id="proj-1", document_id="doc-1", document_version_id="v2"
    )
    assert res is True
    mock_vector_store.delete_by_filter.assert_awaited_once_with(
        project_id="proj-1", document_id="doc-1", document_version_id="v2"
    )


@pytest.mark.anyio
async def test_delete_chunks(mock_vector_store, mock_indexing_settings):
    """Verify delete_chunks delegates point IDs to vector store delete."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    deleted_count = await service.delete_chunks(project_id="proj-1", point_ids=["pt-1", "pt-2"])
    assert deleted_count == 2
    mock_vector_store.delete.assert_awaited_once_with(point_ids=["pt-1", "pt-2"])


# ==============================================================================
# 8. Service Factory & Settings Propagation Tests
# ==============================================================================


def test_get_indexing_service_singleton_and_reset(mock_indexing_settings):
    """Verify get_indexing_service returns cached singleton and reset clears it."""
    s1 = get_indexing_service(settings=mock_indexing_settings)
    s2 = get_indexing_service(settings=mock_indexing_settings)

    assert s1 is s2

    reset_indexing_service()
    s3 = get_indexing_service(settings=mock_indexing_settings)
    assert s1 is not s3


def test_indexing_config_from_settings_and_env(mock_indexing_settings):
    """Verify IndexingConfig loads correctly from Settings and env."""
    cfg = IndexingConfig.from_settings(mock_indexing_settings)
    assert cfg.batch_size == 3
    assert cfg.expected_vector_size == 4
    assert cfg.max_retries == 2
    assert cfg.distance == "Cosine"

    cfg_env = IndexingConfig.from_env()
    assert isinstance(cfg_env, IndexingConfig)


def test_rag_indexing_re_exports():
    """Verify src/rag/indexing provides backwards-compatible aliases."""
    from rag.indexing import (
        DocumentIndexingService as RagIndexingService,
        IndexableRecord as RagIndexableRecord,
        IndexingConfig as RagIndexingConfig,
        IndexingReport as RagIndexingReport,
        get_indexing_service as rag_get_indexing_service,
    )
    assert RagIndexingService is DocumentIndexingService
    assert RagIndexableRecord is IndexableRecord
    assert RagIndexingConfig is IndexingConfig
    assert RagIndexingReport is IndexingReport
    assert callable(rag_get_indexing_service)


# ==============================================================================
# 9. Contextual Enrichment & Raw Chunk Integration Tests
# ==============================================================================


@pytest.mark.anyio
async def test_index_document_with_contextually_enriched_document(
    mock_vector_store, mock_indexing_settings, sample_enriched_chunk
):
    """Verify index_document seamlessly handles ContextuallyEnrichedDocument."""
    from contextual_enrichment.models import (
        ContextuallyEnrichedChunk,
        ContextuallyEnrichedDocument,
    )

    ctx_chunk = ContextuallyEnrichedChunk(
        chunk_id=sample_enriched_chunk.chunk_id,
        document_id=sample_enriched_chunk.document_id,
        project_id=sample_enriched_chunk.project_id,
        content=sample_enriched_chunk.content,
        index=sample_enriched_chunk.index,
        document_version_id=sample_enriched_chunk.document_version_id,
        section_path=sample_enriched_chunk.section_path,
        heading=sample_enriched_chunk.heading,
        enriched_metadata=sample_enriched_chunk.enriched_metadata,
        context_text="Context: User Authentication Specification",
        is_contextually_enriched=True,
    )

    ctx_doc = ContextuallyEnrichedDocument(
        document_id="doc-123",
        project_id="proj-abc",
        document_type=DocumentType.MARKDOWN,
        chunks=[ctx_chunk],
    )

    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)
    report = await service.index_document(ctx_doc, [[0.1, 0.2, 0.3, 0.4]])

    assert report.total_attempted == 1
    assert report.total_indexed == 1
    assert report.is_success is True


@pytest.mark.anyio
async def test_index_raw_document_chunk_without_enriched_metadata(
    mock_vector_store, mock_indexing_settings, sample_chunk
):
    """Verify IndexableRecord handles a baseline DocumentChunk lacking ChunkMetadata."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    rec = IndexableRecord.from_chunk_and_vector(sample_chunk, [0.1, 0.2, 0.3, 0.4])
    report = await service.index_records([rec])

    assert report.total_indexed == 1
    assert report.is_success is True


def test_indexable_record_point_id_override():
    """Verify point_id_override takes precedence if supplied."""
    rec = IndexableRecord(
        chunk_id="c1",
        document_id="d1",
        project_id="p1",
        vector=[0.1, 0.2, 0.3, 0.4],
        point_id_override="custom-uuid-override",
    )
    assert rec.point_id == "custom-uuid-override"


# ==============================================================================
# 10. Audit Report & Serialization Tests
# ==============================================================================


def test_indexing_report_serialization():
    """Verify IndexingReport to_dict and __repr__ do not leak sensitive vectors or text."""
    batch = IndexingBatchResult(
        batch_index=1,
        points_count=2,
        point_ids=("id-1", "id-2"),
        chunk_ids=("c-1", "c-2"),
        status="success",
    )
    report = IndexingReport(
        total_attempted=2,
        total_indexed=2,
        total_failed=0,
        batches_count=1,
        failed_batches_count=0,
        point_ids=("id-1", "id-2"),
        collection_name="test_collection",
        model="voyage-3-large",
        elapsed_ms=12.34,
        batches=(batch,),
    )

    d = report.to_dict()
    assert d["total_attempted"] == 2
    assert d["total_indexed"] == 2
    assert d["is_success"] is True
    assert len(d["batches"]) == 1
    assert d["batches"][0]["points_count"] == 2

    rep_str = repr(report)
    assert "IndexingReport(" in rep_str
    assert "test_collection" in rep_str


# ==============================================================================
# 11. Collection Verification Error Handling Tests
# ==============================================================================


@pytest.mark.anyio
async def test_ensure_collection_configuration_mismatch(mock_vector_store, mock_indexing_settings):
    """Verify ensure_collection translates CollectionConfigurationError to IndexingConfigurationError."""
    mock_vector_store.ensure_collection_exists.side_effect = CollectionConfigurationError(
        "Dimension mismatch: existing is 768, requested is 4"
    )
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(IndexingConfigurationError) as exc_info:
        await service.ensure_collection(4)
    assert "mismatch" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_ensure_collection_connection_error(mock_vector_store, mock_indexing_settings):
    """Verify ensure_collection translates VectorStoreConnectionError to IndexingConnectionError."""
    mock_vector_store.ensure_collection_exists.side_effect = VectorStoreConnectionError(
        "Connection refused to Qdrant cluster"
    )
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(IndexingConnectionError) as exc_info:
        await service.ensure_collection(4)
    assert "Failed to connect to Qdrant" in str(exc_info.value)


@pytest.mark.anyio
async def test_delete_document_validation_errors(mock_vector_store, mock_indexing_settings):
    """Verify delete methods reject empty or non-string IDs."""
    service = DocumentIndexingService(vector_store=mock_vector_store, settings=mock_indexing_settings)

    with pytest.raises(InvalidIndexingInputError):
        await service.delete_document("", "doc-1")

    with pytest.raises(InvalidIndexingInputError):
        await service.delete_document("p1", "")

    with pytest.raises(InvalidIndexingInputError):
        await service.delete_document_version("p1", "doc-1", "")

    with pytest.raises(InvalidIndexingInputError):
        await service.delete_chunks("p1", [])
