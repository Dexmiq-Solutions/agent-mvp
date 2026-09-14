"""Unit tests for the Context Assembly stage in the retrieval pipeline."""

from typing import Any, Optional
import pytest

from exceptions.retrieval import (
    ContextAssemblyError,
    ContextAssemblyValidationError,
    RetrievalError,
)
from models.chunk import ChunkModel
from rag.retrieval import (
    AssembledContext as RagAssembledContext,
    AssembledContextItem as RagAssembledContextItem,
    ContextAssemblyConfig as RagContextAssemblyConfig,
    ContextAssemblyError as RagContextAssemblyError,
    ContextAssemblyService as RagContextAssemblyService,
    ContextAssemblyValidationError as RagContextAssemblyValidationError,
    ContextItem as RagContextItem,
    HydratedCandidate as RagHydratedCandidate,
    RetrievalContext as RagRetrievalContext,
    StructuredRetrievalContext as RagStructuredRetrievalContext,
    assemble_context as rag_assemble_context,
    assemble_context_async as rag_assemble_context_async,
    get_context_assembly_service as rag_get_context_assembly_service,
    reset_context_assembly_service as rag_reset_context_assembly_service,
)
from retrieval import (
    AssembledContext,
    AssembledContextItem,
    ContextAssemblyConfig,
    ContextAssemblyError,
    ContextAssemblyService,
    ContextAssemblyValidationError,
    ContextItem,
    HydratedCandidate,
    ProcessedQuery,
    RetrievalContext,
    RetrievalQuerySet,
    StructuredRetrievalContext,
    assemble_context,
    assemble_context_async,
    get_context_assembly_service,
    reset_context_assembly_service,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Reset context assembly singletons before and after each test."""
    reset_context_assembly_service()
    rag_reset_context_assembly_service()
    yield
    reset_context_assembly_service()
    rag_reset_context_assembly_service()


def _make_hydrated_candidate(
    chunk_id: str,
    content: str = "Authoritative stored chunk text",
    rank: int = 1,
    project_id: str = "proj-1",
    document_id: str = "doc-1",
    document_version_id: str | None = "v-1",
    contextual_content: str | None = None,
    rerank_score: float | None = 0.95,
    fusion_score: float | None = 0.035,
    dense_rank: int | None = 1,
    sparse_rank: int | None = 2,
    dense_score: float | None = 0.88,
    sparse_score: float | None = 15.4,
    chunk_index: int = 0,
    heading: str | None = "Introduction",
    heading_level: int | None = 1,
    section_path: tuple[str, ...] = ("Overview", "Introduction"),
    parent_element_id: str | None = "elem-0",
    parent_chunk_id: str | None = None,
    element_types: tuple[str, ...] = ("heading", "paragraph"),
    metadata: dict[str, Any] | None = None,
) -> HydratedCandidate:
    """Helper to construct a test HydratedCandidate instance."""
    return HydratedCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        content=content,
        rank=rank,
        rerank_score=rerank_score,
        fusion_score=fusion_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=dense_score,
        sparse_score=sparse_score,
        document_version_id=document_version_id,
        contextual_content=contextual_content,
        chunk_index=chunk_index,
        heading=heading,
        heading_level=heading_level,
        section_path=section_path,
        parent_element_id=parent_element_id,
        parent_chunk_id=parent_chunk_id,
        element_types=element_types,
        metadata=dict(metadata or {"source": "guide.pdf", "author": "Acme Corp"}),
    )


# ==============================================================================
# 1. Structured Transformation & Contract Tests
# ==============================================================================


def test_transform_hydrated_candidates_into_structured_context():
    """Verify hydrated candidates are cleanly transformed into structured AssembledContext."""
    service = ContextAssemblyService()
    c1 = _make_hydrated_candidate("c-1", content="Text for chunk 1", rank=1)
    c2 = _make_hydrated_candidate("c-2", content="Text for chunk 2", rank=2)

    context = service.assemble([c1, c2], project_id="proj-1")

    assert isinstance(context, AssembledContext)
    assert len(context) == 2
    assert not context.is_empty
    assert context.project_id == "proj-1"

    item1 = context[0]
    assert isinstance(item1, AssembledContextItem)
    assert item1.chunk_id == "c-1"
    assert item1.content == "Text for chunk 1"
    assert item1.text == "Text for chunk 1"
    assert item1.rank == 1

    item2 = context[1]
    assert item2.chunk_id == "c-2"
    assert item2.content == "Text for chunk 2"
    assert item2.text == "Text for chunk 2"
    assert item2.rank == 2


def test_assembled_context_iteration_and_indexing():
    """Verify AssembledContext container methods (__iter__, __len__, __getitem__)."""
    service = ContextAssemblyService()
    c1 = _make_hydrated_candidate("c-1", rank=1)
    c2 = _make_hydrated_candidate("c-2", rank=2)

    context = service.assemble([c1, c2], project_id="proj-1")

    assert len(context) == 2
    items = list(context)
    assert len(items) == 2
    assert items[0].chunk_id == "c-1"
    assert items[1].chunk_id == "c-2"
    assert context[0].chunk_id == "c-1"
    assert context[1].chunk_id == "c-2"
    assert context.chunk_ids == ("c-1", "c-2")
    assert context.document_ids == ("doc-1",)


# ==============================================================================
# 2. Retrieval Order Preservation Tests
# ==============================================================================


def test_preserves_exact_retrieval_order():
    """Verify the stage strictly preserves input ordering [A, C, B]."""
    service = ContextAssemblyService()
    cand_a = _make_hydrated_candidate("chunk-A", rank=1)
    cand_c = _make_hydrated_candidate("chunk-C", rank=2)
    cand_b = _make_hydrated_candidate("chunk-B", rank=3)

    context = service.assemble([cand_a, cand_c, cand_b], project_id="proj-1")

    ordered_ids = [item.chunk_id for item in context]
    assert ordered_ids == ["chunk-A", "chunk-C", "chunk-B"]
    assert context[0].rank == 1
    assert context[1].rank == 2
    assert context[2].rank == 3


def test_does_not_group_by_document_which_would_alter_relevance_order():
    """Verify chunks from different documents are NOT clustered by document."""
    service = ContextAssemblyService()
    # Interleaved documents: doc-1, doc-2, doc-1
    c1 = _make_hydrated_candidate("c-1", document_id="doc-1", rank=1)
    c2 = _make_hydrated_candidate("c-2", document_id="doc-2", rank=2)
    c3 = _make_hydrated_candidate("c-3", document_id="doc-1", rank=3)

    context = service.assemble([c1, c2, c3], project_id="proj-1")

    assert [item.document_id for item in context] == ["doc-1", "doc-2", "doc-1"]
    assert [item.chunk_id for item in context] == ["c-1", "c-2", "c-3"]
    assert context.document_ids == ("doc-1", "doc-2")


# ==============================================================================
# 3. Identity Coordinates Preservation Tests
# ==============================================================================


def test_preserves_all_identity_coordinates():
    """Verify chunk_id, document_id, document_version_id, project_id, chunk_index are intact."""
    cand = _make_hydrated_candidate(
        chunk_id="chunk-xyz-123",
        document_id="doc-abc-456",
        document_version_id="ver-789",
        project_id="tenant-prod-1",
        chunk_index=42,
    )
    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="tenant-prod-1")

    item = context[0]
    assert item.chunk_id == "chunk-xyz-123"
    assert item.document_id == "doc-abc-456"
    assert item.document_version_id == "ver-789"
    assert item.project_id == "tenant-prod-1"
    assert item.chunk_index == 42
    assert item.index == 42


# ==============================================================================
# 4. Provenance & Hierarchy Preservation Tests
# ==============================================================================


def test_preserves_hierarchy_coordinates_and_provenance():
    """Verify structural hierarchy and first-stage/second-stage retrieval provenance are retained."""
    cand = _make_hydrated_candidate(
        chunk_id="c-hier",
        heading="Security Architecture",
        heading_level=2,
        section_path=("System Overview", "Architecture", "Security Architecture"),
        parent_element_id="elem-parent-99",
        parent_chunk_id="c-parent-1",
        element_types=("heading", "list", "code"),
        rerank_score=0.987,
        fusion_score=0.045,
        dense_rank=1,
        sparse_rank=3,
        dense_score=0.92,
        sparse_score=18.4,
        metadata={"source": "security_whitepaper.pdf", "tier": "critical"},
    )
    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="proj-1")

    item = context[0]
    assert item.heading == "Security Architecture"
    assert item.heading_level == 2
    assert item.section_path == ("System Overview", "Architecture", "Security Architecture")
    assert item.parent_element_id == "elem-parent-99"
    assert item.parent_chunk_id == "c-parent-1"
    assert item.element_types == ("heading", "list", "code")
    assert item.rerank_score == 0.987
    assert item.fusion_score == 0.045
    assert item.score == 0.987
    assert item.dense_rank == 1
    assert item.sparse_rank == 3
    assert item.dense_score == 0.92
    assert item.sparse_score == 18.4
    assert item.source == "security_whitepaper.pdf"
    assert item.metadata["tier"] == "critical"


# ==============================================================================
# 5. Stored Content Representation Tests (No Duplicate Concatenation)
# ==============================================================================


def test_uses_contextual_content_when_present_and_enabled():
    """When contextual_content is present and enabled, text representation uses it."""
    cand = _make_hydrated_candidate(
        chunk_id="c-ctx",
        content="Verbatim chunk content about OAuth2 tokens.",
        contextual_content="Document: API Spec > Section: Auth\n\nVerbatim chunk content about OAuth2 tokens.",
    )
    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="proj-1", use_contextual_enrichment=True)

    item = context[0]
    assert item.content == "Verbatim chunk content about OAuth2 tokens."
    assert item.contextual_content == "Document: API Spec > Section: Auth\n\nVerbatim chunk content about OAuth2 tokens."
    assert item.text == "Document: API Spec > Section: Auth\n\nVerbatim chunk content about OAuth2 tokens."
    # Ensure NO duplicate concatenation occurred
    assert item.text != f"{item.contextual_content}\n\n{item.content}"


def test_uses_verbatim_content_when_contextual_enrichment_absent():
    """When contextual_content is None, text representation falls back to verbatim content."""
    cand = _make_hydrated_candidate(
        chunk_id="c-plain",
        content="Verbatim chunk content only.",
        contextual_content=None,
    )
    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="proj-1")

    item = context[0]
    assert item.content == "Verbatim chunk content only."
    assert item.contextual_content is None
    assert item.text == "Verbatim chunk content only."


def test_uses_verbatim_content_when_contextual_enrichment_disabled():
    """When use_contextual_enrichment=False, text representation uses verbatim content."""
    cand = _make_hydrated_candidate(
        chunk_id="c-ctx-off",
        content="Raw chunk content.",
        contextual_content="Context Header\n\nRaw chunk content.",
    )
    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="proj-1", use_contextual_enrichment=False)

    item = context[0]
    assert item.content == "Raw chunk content."
    assert item.contextual_content == "Context Header\n\nRaw chunk content."
    assert item.text == "Raw chunk content."


# ==============================================================================
# 6. Project Isolation Tests
# ==============================================================================


def test_project_isolation_discards_cross_tenant_candidates_by_default():
    """Default mode defensively discards candidates with mismatched project_id."""
    service = ContextAssemblyService()
    c_valid = _make_hydrated_candidate("c-valid", project_id="proj-1")
    c_leak = _make_hydrated_candidate("c-leak", project_id="proj-2")

    context = service.assemble([c_valid, c_leak], project_id="proj-1")

    assert len(context) == 1
    assert context[0].chunk_id == "c-valid"
    assert context.project_id == "proj-1"
    assert context.metadata["discarded_candidates"] == 1


def test_strict_project_validation_raises_on_cross_tenant_candidate():
    """Strict mode raises ContextAssemblyValidationError on cross-tenant candidates."""
    service = ContextAssemblyService()
    c_valid = _make_hydrated_candidate("c-valid", project_id="proj-1")
    c_leak = _make_hydrated_candidate("c-leak", project_id="proj-2")

    with pytest.raises(ContextAssemblyValidationError) as exc_info:
        service.assemble([c_valid, c_leak], project_id="proj-1", strict_project_validation=True)

    assert "violates project boundary" in str(exc_info.value)


def test_project_id_inferred_from_first_valid_candidate():
    """When project_id is None, it is inferred from the candidate set."""
    service = ContextAssemblyService()
    c1 = _make_hydrated_candidate("c-1", project_id="inferred-proj")
    c2 = _make_hydrated_candidate("c-2", project_id="inferred-proj")

    context = service.assemble([c1, c2])

    assert context.project_id == "inferred-proj"
    assert len(context) == 2


def test_raises_when_no_project_id_can_be_inferred_from_non_empty_candidates():
    """Raises ContextAssemblyValidationError when project_id is omitted and candidates lack it."""
    service = ContextAssemblyService()
    # Mock candidate without project_id
    cand = ChunkModel(chunk_id="c-1", content="foo", project_id=None, document_id="d-1")

    with pytest.raises(ContextAssemblyValidationError) as exc_info:
        service.assemble([cand])

    assert "project_id must be provided explicitly" in str(exc_info.value)


# ==============================================================================
# 7. Empty Input Short-Circuit Tests
# ==============================================================================


def test_empty_candidates_returns_empty_context():
    """Empty candidate sequence returns empty AssembledContext without errors."""
    service = ContextAssemblyService()
    context = service.assemble([], project_id="proj-1")

    assert isinstance(context, AssembledContext)
    assert len(context) == 0
    assert context.is_empty
    assert context.items == ()
    assert context.chunk_ids == ()
    assert context.document_ids == ()
    assert context.project_id == "proj-1"
    assert context.metadata["candidates_in"] == 0
    assert context.metadata["items_out"] == 0


def test_empty_candidates_with_unspecified_project_id():
    """Empty candidate sequence without project_id returns empty AssembledContext."""
    service = ContextAssemblyService()
    context = service.assemble([])

    assert len(context) == 0
    assert context.is_empty
    assert context.project_id == ""


# ==============================================================================
# 8. Missing / Incomplete Hydrated Results Tests
# ==============================================================================


def test_discards_null_or_malformed_candidates_without_fabricating():
    """Null or malformed candidate entries are safely discarded; no fabrication."""
    service = ContextAssemblyService()
    valid_cand = _make_hydrated_candidate("c-1", content="Genuine content")

    # Sequence containing None and object missing chunk_id
    context = service.assemble(
        [None, {"some": "dict"}, valid_cand, "not-a-candidate"],  # type: ignore
        project_id="proj-1",
    )

    assert len(context) == 1
    assert context[0].chunk_id == "c-1"
    assert context[0].content == "Genuine content"
    assert context.metadata["discarded_candidates"] == 3


def test_discards_empty_chunk_id():
    """Candidate with whitespace or empty chunk_id is discarded."""
    service = ContextAssemblyService()
    cand_empty = _make_hydrated_candidate("   ", content="Some text")
    cand_valid = _make_hydrated_candidate("c-valid", content="Valid text")

    context = service.assemble([cand_empty, cand_valid], project_id="proj-1")

    assert len(context) == 1
    assert context[0].chunk_id == "c-valid"


# ==============================================================================
# 9. Result / Context Limits Tests
# ==============================================================================


def test_bounds_context_to_max_context_items():
    """Configured max_context_items bounds the number of assembled items."""
    service = ContextAssemblyService()
    candidates = [_make_hydrated_candidate(f"c-{i}", rank=i) for i in range(1, 11)]

    context = service.assemble(candidates, project_id="proj-1", max_context_items=3)

    assert len(context) == 3
    assert [item.chunk_id for item in context] == ["c-1", "c-2", "c-3"]


def test_invalid_max_context_items_raises_validation_error():
    """max_context_items <= 0 raises ContextAssemblyValidationError."""
    service = ContextAssemblyService()
    cand = _make_hydrated_candidate("c-1")

    with pytest.raises(ContextAssemblyValidationError) as exc_info:
        service.assemble([cand], project_id="proj-1", max_context_items=0)

    assert "max_context_items must be a positive integer" in str(exc_info.value)


# ==============================================================================
# 10. Deterministic Output Tests
# ==============================================================================


def test_deterministic_output_across_repeated_runs():
    """Same input and configuration strictly produces identical structured output."""
    service = ContextAssemblyService()
    candidates = [
        _make_hydrated_candidate("c-1", rank=1, content="Alpha text"),
        _make_hydrated_candidate("c-2", rank=2, content="Beta text"),
        _make_hydrated_candidate("c-3", rank=3, content="Gamma text"),
    ]

    res1 = service.assemble(candidates, project_id="proj-1")
    res2 = service.assemble(candidates, project_id="proj-1")

    assert res1.chunk_ids == res2.chunk_ids
    assert [item.to_dict() for item in res1] == [item.to_dict() for item in res2]


# ==============================================================================
# 11. No Unintended Semantic Rewriting
# ==============================================================================


def test_no_unintended_semantic_rewriting():
    """Verbatim chunk text is strictly unchanged without LLM or heuristic mutation."""
    verbatim = "Technical text with code: def process(x): return x * 2; and symbols #@!&"
    cand = _make_hydrated_candidate("c-verbatim", content=verbatim)

    service = ContextAssemblyService()
    context = service.assemble([cand], project_id="proj-1")

    assert context[0].content == verbatim
    assert context[0].text == verbatim


# ==============================================================================
# 12. Deduplication Tests
# ==============================================================================


def test_deduplication_preserves_highest_ranking_occurrence():
    """Duplicate chunk_ids preserve the first occurrence in ranking order."""
    service = ContextAssemblyService()
    c1 = _make_hydrated_candidate("c-dup", content="First occurrence (rank 1)", rank=1)
    c2 = _make_hydrated_candidate("c-other", content="Other chunk", rank=2)
    c3 = _make_hydrated_candidate("c-dup", content="Second occurrence (rank 3)", rank=3)

    context = service.assemble([c1, c2, c3], project_id="proj-1")

    assert len(context) == 2
    assert [item.chunk_id for item in context] == ["c-dup", "c-other"]
    assert context[0].content == "First occurrence (rank 1)"
    assert context[0].rank == 1


# ==============================================================================
# 13. Query Attachment & Extraction Tests
# ==============================================================================


def test_query_attachment_from_various_representations():
    """Query is preserved as string from str, ProcessedQuery, and RetrievalQuerySet."""
    service = ContextAssemblyService()
    cand = _make_hydrated_candidate("c-1")

    # From raw string
    ctx1 = service.assemble([cand], project_id="p-1", query="  rag query  ")
    assert ctx1.query == "rag query"

    # From ProcessedQuery
    pq = ProcessedQuery(original_query="raw", processed_query="normalized query")
    ctx2 = service.assemble([cand], project_id="p-1", query=pq)
    assert ctx2.query == "normalized query"

    # From RetrievalQuerySet
    rqs = RetrievalQuerySet(original_query="primary query")
    ctx3 = service.assemble([cand], project_id="p-1", query=rqs)
    assert ctx3.query == "primary query"


# ==============================================================================
# 14. Serialization & Convenience APIs Tests
# ==============================================================================


def test_serialization_to_dict():
    """Verify to_dict on AssembledContext and AssembledContextItem produces clean dictionaries."""
    service = ContextAssemblyService()
    cand = _make_hydrated_candidate(
        chunk_id="c-serial",
        content="Content to serialize",
        metadata={"source": "api_doc.pdf"},
    )
    context = service.assemble([cand], project_id="proj-1", query="test query")

    ctx_dict = context.to_dict()
    assert ctx_dict["project_id"] == "proj-1"
    assert ctx_dict["query"] == "test query"
    assert ctx_dict["item_count"] == 1
    assert ctx_dict["chunk_ids"] == ["c-serial"]
    assert ctx_dict["document_ids"] == ["doc-1"]
    assert len(ctx_dict["items"]) == 1

    item_dict = ctx_dict["items"][0]
    assert item_dict["chunk_id"] == "c-serial"
    assert item_dict["content"] == "Content to serialize"
    assert item_dict["source"] == "api_doc.pdf"


def test_functional_convenience_entrypoint():
    """Verify assemble_context functional entrypoint coordinates with active service."""
    cand = _make_hydrated_candidate("c-func")
    context = assemble_context([cand], project_id="proj-1")

    assert isinstance(context, AssembledContext)
    assert len(context) == 1
    assert context[0].chunk_id == "c-func"


@pytest.mark.asyncio
async def test_async_functional_convenience_entrypoint():
    """Verify assemble_context_async convenience wrapper."""
    cand = _make_hydrated_candidate("c-async")
    context = await assemble_context_async([cand], project_id="proj-1")

    assert isinstance(context, AssembledContext)
    assert len(context) == 1
    assert context[0].chunk_id == "c-async"


def test_type_aliases_and_rag_parity():
    """Verify type aliases and RAG package re-exports match 1-to-1."""
    assert ContextItem is AssembledContextItem
    assert RetrievalContext is AssembledContext
    assert StructuredRetrievalContext is AssembledContext

    # Parity with rag.retrieval
    assert RagContextItem is ContextItem
    assert RagAssembledContextItem is AssembledContextItem
    assert RagAssembledContext is AssembledContext
    assert RagRetrievalContext is RetrievalContext
    assert RagStructuredRetrievalContext is StructuredRetrievalContext
    assert RagContextAssemblyConfig is ContextAssemblyConfig
    assert RagContextAssemblyService is ContextAssemblyService
    assert RagContextAssemblyError is ContextAssemblyError
    assert RagContextAssemblyValidationError is ContextAssemblyValidationError
    assert rag_assemble_context is assemble_context
    assert rag_assemble_context_async is assemble_context_async
    assert rag_get_context_assembly_service is get_context_assembly_service
    assert rag_reset_context_assembly_service is reset_context_assembly_service
