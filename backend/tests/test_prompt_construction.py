"""Unit tests for the Prompt Construction stage of the RAG generation pipeline."""

from typing import Any, Optional
import pytest

from exceptions.generation import (
    GenerationError,
    PromptConstructionError,
    PromptConstructionValidationError,
)
from generation import (
    DEFAULT_SYSTEM_INSTRUCTION,
    ConstructedPrompt,
    GenerationPrompt,
    PromptConstructionConfig,
    PromptConstructionService,
    StructuredPrompt,
    construct_prompt,
    construct_prompt_async,
    get_prompt_construction_service,
    reset_prompt_construction_service,
)
from generation.prompt import (
    ConstructedPrompt as PromptConstructedPrompt,
    PromptConstructionConfig as PromptPromptConstructionConfig,
    PromptConstructionService as PromptPromptConstructionService,
    construct_prompt as prompt_construct_prompt,
    construct_prompt_async as prompt_construct_prompt_async,
    get_prompt_construction_service as prompt_get_prompt_construction_service,
    reset_prompt_construction_service as prompt_reset_prompt_construction_service,
)
from rag.generation import (
    ConstructedPrompt as RagConstructedPrompt,
    GenerationPrompt as RagGenerationPrompt,
    PromptConstructionConfig as RagPromptConstructionConfig,
    PromptConstructionError as RagPromptConstructionError,
    PromptConstructionService as RagPromptConstructionService,
    PromptConstructionValidationError as RagPromptConstructionValidationError,
    StructuredPrompt as RagStructuredPrompt,
    construct_prompt as rag_construct_prompt,
    construct_prompt_async as rag_construct_prompt_async,
    get_prompt_construction_service as rag_get_prompt_construction_service,
    reset_prompt_construction_service as rag_reset_prompt_construction_service,
)
from retrieval.models import (
    AssembledContext,
    AssembledContextItem,
    PolicyDecision,
    ProcessedQuery,
    RetrievalQuerySet,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Reset prompt construction singletons before and after each test."""
    reset_prompt_construction_service()
    prompt_reset_prompt_construction_service()
    rag_reset_prompt_construction_service()
    yield
    reset_prompt_construction_service()
    prompt_reset_prompt_construction_service()
    rag_reset_prompt_construction_service()


def _make_context_item(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    project_id: str = "proj-1",
    content: str = "Verbatim chunk text content.",
    rank: int = 1,
    text: Optional[str] = None,
    document_version_id: Optional[str] = "v-1",
    heading: Optional[str] = "Overview",
    section_path: tuple[str, ...] = ("Architecture", "Overview"),
    source: Optional[str] = "architecture_spec.pdf",
    metadata: Optional[dict[str, Any]] = None,
) -> AssembledContextItem:
    """Helper to construct an AssembledContextItem for prompt tests."""
    resolved_text = text if text is not None else content
    cand_metadata = dict(metadata or {})
    if source and "source" not in cand_metadata:
        cand_metadata["source"] = source

    return AssembledContextItem(
        chunk_id=chunk_id,
        document_id=document_id,
        project_id=project_id,
        content=content,
        rank=rank,
        text=resolved_text,
        document_version_id=document_version_id,
        heading=heading,
        section_path=section_path,
        metadata=cand_metadata,
    )


def _make_assembled_context(
    items: Optional[list[AssembledContextItem]] = None,
    project_id: str = "proj-1",
    query: Optional[str] = "What is the architecture?",
) -> AssembledContext:
    """Helper to construct an AssembledContext."""
    item_tuple = tuple(items or [])
    return AssembledContext(
        items=item_tuple,
        project_id=project_id,
        query=query,
    )


# ---------------------------------------------------------------------------
# 1. Basic Construction & Logical Separation
# ---------------------------------------------------------------------------


def test_basic_prompt_construction():
    """Verify valid structured prompt produced from original query and final context."""
    item = _make_context_item(
        chunk_id="chk-100",
        document_id="doc-100",
        content="The system uses PostgreSQL for relational storage.",
    )
    context = _make_assembled_context(items=[item])
    query = "Where is data stored?"

    result = construct_prompt(query=query, context=context)

    assert isinstance(result, ConstructedPrompt)
    assert result.user_query == "Where is data stored?"
    assert result.project_id == "proj-1"
    assert result.context_items_count == 1

    # Logical separation:
    # 1. System instruction is populated
    assert result.system_instruction == DEFAULT_SYSTEM_INSTRUCTION
    # 2. Context text contains chunk content and identifiers
    assert "PostgreSQL for relational storage" in result.context_text
    assert "chk-100" in result.context_text
    assert "doc-100" in result.context_text
    # 3. User prompt cleanly combines context header, context, query header, query
    assert "## Retrieved Context" in result.user_prompt
    assert "## User Query" in result.user_prompt
    assert "Where is data stored?" in result.user_prompt

    # Provider-independent messages
    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == DEFAULT_SYSTEM_INSTRUCTION
    assert result.messages[1]["role"] == "user"
    assert result.messages[1]["content"] == result.user_prompt


def test_custom_system_instruction_override():
    """Verify custom system instruction can be passed at call time."""
    context = _make_assembled_context(items=[_make_context_item()])
    custom_instruction = "You are an expert software architect. Be succinct."

    result = construct_prompt(
        query="Explain the caching layer.",
        context=context,
        system_instruction=custom_instruction,
    )

    assert result.system_instruction == custom_instruction
    assert result.messages[0]["content"] == custom_instruction


# ---------------------------------------------------------------------------
# 2. Context Integrity, Ordering, and Multiple Items
# ---------------------------------------------------------------------------


def test_multiple_context_items_ordering_preserved():
    """Verify Final Context ordering is preserved strictly without reordering."""
    item_a = _make_context_item(chunk_id="chunk-A", content="First ranked chunk content.", rank=1)
    item_b = _make_context_item(chunk_id="chunk-B", content="Second ranked chunk content.", rank=2)
    item_c = _make_context_item(chunk_id="chunk-C", content="Third ranked chunk content.", rank=3)

    # Retrieval order: B, A, C (must NOT be sorted by rank or document)
    context = _make_assembled_context(items=[item_b, item_a, item_c])

    result = construct_prompt(query="Test order", context=context)

    assert result.context_items_count == 3
    pos_b = result.context_text.find("chunk-B")
    pos_a = result.context_text.find("chunk-A")
    pos_c = result.context_text.find("chunk-C")

    assert pos_b < pos_a < pos_c, "Context items must strictly preserve retrieval order"
    assert "[Context Item 1]" in result.context_text
    assert "[Context Item 2]" in result.context_text
    assert "[Context Item 3]" in result.context_text


def test_context_content_verbatim_integrity():
    """Verify context text is faithfully rendered without summarizing, rewriting, or altering."""
    exact_text = (
        "Specialized paragraph with verbatim symbols: & < > % $ # @ ! \n"
        "And second line with exact whitespace:    four spaces indent."
    )
    item = _make_context_item(content=exact_text, text=exact_text)
    context = _make_assembled_context(items=[item])

    result = construct_prompt(query="Verify integrity", context=context)

    assert exact_text in result.context_text
    assert exact_text in result.user_prompt


# ---------------------------------------------------------------------------
# 3. Original Query Preservation
# ---------------------------------------------------------------------------


def test_preserves_original_query_from_string():
    """Original query string is preserved unchanged."""
    raw_query = "What is the RTO and RPO for disaster recovery?"
    context = _make_assembled_context(items=[_make_context_item()])

    result = construct_prompt(query=raw_query, context=context)
    assert result.user_query == raw_query


def test_preserves_original_query_from_processed_query():
    """When ProcessedQuery is passed, original_query is strictly used (not processed_query)."""
    pq = ProcessedQuery(
        original_query="What is the latency SLA???",
        processed_query="what is the latency sla",
    )
    context = _make_assembled_context(items=[_make_context_item()])

    result = construct_prompt(query=pq, context=context)
    assert result.user_query == "What is the latency SLA???"
    assert "What is the latency SLA???" in result.user_prompt


def test_preserves_original_query_from_retrieval_query_set():
    """When RetrievalQuerySet is passed, original_query is strictly used (not transformed_query)."""
    rqs = RetrievalQuerySet(
        original_query="How do we scale workers?",
        transformed_query="scaling workers in distributed architecture autoscaling rules",
        is_transformed=True,
        strategy_used="llm_rewrite",
    )
    context = _make_assembled_context(items=[_make_context_item()])

    result = construct_prompt(query=rqs, context=context)
    assert result.user_query == "How do we scale workers?"
    assert "scaling workers in distributed architecture autoscaling rules" not in result.user_prompt


# ---------------------------------------------------------------------------
# 4. Metadata and Provenance Representation
# ---------------------------------------------------------------------------


def test_metadata_and_provenance_included():
    """Verify document_id, chunk_id, document_version_id, source, and section path are rendered."""
    item = _make_context_item(
        chunk_id="chk-xyz",
        document_id="doc-abc",
        document_version_id="ver-42",
        source="system_design.docx",
        section_path=("System Overview", "Database Tier"),
        content="Primary PostgreSQL node with two read replicas.",
    )
    context = _make_assembled_context(items=[item])

    result = construct_prompt(query="DB setup", context=context)

    assert "Document ID: doc-abc" in result.context_text
    assert "Document Version: ver-42" in result.context_text
    assert "Chunk ID: chk-xyz" in result.context_text
    assert "Source: system_design.docx" in result.context_text
    assert "Section: System Overview > Database Tier" in result.context_text


def test_metadata_and_provenance_optional_exclusion():
    """Verify metadata and provenance can be selectively excluded via config or arguments."""
    item = _make_context_item(
        chunk_id="chk-xyz",
        document_id="doc-abc",
        document_version_id="ver-42",
        source="system_design.docx",
        section_path=("System Overview", "Database Tier"),
        content="Primary PostgreSQL node with two read replicas.",
    )
    context = _make_assembled_context(items=[item])

    result = construct_prompt(
        query="DB setup",
        context=context,
        include_metadata=False,
        include_provenance=False,
    )

    assert "Document ID:" not in result.context_text
    assert "Document Version:" not in result.context_text
    assert "Chunk ID:" not in result.context_text
    assert "Source:" not in result.context_text
    assert "Section:" not in result.context_text
    # But content must still be present!
    assert "Primary PostgreSQL node with two read replicas." in result.context_text


# ---------------------------------------------------------------------------
# 5. Empty Context Handling
# ---------------------------------------------------------------------------


def test_empty_context_allowed_by_default():
    """When context has 0 items, prompt is constructed with empty context placeholder."""
    context = _make_assembled_context(items=[])

    result = construct_prompt(query="Any data?", context=context)

    assert result.context_items_count == 0
    assert "[No retrieved context provided]" in result.context_text
    assert "[No retrieved context provided]" in result.user_prompt
    assert "Any data?" in result.user_prompt


def test_empty_context_raises_when_disallowed():
    """When allow_empty_context is False, empty context raises validation error."""
    config = PromptConstructionConfig(allow_empty_context=False)
    service = PromptConstructionService(config=config)
    context = _make_assembled_context(items=[])

    with pytest.raises(PromptConstructionValidationError) as exc_info:
        service.construct(query="Any data?", context=context)

    assert "zero items and allow_empty_context is False" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Special Characters, Code Blocks, and Unicode
# ---------------------------------------------------------------------------


def test_special_characters_and_code_blocks():
    """Verify prompt formatting does not corrupt markdown code blocks, backticks, or unusual characters."""
    code_text = (
        "Here is a code snippet:\n"
        "```python\n"
        "def compute_hash(data: str) -> str:\n"
        '    """Compute SHA-256 hash."""\n'
        "    return hashlib.sha256(data.encode('utf-8')).hexdigest()\n"
        "```\n"
        "Formula: f(x) = x^2 + 2x + 1; where x ∈ ℝ."
    )
    item = _make_context_item(content=code_text)
    context = _make_assembled_context(items=[item])

    query = "How to hash data with `hashlib` & SHA-256?"
    result = construct_prompt(query=query, context=context)

    assert code_text in result.context_text
    assert query in result.user_prompt
    assert "def compute_hash(data: str) -> str:" in result.user_prompt


def test_unicode_and_emojis():
    """Verify non-ASCII, emojis, and multilingual text are preserved verbatim."""
    multilingual_text = "日本語のドキュメント: 検索拡張生成 (RAG) 🚀 🔍\nÜbersicht über die Architektur — café résumé."
    multilingual_query = "日本のRAGアーキテクチャは？ 🌟"

    item = _make_context_item(content=multilingual_text)
    context = _make_assembled_context(items=[item])

    result = construct_prompt(query=multilingual_query, context=context)

    assert multilingual_text in result.context_text
    assert multilingual_query in result.user_prompt
    assert result.user_query == multilingual_query


# ---------------------------------------------------------------------------
# 7. Provider Independence & Model Contracts
# ---------------------------------------------------------------------------


def test_provider_independence_contract():
    """Verify ConstructedPrompt provides standard chat messages and completion prompt."""
    context = _make_assembled_context(items=[_make_context_item()])
    prompt = construct_prompt(query="Test provider independence", context=context)

    # 1. Standard chat messages
    messages = prompt.to_messages()
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert set(messages[0].keys()) == {"role", "content"}
    assert set(messages[1].keys()) == {"role", "content"}
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # 2. Raw completion prompt
    raw = prompt.raw_prompt
    assert isinstance(raw, str)
    assert prompt.system_instruction in raw
    assert prompt.user_prompt in raw

    # 3. Serialization to dict
    data = prompt.to_dict()
    assert isinstance(data, dict)
    assert data["user_query"] == "Test provider independence"
    assert data["context_items_count"] == 1
    assert "elapsed_ms" in data["metadata"]


# ---------------------------------------------------------------------------
# 8. Tenant Isolation & Project Boundaries
# ---------------------------------------------------------------------------


def test_tenant_boundary_matching():
    """Verify matching project_id succeeds."""
    context = _make_assembled_context(items=[_make_context_item(project_id="tenant-99")], project_id="tenant-99")
    result = construct_prompt(query="Tenant query", context=context, project_id="tenant-99")
    assert result.project_id == "tenant-99"


def test_tenant_boundary_mismatch_raises():
    """Verify mismatched project_id raises PromptConstructionValidationError."""
    context = _make_assembled_context(project_id="tenant-A")

    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="Cross-tenant query", context=context, project_id="tenant-B")

    assert "violates requested project boundary" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. Validation Failures
# ---------------------------------------------------------------------------


def test_validation_failure_on_none_query():
    """None query raises PromptConstructionValidationError."""
    context = _make_assembled_context(items=[_make_context_item()])
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query=None, context=context)
    assert "cannot be None" in str(exc_info.value)


def test_validation_failure_on_empty_query():
    """Empty string or whitespace query raises PromptConstructionValidationError."""
    context = _make_assembled_context(items=[_make_context_item()])
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="   ", context=context)
    assert "empty or contain only whitespace" in str(exc_info.value)


def test_validation_failure_on_invalid_query_type():
    """Non-string/non-query object raises PromptConstructionValidationError."""
    context = _make_assembled_context(items=[_make_context_item()])
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query=12345, context=context)
    assert "Expected str or object with 'original_query'" in str(exc_info.value)


def test_validation_failure_on_none_context():
    """None context raises PromptConstructionValidationError."""
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="Valid query", context=None)
    assert "cannot be None" in str(exc_info.value)


def test_validation_failure_on_invalid_context_type():
    """Invalid context type raises PromptConstructionValidationError."""
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="Valid query", context="not a context object")
    assert "Expected AssembledContext or Sequence" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10. Async Entrypoint & Singleton Management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_construct_prompt():
    """Verify construct_prompt_async executes correctly and returns ConstructedPrompt."""
    context = _make_assembled_context(items=[_make_context_item()])
    result = await construct_prompt_async(query="Async query", context=context)

    assert isinstance(result, ConstructedPrompt)
    assert result.user_query == "Async query"
    assert result.context_items_count == 1


def test_singleton_get_and_reset():
    """Verify singleton caching and reset functionality."""
    service1 = get_prompt_construction_service()
    service2 = get_prompt_construction_service()
    assert service1 is service2

    reset_prompt_construction_service()
    service3 = get_prompt_construction_service()
    assert service3 is not service1


# ---------------------------------------------------------------------------
# 11. Re-exports Across Packages
# ---------------------------------------------------------------------------


def test_reexports_across_packages():
    """Verify components can be imported from generation, generation.prompt, and rag.generation."""
    assert ConstructedPrompt is PromptConstructedPrompt
    assert ConstructedPrompt is RagConstructedPrompt
    assert GenerationPrompt is RagGenerationPrompt
    assert StructuredPrompt is RagStructuredPrompt
    assert PromptConstructionService is PromptPromptConstructionService
    assert PromptConstructionService is RagPromptConstructionService
    assert PromptConstructionConfig is PromptPromptConstructionConfig
    assert PromptConstructionConfig is RagPromptConstructionConfig
    assert PromptConstructionError is RagPromptConstructionError
    assert PromptConstructionValidationError is RagPromptConstructionValidationError
    assert construct_prompt is prompt_construct_prompt
    assert construct_prompt is rag_construct_prompt
    assert construct_prompt_async is prompt_construct_prompt_async
    assert construct_prompt_async is rag_construct_prompt_async


# ---------------------------------------------------------------------------
# 12. Additional Edge Cases & Configuration
# ---------------------------------------------------------------------------


def test_duck_typed_query_object():
    """Custom object with original_query is handled correctly."""
    class CustomQueryHolder:
        def __init__(self, orig: str, transformed: str):
            self.original_query = orig
            self.transformed = transformed

    query_obj = CustomQueryHolder("Original specialized query", "transformed query")
    context = _make_assembled_context(items=[_make_context_item()])

    result = construct_prompt(query=query_obj, context=context)
    assert result.user_query == "Original specialized query"
    assert "transformed query" not in result.user_prompt


def test_duck_typed_context_sequence():
    """Direct sequence of context items (e.g. list) is formatted cleanly."""
    item = _make_context_item(chunk_id="chk-seq", content="Sequence item text.", project_id="tenant-seq")
    items_list = [item]

    result = construct_prompt(query="Sequence test", context=items_list)
    assert result.context_items_count == 1
    assert "Sequence item text." in result.context_text
    assert result.project_id == "tenant-seq"


def test_heading_fallback_when_section_path_empty():
    """When section_path is empty, item uses heading if present."""
    item = _make_context_item(
        chunk_id="chk-heading",
        content="Heading fallback test.",
        heading="Disaster Recovery Strategy",
        section_path=(),
    )
    context = _make_assembled_context(items=[item])

    result = construct_prompt(query="Heading test", context=context)
    assert "Section: Disaster Recovery Strategy" in result.context_text


def test_config_from_settings_and_env():
    """PromptConstructionConfig correctly initializes from Settings and env."""
    config = PromptConstructionConfig.from_settings()
    assert isinstance(config.default_system_instruction, str)
    assert config.include_metadata is True
    assert config.include_provenance is True
    assert config.allow_empty_context is True

    env_config = PromptConstructionConfig.from_env()
    assert env_config == config


def test_validation_failure_on_none_item_in_context():
    """A context item that is None raises PromptConstructionValidationError."""
    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="Valid query", context=[None])
    assert "Context item cannot be None" in str(exc_info.value)


def test_validation_failure_on_malformed_item():
    """An item missing chunk_id and content raises PromptConstructionValidationError."""
    class MalformedItem:
        pass

    with pytest.raises(PromptConstructionValidationError) as exc_info:
        construct_prompt(query="Valid query", context=[MalformedItem()])
    assert "Malformed context item" in str(exc_info.value)


def test_constructed_prompt_repr():
    """Safe __repr__ displays query preview without dumping huge context."""
    long_query = "A" * 100
    context = _make_assembled_context(items=[_make_context_item()])
    result = construct_prompt(query=long_query, context=context)

    repr_str = repr(result)
    assert "ConstructedPrompt(" in repr_str
    assert "context_items_count=1" in repr_str
    assert "..." in repr_str


def test_metadata_in_constructed_prompt():
    """Metrics in metadata accurately record character and item counts."""
    item = _make_context_item(content="Short content.")
    context = _make_assembled_context(items=[item])
    result = construct_prompt(query="Short query", context=context)

    meta = result.metadata
    assert "elapsed_ms" in meta
    assert meta["context_items_count"] == 1
    assert meta["query_characters"] == len("Short query")
    assert meta["context_characters"] > 0
    assert meta["prompt_characters"] > 0


def test_empty_system_instruction_override():
    """Empty system instruction produces only user message without system role."""
    context = _make_assembled_context(items=[_make_context_item()])
    result = construct_prompt(query="No system query", context=context, system_instruction="")

    assert result.system_instruction == ""
    assert len(result.messages) == 1
    assert result.messages[0]["role"] == "user"
    assert result.raw_prompt == result.user_prompt


def test_exception_inheritance():
    """Verify exception hierarchy matches domain design."""
    assert issubclass(PromptConstructionValidationError, PromptConstructionError)
    assert issubclass(PromptConstructionError, GenerationError)
    assert issubclass(GenerationError, Exception)
