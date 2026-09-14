"""Unit tests for the Context Formatting stage of the RAG generation pipeline."""

from typing import Any, Optional
import pytest

from exceptions.generation import (
    ContextFormattingError,
    ContextFormattingValidationError,
    GenerationError,
)
from rag.generation import (
    DEFAULT_CONTEXT_FOOTER,
    DEFAULT_CONTEXT_HEADER,
    DEFAULT_EMPTY_CONTEXT_TEXT,
    DEFAULT_ITEM_TEMPLATE,
    BaseContextFormatter,
    ContextFormattingConfig,
    ContextFormattingService,
    FormattedContext,
    FormattedContextItem,
    TextContextFormatter,
    format_context,
    format_context_async,
    get_context_formatting_service,
    reset_context_formatting_service,
    construct_prompt,
)
from rag.generation.formatting import (
    ContextFormattingConfig as FormattingContextFormattingConfig,
    ContextFormattingService as FormattingContextFormattingService,
    FormattedContext as FormattingFormattedContext,
    FormattedContextItem as FormattingFormattedContextItem,
    TextContextFormatter as FormattingTextContextFormatter,
    format_context as formatting_format_context,
    format_context_async as formatting_format_context_async,
    get_context_formatting_service as formatting_get_context_formatting_service,
    reset_context_formatting_service as formatting_reset_context_formatting_service,
)
from rag.generation import (
    ContextFormattingConfig as RagContextFormattingConfig,
    ContextFormattingError as RagContextFormattingError,
    ContextFormattingService as RagContextFormattingService,
    ContextFormattingValidationError as RagContextFormattingValidationError,
    FormattedContext as RagFormattedContext,
    FormattedContextItem as RagFormattedContextItem,
    TextContextFormatter as RagTextContextFormatter,
    format_context as rag_format_context,
    format_context_async as rag_format_context_async,
    get_context_formatting_service as rag_get_context_formatting_service,
    reset_context_formatting_service as rag_reset_context_formatting_service,
)
from rag.retrieval.models import (
    AssembledContext,
    AssembledContextItem,
)


@pytest.fixture(autouse=True)
def reset_service_state():
    """Reset context formatting singletons before and after each test."""
    reset_context_formatting_service()
    formatting_reset_context_formatting_service()
    rag_reset_context_formatting_service()
    yield
    reset_context_formatting_service()
    formatting_reset_context_formatting_service()
    rag_reset_context_formatting_service()


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
    title: Optional[str] = None,
    page: Optional[int | str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AssembledContextItem:
    """Helper to construct an AssembledContextItem for formatting tests."""
    resolved_text = text if text is not None else content
    cand_metadata = dict(metadata or {})
    if source and "source" not in cand_metadata:
        cand_metadata["source"] = source
    if title and "title" not in cand_metadata:
        cand_metadata["title"] = title
    if page is not None and "page" not in cand_metadata:
        cand_metadata["page"] = page

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
# 1. Basic Formatting & Delimiters
# ---------------------------------------------------------------------------


def test_basic_context_formatting():
    """Verify valid FormattedContext produced from AssembledContext matching specification."""
    item = _make_context_item(
        chunk_id="chk-100",
        document_id="doc-100",
        title="Authentication Requirements",
        heading="Authentication",
        section_path=("Authentication",),
        content="OAuth 2.0 is used for user authentication.",
    )
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert isinstance(result, FormattedContext)
    assert result.item_count == 1
    assert result.project_id == "proj-1"
    assert not result.is_empty
    assert len(result) == 1

    # Delimiters
    assert result.text.startswith(DEFAULT_CONTEXT_HEADER)
    assert result.text.endswith(DEFAULT_CONTEXT_FOOTER)
    assert "[Context 1]" in result.text
    assert "Document: Authentication Requirements" in result.text
    assert "Section: Authentication" in result.text
    assert "OAuth 2.0 is used for user authentication." in result.text

    # Individual item
    assert len(result.items) == 1
    item_0 = result[0]
    assert isinstance(item_0, FormattedContextItem)
    assert item_0.index == 1
    assert item_0.content == "OAuth 2.0 is used for user authentication."
    assert item_0.document_id == "doc-100"
    assert str(result) == result.text


def test_custom_delimiters_and_headers():
    """Verify configurable headers, footers, and item label templates."""
    config = ContextFormattingConfig(
        context_header="=== EVIDENCE START ===",
        context_footer="=== EVIDENCE END ===",
        item_label_template="[Evidence Item #{index}]",
    )
    service = ContextFormattingService(config=config)
    item = _make_context_item(content="Evidence details.")
    context = _make_assembled_context(items=[item])

    result = service.format(context)

    assert result.text.startswith("=== EVIDENCE START ===")
    assert result.text.endswith("=== EVIDENCE END ===")
    assert "[Evidence Item #1]" in result.text
    assert "Evidence details." in result.text


def test_optional_header_footer_omission():
    """Verify header and footer can be cleanly omitted when disabled."""
    config = ContextFormattingConfig(
        include_header=False,
        include_footer=False,
    )
    service = ContextFormattingService(config=config)
    item = _make_context_item(content="Body text only.")
    context = _make_assembled_context(items=[item])

    result = service.format(context)

    assert DEFAULT_CONTEXT_HEADER not in result.text
    assert DEFAULT_CONTEXT_FOOTER not in result.text
    assert result.text.startswith("[Context 1]")


# ---------------------------------------------------------------------------
# 2. Strict Ordering Preservation
# ---------------------------------------------------------------------------


def test_multiple_items_ordering_preserved():
    """Verify Context Assembly ordering is strictly preserved without reordering."""
    item_a = _make_context_item(chunk_id="chk-A", title="Doc A", content="Content of chunk A.", rank=1)
    item_b = _make_context_item(chunk_id="chk-B", title="Doc B", content="Content of chunk B.", rank=2)
    item_c = _make_context_item(chunk_id="chk-C", title="Doc C", content="Content of chunk C.", rank=3)

    # Retrieval order: B, A, C (must strictly remain B, A, C)
    context = _make_assembled_context(items=[item_b, item_a, item_c])

    result = format_context(context)

    assert result.item_count == 3
    pos_b = result.text.find("chk-B")
    pos_a = result.text.find("chk-A")
    pos_c = result.text.find("chk-C")

    assert pos_b < pos_a < pos_c, "Context items must strictly preserve retrieval order"

    assert "[Context 1]" in result.items[0].text
    assert "chk-B" in result.items[0].text
    assert "[Context 2]" in result.items[1].text
    assert "chk-A" in result.items[1].text
    assert "[Context 3]" in result.items[2].text
    assert "chk-C" in result.items[2].text


# ---------------------------------------------------------------------------
# 3. Evidence Integrity & Contradictory Content Preservation
# ---------------------------------------------------------------------------


def test_verbatim_content_integrity():
    """Verify retrieved content is preserved verbatim without rewriting, summarization, or alteration."""
    raw_content = (
        "  Leading whitespace preserved.\n"
        "Special symbols: $ & < > \" ' % ; : ! ?\n"
        "Tabs:\t\tDouble indent.\n"
        "Trailing whitespace.  "
    )
    item = _make_context_item(content=raw_content, text=raw_content)
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert raw_content in result.text
    assert result.items[0].content == raw_content


def test_conflicting_contradictory_evidence_preserved():
    """Verify conflicting statements across chunks are both preserved without resolving contradictions."""
    item1 = _make_context_item(
        chunk_id="c1",
        title="Authentication Spec v1",
        content="The system uses OAuth 2.0 for user authentication.",
    )
    item2 = _make_context_item(
        chunk_id="c2",
        title="Authentication Spec v2",
        content="The system uses SAML 2.0 for enterprise single sign-on.",
    )
    context = _make_assembled_context(items=[item1, item2])

    result = format_context(context)

    assert "The system uses OAuth 2.0 for user authentication." in result.text
    assert "The system uses SAML 2.0 for enterprise single sign-on." in result.text
    assert result.item_count == 2


# ---------------------------------------------------------------------------
# 4. Metadata-Aware Representation & Hierarchy
# ---------------------------------------------------------------------------


def test_metadata_aware_representation():
    """Verify Document title/ID, version, source, page, section hierarchy are cleanly exposed."""
    item = _make_context_item(
        chunk_id="chk-xyz",
        document_id="doc-999",
        title="System Architecture Document",
        document_version_id="v2.1",
        source="specs/system_architecture.pdf",
        page=42,
        section_path=("Compute Tier", "Worker Services", "Autoscaling"),
        content="Workers scale based on queue depth exceeding 100 messages.",
    )
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert "Document: System Architecture Document" in result.text
    assert "Document ID: doc-999" in result.text
    assert "Document Version: v2.1" in result.text
    assert "Chunk ID: chk-xyz" in result.text
    assert "Source: specs/system_architecture.pdf" in result.text
    assert "Page: 42" in result.text
    assert "Section: Compute Tier > Worker Services > Autoscaling" in result.text
    assert "Workers scale based on queue depth exceeding 100 messages." in result.text


def test_omits_internal_retrieval_plumbing():
    """Verify internal retrieval details (fusion_score, rerank_score, dense_rank) are NOT exposed in formatted text."""
    item = _make_context_item(
        chunk_id="chk-clean",
        content="Payment processing SLA is 200ms.",
        metadata={"internal_db_row": 98172, "embedding_model": "voyage-4"},
    )
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert "fusion_score" not in result.text
    assert "rerank_score" not in result.text
    assert "dense_rank" not in result.text
    assert "embedding_model" not in result.text
    assert "internal_db_row" not in result.text


def test_clean_omission_of_missing_metadata():
    """Verify missing optional fields (page, version, section) are omitted cleanly without placeholders."""
    item = _make_context_item(
        chunk_id="chk-minimal",
        document_id="doc-min",
        document_version_id=None,
        heading=None,
        section_path=(),
        source=None,
        title=None,
        page=None,
        content="Minimal item text without optional metadata.",
    )
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert "Document Version:" not in result.text
    assert "Source:" not in result.text
    assert "Page:" not in result.text
    assert "Section:" not in result.text
    assert "None" not in result.text
    assert "null" not in result.text
    assert "N/A" not in result.text
    assert "Minimal item text without optional metadata." in result.text


def test_hierarchy_formatting():
    """Verify section hierarchy formatting for tuple section paths and standalone headings."""
    # 1. Multi-level hierarchy
    item_nested = _make_context_item(
        chunk_id="c-nested",
        section_path=("Product Requirements", "Payments", "Refund Processing"),
        content="Refunds are processed within 3 business days.",
    )
    res_nested = format_context(_make_assembled_context(items=[item_nested]))
    assert "Section: Product Requirements > Payments > Refund Processing" in res_nested.text

    # 2. Standalone heading fallback
    item_heading = _make_context_item(
        chunk_id="c-head",
        heading="Incident Escalation Procedures",
        section_path=(),
        content="Escalate P1 incidents to on-call within 5 minutes.",
    )
    res_heading = format_context(_make_assembled_context(items=[item_heading]))
    assert "Section: Incident Escalation Procedures" in res_heading.text


# ---------------------------------------------------------------------------
# 5. Empty Context Handling
# ---------------------------------------------------------------------------


def test_empty_context_allowed_by_default():
    """When context has 0 items, FormattedContext contains default placeholder."""
    context = _make_assembled_context(items=[])

    result = format_context(context)

    assert result.item_count == 0
    assert result.is_empty
    assert result.text == DEFAULT_EMPTY_CONTEXT_TEXT
    assert len(result.items) == 0


def test_empty_context_raises_when_disallowed():
    """When allow_empty_context is False, empty context raises ContextFormattingValidationError."""
    config = ContextFormattingConfig(allow_empty_context=False)
    service = ContextFormattingService(config=config)
    context = _make_assembled_context(items=[])

    with pytest.raises(ContextFormattingValidationError) as exc_info:
        service.format(context)

    assert "zero items and allow_empty_context is False" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Special Characters, Code, Unicode, and Markdown
# ---------------------------------------------------------------------------


def test_special_characters_markdown_and_code():
    """Verify formatter does not corrupt Markdown formatting, code fences, quotes, or formulas."""
    code_content = (
        "### Database Schema\n\n"
        "```sql\n"
        "CREATE TABLE users (\n"
        "    id UUID PRIMARY KEY,\n"
        "    email VARCHAR(255) NOT NULL UNIQUE\n"
        ");\n"
        "```\n\n"
        "> Quote: 'Always use parameterized queries.'"
    )
    item = _make_context_item(content=code_content)
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert code_content in result.text
    assert "```sql" in result.text
    assert "CREATE TABLE users (" in result.text


def test_unicode_and_emojis():
    """Verify non-ASCII multilingual text and emojis are faithfully represented."""
    text = "RAG システムの概要 🚀 🌟: ドキュメント検索とプロンプト構築。\nCafé & Résumé — Überwachung."
    item = _make_context_item(content=text)
    context = _make_assembled_context(items=[item])

    result = format_context(context)

    assert text in result.text


# ---------------------------------------------------------------------------
# 7. Query Separation & Boundary Guarantees
# ---------------------------------------------------------------------------


def test_query_separation():
    """Verify Context Formatting does NOT take, inject, or duplicate the user query."""
    context = _make_assembled_context(
        items=[_make_context_item(content="Only context content here.")],
        query="What is the query that must NOT appear in formatted context?",
    )

    result = format_context(context)

    assert "What is the query that must NOT appear in formatted context?" not in result.text
    assert "Only context content here." in result.text


def test_determinism():
    """Verify identical context and config produce identical formatted output every time."""
    item1 = _make_context_item(chunk_id="c1", content="Chunk 1 content.")
    item2 = _make_context_item(chunk_id="c2", content="Chunk 2 content.")
    context = _make_assembled_context(items=[item1, item2])

    result1 = format_context(context)
    result2 = format_context(context)

    assert result1.text == result2.text
    assert len(result1.items) == len(result2.items)


# ---------------------------------------------------------------------------
# 8. Tenant Isolation & Project Boundaries
# ---------------------------------------------------------------------------


def test_tenant_boundary_matching():
    """Verify matching project_id succeeds cleanly."""
    context = _make_assembled_context(
        items=[_make_context_item(project_id="tenant-100")],
        project_id="tenant-100",
    )
    result = format_context(context, project_id="tenant-100")
    assert result.project_id == "tenant-100"


def test_tenant_boundary_mismatch_raises():
    """Verify mismatched project_id raises ContextFormattingValidationError."""
    context = _make_assembled_context(project_id="tenant-A")

    with pytest.raises(ContextFormattingValidationError) as exc_info:
        format_context(context, project_id="tenant-B")

    assert "violates requested project boundary" in str(exc_info.value)


def test_strict_project_validation_on_items():
    """When strict_project_validation is True, cross-tenant items raise validation error."""
    config = ContextFormattingConfig(strict_project_validation=True)
    service = ContextFormattingService(config=config)

    item1 = _make_context_item(chunk_id="c1", project_id="tenant-A")
    item2 = _make_context_item(chunk_id="c2", project_id="tenant-B")
    items = [item1, item2]

    with pytest.raises(ContextFormattingValidationError) as exc_info:
        service.format(context=items, project_id="tenant-A")

    assert "violates requested boundary" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. Validation & Error Handling
# ---------------------------------------------------------------------------


def test_validation_failure_on_none_context():
    """None context raises ContextFormattingValidationError."""
    with pytest.raises(ContextFormattingValidationError) as exc_info:
        format_context(None)
    assert "cannot be None" in str(exc_info.value)


def test_validation_failure_on_invalid_context_type():
    """Invalid context type raises ContextFormattingValidationError."""
    with pytest.raises(ContextFormattingValidationError) as exc_info:
        format_context(12345)
    assert "Expected AssembledContext or Sequence" in str(exc_info.value)


def test_validation_failure_on_none_item_in_sequence():
    """None item inside context sequence raises ContextFormattingValidationError."""
    with pytest.raises(ContextFormattingValidationError) as exc_info:
        format_context([None])
    assert "cannot be None" in str(exc_info.value)


def test_validation_failure_on_malformed_item():
    """Item missing both text and content raises ContextFormattingValidationError."""
    class EmptyItem:
        chunk_id = "chk-bad"

    with pytest.raises(ContextFormattingValidationError) as exc_info:
        format_context([EmptyItem()])
    assert "missing both 'text' and 'content'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10. Async Entrypoint & Singleton Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_format_context():
    """Verify format_context_async executes and returns FormattedContext."""
    item = _make_context_item(content="Async content.")
    context = _make_assembled_context(items=[item])

    result = await format_context_async(context)

    assert isinstance(result, FormattedContext)
    assert result.item_count == 1
    assert "Async content." in result.text


def test_singleton_get_and_reset():
    """Verify singleton caching and reset lifecycle."""
    s1 = get_context_formatting_service()
    s2 = get_context_formatting_service()
    assert s1 is s2

    reset_context_formatting_service()
    s3 = get_context_formatting_service()
    assert s3 is not s1


# ---------------------------------------------------------------------------
# 11. Re-exports Across Packages
# ---------------------------------------------------------------------------


def test_reexports_across_packages():
    """Verify components can be imported from rag.generation, generation.formatting, and rag.generation."""
    assert FormattedContext is FormattingFormattedContext
    assert FormattedContext is RagFormattedContext
    assert FormattedContextItem is FormattingFormattedContextItem
    assert FormattedContextItem is RagFormattedContextItem
    assert ContextFormattingService is FormattingContextFormattingService
    assert ContextFormattingService is RagContextFormattingService
    assert ContextFormattingConfig is FormattingContextFormattingConfig
    assert ContextFormattingConfig is RagContextFormattingConfig
    assert TextContextFormatter is FormattingTextContextFormatter
    assert TextContextFormatter is RagTextContextFormatter
    assert format_context is formatting_format_context
    assert format_context is rag_format_context
    assert format_context_async is formatting_format_context_async
    assert format_context_async is rag_format_context_async
    assert ContextFormattingError is RagContextFormattingError
    assert ContextFormattingValidationError is RagContextFormattingValidationError


# ---------------------------------------------------------------------------
# 12. Strategy Extensibility (BaseContextFormatter)
# ---------------------------------------------------------------------------


def test_custom_formatter_strategy():
    """Verify plugging in a custom formatter strategy via BaseContextFormatter."""
    class MarkdownListFormatter(BaseContextFormatter):
        def format(self, items, project_id=None):
            lines = ["# Sources"]
            for idx, itm in enumerate(items, 1):
                text = getattr(itm, "text", "") or getattr(itm, "content", "")
                lines.append(f"{idx}. {text}")
            return FormattedContext(
                text="\n".join(lines),
                item_count=len(items),
                project_id=project_id,
                strategy="markdown_list",
            )

    custom_formatter = MarkdownListFormatter()
    service = ContextFormattingService(formatter=custom_formatter)
    item1 = _make_context_item(content="Point 1")
    item2 = _make_context_item(content="Point 2")

    result = service.format([item1, item2])

    assert result.strategy == "markdown_list"
    assert result.text == "# Sources\n1. Point 1\n2. Point 2"
    assert result.item_count == 2


# ---------------------------------------------------------------------------
# 13. Pipeline Integration with Prompt Construction
# ---------------------------------------------------------------------------


def test_integration_with_prompt_construction():
    """Verify full generation preparation pipeline: Final Context -> Context Formatting -> Formatted Context -> Prompt Construction."""
    # 1. Authoritative Final Context
    item = _make_context_item(
        chunk_id="chk-arch",
        document_id="doc-arch",
        title="Software Architecture Guide",
        heading="Service Communication",
        content="Microservices communicate via gRPC over HTTP/2.",
    )
    final_context = _make_assembled_context(items=[item])

    # 2. Context Formatting stage
    formatted_context = format_context(final_context)
    assert isinstance(formatted_context, FormattedContext)
    assert "Software Architecture Guide" in formatted_context.text

    # 3. Prompt Construction stage consuming FormattedContext directly
    query = "How do microservices communicate?"
    prompt = construct_prompt(query=query, context=formatted_context)

    # 4. Verification of separation and composition
    assert prompt.user_query == query
    assert prompt.context_text == formatted_context.text
    assert formatted_context.text in prompt.user_prompt
    assert query in prompt.user_prompt
    assert prompt.context_items_count == 1
    assert len(prompt.messages) == 2


# ---------------------------------------------------------------------------
# 14. Performance & Token Efficiency ($O(N)$ single-pass)
# ---------------------------------------------------------------------------


def test_performance_single_pass():
    """Verify single-pass formatting scales linearly with item count."""
    items = [
        _make_context_item(
            chunk_id=f"chunk-{i}",
            title=f"Doc {i}",
            content=f"Content for chunk number {i}.",
            rank=i,
        )
        for i in range(1, 101)
    ]
    context = _make_assembled_context(items=items)

    result = format_context(context)

    assert result.item_count == 100
    assert "[Context 1]" in result.text
    assert "[Context 100]" in result.text
    assert "elapsed_ms" in result.metadata
    assert result.metadata["elapsed_ms"] < 50.0  # 100 items formatted in well under 50ms


def test_exception_inheritance():
    """Verify exception hierarchy matches domain design."""
    assert issubclass(ContextFormattingValidationError, ContextFormattingError)
    assert issubclass(ContextFormattingError, GenerationError)
    assert issubclass(GenerationError, Exception)
