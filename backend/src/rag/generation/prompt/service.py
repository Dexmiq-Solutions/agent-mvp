"""Prompt Construction service transforming original query and final context into structured prompts."""

import time
from typing import Any, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.generation import PromptConstructionValidationError
from rag.generation.prompt.config import PromptConstructionConfig
from rag.generation.prompt.models import ConstructedPrompt

logger = get_logger(__name__)

_prompt_construction_service: Optional["PromptConstructionService"] = None


class PromptConstructionService:
    """Service responsible for constructing structured, provider-independent generation prompts.

    Positions at the generation boundary directly following Context Assembly and Relevance Check.
    Consumes the original user query and the authoritative Final Context, faithfully preserving
    retrieval ordering, source provenance, chunk/document identity, and content integrity,
    while cleanly separating instructions, retrieved context, and original user query.
    """

    def __init__(
        self,
        config: Optional[PromptConstructionConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize PromptConstructionService.

        Args:
            config: Optional PromptConstructionConfig override.
            settings: Optional application Settings instance.
        """
        self._settings = settings or get_settings()
        self._config = config or PromptConstructionConfig.from_settings(self._settings)

    @property
    def config(self) -> PromptConstructionConfig:
        """Return active prompt construction configuration."""
        return self._config

    def _extract_original_query(self, query: Any) -> str:
        """Extract and validate the original user query string from query input.

        Strictly extracts the original user query. Never substitutes transformed,
        rewritten, or expanded retrieval queries.

        Args:
            query: User query as string, ProcessedQuery, RetrievalQuerySet, or compatible object.

        Returns:
            str: Validated non-empty original query string.

        Raises:
            PromptConstructionValidationError: If query is missing, invalid type, or empty.
        """
        if query is None:
            raise PromptConstructionValidationError("Original user query cannot be None.")

        query_str: Optional[str] = None

        if isinstance(query, str):
            query_str = query
        elif hasattr(query, "original_query") and isinstance(query.original_query, str):
            # Prioritize original_query on ProcessedQuery or RetrievalQuerySet
            query_str = query.original_query
        elif hasattr(query, "query") and isinstance(query.query, str):
            query_str = query.query
        elif not isinstance(query, (int, float, bool, list, dict, set, tuple)):
            # Duck-typed object with str representation if not standard scalar/collection
            candidate = str(query).strip()
            if candidate:
                query_str = candidate

        if query_str is None:
            raise PromptConstructionValidationError(
                f"Expected str or object with 'original_query', got '{type(query).__name__}'."
            )

        clean_query = query_str.strip()
        if not clean_query:
            raise PromptConstructionValidationError(
                "Original user query cannot be empty or contain only whitespace."
            )

        return clean_query

    def _extract_context_items(self, context: Any) -> tuple[Sequence[Any], Optional[str]]:
        """Extract ordered context items and project boundary identifier from context.

        Args:
            context: AssembledContext, Sequence of context items, or compatible container.

        Returns:
            tuple[Sequence[Any], Optional[str]]: Ordered items and resolved project_id.

        Raises:
            PromptConstructionValidationError: If context is None or malformed.
        """
        if context is None:
            raise PromptConstructionValidationError("Retrieved context cannot be None.")

        # Handle AssembledContext / StructuredRetrievalContext
        if hasattr(context, "items") and isinstance(context.items, Sequence):
            items = context.items
            project_id = getattr(context, "project_id", None)
            return items, str(project_id).strip() if project_id else None

        # Handle raw sequence of items
        if isinstance(context, Sequence) and not isinstance(context, (str, bytes, dict)):
            # Check if items carry project_id
            project_id = None
            for item in context:
                pid = getattr(item, "project_id", None)
                if pid and isinstance(pid, str) and pid.strip():
                    project_id = pid.strip()
                    break
            return context, project_id

        raise PromptConstructionValidationError(
            f"Expected AssembledContext or Sequence of context items, got '{type(context).__name__}'."
        )

    def _format_context_item(
        self,
        index: int,
        item: Any,
        include_metadata: bool,
        include_provenance: bool,
    ) -> str:
        """Format an individual context item preserving identity, provenance, and text verbatim.

        Args:
            index: 1-based display index for context ordering.
            item: AssembledContextItem or compatible context candidate.
            include_metadata: If True, renders document/chunk identifiers.
            include_provenance: If True, renders headings, section path, and source.

        Returns:
            str: Formatted context item string.

        Raises:
            PromptConstructionValidationError: If context item is null or malformed.
        """
        if item is None:
            raise PromptConstructionValidationError("Context item cannot be None.")

        chunk_id = getattr(item, "chunk_id", None)
        document_id = getattr(item, "document_id", None)
        if not chunk_id and not hasattr(item, "text") and not hasattr(item, "content"):
            raise PromptConstructionValidationError(
                f"Malformed context item at index {index}: missing chunk_id and content."
            )

        # Faithful representation of resolved text: verbatim from context assembly
        text = (
            getattr(item, "text", None)
            or getattr(item, "content", None)
            or getattr(item, "chunk_text", None)
            or ""
        )
        text_str = str(text)

        lines: list[str] = [f"[Context Item {index}]"]

        if include_metadata:
            if document_id:
                lines.append(f"Document ID: {document_id}")
            doc_version = getattr(item, "document_version_id", None)
            if doc_version:
                lines.append(f"Document Version: {doc_version}")
            if chunk_id:
                lines.append(f"Chunk ID: {chunk_id}")

        if include_provenance:
            source = (
                getattr(item, "source", None)
                or (item.metadata.get("source") if hasattr(item, "metadata") and isinstance(item.metadata, dict) else None)
            )
            if source:
                lines.append(f"Source: {source}")

            section_path = getattr(item, "section_path", ()) or ()
            if section_path:
                if isinstance(section_path, (list, tuple)):
                    path_str = " > ".join(str(p) for p in section_path if str(p).strip())
                else:
                    path_str = str(section_path)
                if path_str:
                    lines.append(f"Section: {path_str}")
            elif getattr(item, "heading", None):
                lines.append(f"Section: {item.heading}")

        lines.append("Content:")
        lines.append(text_str)

        return "\n".join(lines)

    def construct(
        self,
        query: Any,
        context: Any,
        system_instruction: Optional[str] = None,
        project_id: Optional[str] = None,
        include_metadata: Optional[bool] = None,
        include_provenance: Optional[bool] = None,
        evaluation_feedback: Optional[str] = None,
    ) -> ConstructedPrompt:
        """Construct a structured prompt from original user query and final retrieval context.

        Args:
            query: Original user query as string, ProcessedQuery, or RetrievalQuerySet.
            context: Authoritative retrieval context (AssembledContext or Sequence of items).
            system_instruction: Optional system instruction override.
            project_id: Optional project/tenant ID enforcing project boundary.
            include_metadata: Optional override for including document/chunk metadata.
            include_provenance: Optional override for including structural provenance.
            evaluation_feedback: Optional feedback from previous failed evaluation attempt for regeneration.

        Returns:
            ConstructedPrompt: Provider-independent structured prompt container.

        Raises:
            PromptConstructionValidationError: If inputs fail validation or context is invalid.
        """
        start_time = time.perf_counter()

        # 1. Extract and Validate Original User Query
        original_query = self._extract_original_query(query)

        # 2. Extract and Validate Context Items & Tenant Isolation
        raw_items, ctx_project_id = self._extract_context_items(context)

        eff_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise PromptConstructionValidationError("project_id must be a non-empty string when specified.")
            eff_project_id = project_id.strip()

        if eff_project_id is not None and ctx_project_id is not None and eff_project_id != ctx_project_id:
            raise PromptConstructionValidationError(
                f"Context project_id '{ctx_project_id}' violates requested project boundary '{eff_project_id}'."
            )

        resolved_project_id = eff_project_id or ctx_project_id or ""

        # 3. Handle Empty Context
        if len(raw_items) == 0 and not self._config.allow_empty_context:
            raise PromptConstructionValidationError(
                "Retrieved context contains zero items and allow_empty_context is False."
            )

        # 4. Resolve Configuration Overrides
        eff_system_instruction = (
            system_instruction
            if system_instruction is not None
            else self._config.default_system_instruction
        )
        eff_include_metadata = (
            include_metadata
            if include_metadata is not None
            else self._config.include_metadata
        )
        eff_include_provenance = (
            include_provenance
            if include_provenance is not None
            else self._config.include_provenance
        )

        # 5. Format Retrieved Context Faithfully Preserving Order and Content
        if hasattr(context, "text") and hasattr(context, "items") and not hasattr(context, "chunk_id"):
            # Directly reuse pre-formatted FormattedContext produced upstream by Context Formatting
            formatted_context = str(context.text)
        elif len(raw_items) == 0:
            formatted_context = self._config.empty_context_text
        else:
            formatted_items: list[str] = []
            for idx, item in enumerate(raw_items, start=1):
                formatted_item = self._format_context_item(
                    index=idx,
                    item=item,
                    include_metadata=eff_include_metadata,
                    include_provenance=eff_include_provenance,
                )
                formatted_items.append(formatted_item)
            formatted_context = "\n\n".join(formatted_items)

        # 6. Assemble Composite User Prompt with Clean Logical Separation
        user_prompt_sections: list[str] = [
            self._config.context_header,
            formatted_context,
            "",
            self._config.query_header,
            original_query,
        ]
        if evaluation_feedback and str(evaluation_feedback).strip():
            user_prompt_sections.extend([
                "",
                "[PREVIOUS ATTEMPT FEEDBACK - REGENERATION REQUIRED]",
                f"Your previous response was rejected by the quality gate for the following reason:\n{str(evaluation_feedback).strip()}",
                "Please regenerate your answer addressing this feedback. Ensure every claim is strictly supported by the retrieved context above and satisfies all safety criteria.",
            ])
        user_prompt = "\n".join(user_prompt_sections)

        # 7. Construct Provider-Independent Chat Messages
        messages: list[dict[str, str]] = []
        if eff_system_instruction and eff_system_instruction.strip():
            messages.append({"role": "system", "content": eff_system_instruction.strip()})
        messages.append({"role": "user", "content": user_prompt})

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "Prompt construction completed in %.2fms: project_id='%s', context_items=%d, prompt_chars=%d",
            elapsed_ms,
            resolved_project_id,
            len(raw_items),
            len(user_prompt),
        )

        return ConstructedPrompt(
            system_instruction=eff_system_instruction.strip() if eff_system_instruction else "",
            user_query=original_query,
            context_text=formatted_context,
            user_prompt=user_prompt,
            messages=tuple(messages),
            project_id=resolved_project_id if resolved_project_id else None,
            context_items_count=len(raw_items),
            metadata={
                "elapsed_ms": round(elapsed_ms, 2),
                "context_items_count": len(raw_items),
                "prompt_characters": len(user_prompt),
                "context_characters": len(formatted_context),
                "query_characters": len(original_query),
                "has_evaluation_feedback": bool(evaluation_feedback and str(evaluation_feedback).strip()),
            },
        )


def get_prompt_construction_service(
    config: Optional[PromptConstructionConfig] = None,
    settings: Optional[Settings] = None,
) -> PromptConstructionService:
    """Retrieve or initialize the singleton PromptConstructionService instance.

    Args:
        config: Optional PromptConstructionConfig override.
        settings: Optional Settings override.

    Returns:
        PromptConstructionService: Configured service instance.
    """
    global _prompt_construction_service
    if _prompt_construction_service is None or any(arg is not None for arg in (config, settings)):
        service = PromptConstructionService(
            config=config,
            settings=settings,
        )
        if all(arg is None for arg in (config, settings)):
            _prompt_construction_service = service
        return service
    return _prompt_construction_service


def reset_prompt_construction_service() -> None:
    """Reset the cached singleton PromptConstructionService instance (useful for tests)."""
    global _prompt_construction_service
    _prompt_construction_service = None


def construct_prompt(
    query: Any,
    context: Any,
    system_instruction: Optional[str] = None,
    project_id: Optional[str] = None,
    include_metadata: Optional[bool] = None,
    include_provenance: Optional[bool] = None,
    evaluation_feedback: Optional[str] = None,
    service: Optional[PromptConstructionService] = None,
) -> ConstructedPrompt:
    """Functional convenience entrypoint to construct a structured prompt.

    Args:
        query: Original user query as string, ProcessedQuery, or RetrievalQuerySet.
        context: Authoritative retrieval context (AssembledContext or Sequence of items).
        system_instruction: Optional system instruction override.
        project_id: Optional project boundary identifier.
        include_metadata: Optional flag to include document/chunk identifiers.
        include_provenance: Optional flag to include structural provenance.
        evaluation_feedback: Optional corrective feedback string for regeneration.
        service: Optional PromptConstructionService override.

    Returns:
        ConstructedPrompt: Provider-independent structured prompt container.
    """
    active_service = service or get_prompt_construction_service()
    return active_service.construct(
        query=query,
        context=context,
        system_instruction=system_instruction,
        project_id=project_id,
        include_metadata=include_metadata,
        include_provenance=include_provenance,
        evaluation_feedback=evaluation_feedback,
    )


async def construct_prompt_async(
    query: Any,
    context: Any,
    system_instruction: Optional[str] = None,
    project_id: Optional[str] = None,
    include_metadata: Optional[bool] = None,
    include_provenance: Optional[bool] = None,
    evaluation_feedback: Optional[str] = None,
    service: Optional[PromptConstructionService] = None,
) -> ConstructedPrompt:
    """Async convenience wrapper for construct_prompt in async RAG pipeline chains.

    Retains fast, synchronous in-memory formatting internally without artificial async delay.
    """
    return construct_prompt(
        query=query,
        context=context,
        system_instruction=system_instruction,
        project_id=project_id,
        include_metadata=include_metadata,
        include_provenance=include_provenance,
        evaluation_feedback=evaluation_feedback,
        service=service,
    )
