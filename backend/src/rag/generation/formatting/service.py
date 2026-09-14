"""Context Formatting service transforming authoritative retrieval results into model-readable representations."""

import time
from typing import Any, Optional, Sequence

from core.config import Settings, get_settings
from observability.logging import get_logger
from exceptions.generation import ContextFormattingValidationError
from rag.generation.formatting.base import BaseContextFormatter
from rag.generation.formatting.config import ContextFormattingConfig
from rag.generation.formatting.models import FormattedContext
from rag.generation.formatting.text import TextContextFormatter

logger = get_logger(__name__)

_context_formatting_service: Optional["ContextFormattingService"] = None


class ContextFormattingService:
    """Service responsible for formatting authoritative retrieval contexts into model representations.

    Positions at the generation boundary directly following Context Assembly and Relevance Check,
    and directly preceding Prompt Construction.
    Transforms raw structured context items into a clean, model-readable, metadata-aware representation
    while strictly preserving retrieval ordering, source coordinates, and evidence integrity.
    """

    def __init__(
        self,
        config: Optional[ContextFormattingConfig] = None,
        settings: Optional[Settings] = None,
        formatter: Optional[BaseContextFormatter] = None,
    ) -> None:
        """Initialize ContextFormattingService.

        Args:
            config: Optional ContextFormattingConfig override.
            settings: Optional application Settings instance.
            formatter: Optional BaseContextFormatter override.
        """
        self._settings = settings or get_settings()
        self._config = config or ContextFormattingConfig.from_settings(self._settings)
        self._formatter = formatter or self._resolve_formatter(self._config)

    @property
    def config(self) -> ContextFormattingConfig:
        """Return active context formatting configuration."""
        return self._config

    @property
    def formatter(self) -> BaseContextFormatter:
        """Return active context formatter strategy."""
        return self._formatter

    def _resolve_formatter(self, config: ContextFormattingConfig) -> BaseContextFormatter:
        """Resolve formatter strategy instance according to configuration."""
        strategy = (config.strategy or "text").lower().strip()
        if strategy == "text":
            return TextContextFormatter(config=config)
        # Clean extension point for future formatters (xml, json, citation, token_aware)
        raise ContextFormattingValidationError(f"Unsupported context formatting strategy: '{strategy}'")

    def _extract_items_and_project(
        self,
        context: Any,
    ) -> tuple[Sequence[Any], Optional[str]]:
        """Extract ordered context items and project boundary identifier from context input.

        Args:
            context: AssembledContext, FormattedContext, Sequence of items, or compatible container.

        Returns:
            tuple[Sequence[Any], Optional[str]]: (items, project_id).

        Raises:
            ContextFormattingValidationError: If context is None or malformed.
        """
        if context is None:
            raise ContextFormattingValidationError("Context input cannot be None.")

        # Handle AssembledContext / StructuredRetrievalContext
        if hasattr(context, "items") and isinstance(context.items, Sequence):
            items = context.items
            project_id = getattr(context, "project_id", None)
            return items, str(project_id).strip() if project_id else None

        # Handle raw sequence of items (list, tuple)
        if isinstance(context, Sequence) and not isinstance(context, (str, bytes, dict)):
            project_id = None
            for item in context:
                pid = getattr(item, "project_id", None)
                if pid and isinstance(pid, str) and pid.strip():
                    project_id = pid.strip()
                    break
            return context, project_id

        raise ContextFormattingValidationError(
            f"Expected AssembledContext or Sequence of context items, got '{type(context).__name__}'."
        )

    def format(
        self,
        context: Any,
        project_id: Optional[str] = None,
        formatter: Optional[BaseContextFormatter] = None,
    ) -> FormattedContext:
        """Format authoritative retrieval context into a model-readable FormattedContext.

        Args:
            context: Final retrieval context (AssembledContext, Sequence of items, or FormattedContext).
            project_id: Optional project/tenant ID enforcing project boundary.
            formatter: Optional BaseContextFormatter override.

        Returns:
            FormattedContext: Model-readable formatted context container.

        Raises:
            ContextFormattingValidationError: If context fails validation or violates tenant boundary.
        """
        start_time = time.perf_counter()

        # If already formatted, validate project boundary and return
        if isinstance(context, FormattedContext):
            if project_id is not None and context.project_id is not None:
                if str(project_id).strip() != str(context.project_id).strip():
                    raise ContextFormattingValidationError(
                        f"Context project_id '{context.project_id}' violates requested project boundary '{project_id}'."
                    )
            return context

        # 1. Extract items and context project_id
        raw_items, ctx_project_id = self._extract_items_and_project(context)

        # 2. Resolve and validate tenant boundary
        eff_project_id: Optional[str] = None
        if project_id is not None:
            if not isinstance(project_id, str) or not project_id.strip():
                raise ContextFormattingValidationError("project_id must be a non-empty string when specified.")
            eff_project_id = project_id.strip()

        if eff_project_id is not None and ctx_project_id is not None and eff_project_id != ctx_project_id:
            raise ContextFormattingValidationError(
                f"Context project_id '{ctx_project_id}' violates requested project boundary '{eff_project_id}'."
            )

        resolved_project_id = eff_project_id or ctx_project_id

        # 3. Tenant isolation filtering on individual items if project_id is enforced
        filtered_items: list[Any] = []
        for idx, item in enumerate(raw_items, start=1):
            if item is None:
                raise ContextFormattingValidationError(f"Context item at index {idx} cannot be None.")

            item_pid = getattr(item, "project_id", None)
            if eff_project_id is not None and item_pid is not None:
                if str(item_pid).strip() != eff_project_id:
                    if self._config.strict_project_validation:
                        raise ContextFormattingValidationError(
                            f"Context item {idx} project_id '{item_pid}' violates requested boundary '{eff_project_id}'."
                        )
                    logger.warning(
                        "Discarding context item %d due to tenant mismatch (item='%s', boundary='%s')",
                        idx,
                        item_pid,
                        eff_project_id,
                    )
                    continue
            filtered_items.append(item)

        # 4. Execute active formatter strategy
        active_formatter = formatter or self._formatter
        result = active_formatter.format(items=filtered_items, project_id=resolved_project_id)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "Context formatting completed in %.2fms: project_id='%s', items=%d, characters=%d",
            elapsed_ms,
            resolved_project_id or "",
            result.item_count,
            len(result.text),
        )

        return result


def get_context_formatting_service(
    config: Optional[ContextFormattingConfig] = None,
    settings: Optional[Settings] = None,
    formatter: Optional[BaseContextFormatter] = None,
) -> ContextFormattingService:
    """Retrieve or initialize the singleton ContextFormattingService instance.

    Args:
        config: Optional ContextFormattingConfig override.
        settings: Optional Settings override.
        formatter: Optional BaseContextFormatter override.

    Returns:
        ContextFormattingService: Configured service instance.
    """
    global _context_formatting_service
    if _context_formatting_service is None or any(arg is not None for arg in (config, settings, formatter)):
        service = ContextFormattingService(
            config=config,
            settings=settings,
            formatter=formatter,
        )
        if all(arg is None for arg in (config, settings, formatter)):
            _context_formatting_service = service
        return service
    return _context_formatting_service


def reset_context_formatting_service() -> None:
    """Reset the cached singleton ContextFormattingService instance (useful for tests)."""
    global _context_formatting_service
    _context_formatting_service = None


def format_context(
    context: Any,
    project_id: Optional[str] = None,
    formatter: Optional[BaseContextFormatter] = None,
    service: Optional[ContextFormattingService] = None,
) -> FormattedContext:
    """Functional convenience entrypoint to format authoritative context into FormattedContext.

    Args:
        context: Authoritative retrieval context (AssembledContext or Sequence of items).
        project_id: Optional tenant isolation identifier.
        formatter: Optional BaseContextFormatter strategy override.
        service: Optional ContextFormattingService override.

    Returns:
        FormattedContext: Model-readable formatted context representation.
    """
    active_service = service or get_context_formatting_service()
    return active_service.format(
        context=context,
        project_id=project_id,
        formatter=formatter,
    )


async def format_context_async(
    context: Any,
    project_id: Optional[str] = None,
    formatter: Optional[BaseContextFormatter] = None,
    service: Optional[ContextFormattingService] = None,
) -> FormattedContext:
    """Async convenience wrapper for format_context in asynchronous RAG pipelines.

    Retains synchronous in-memory execution internally without artificial async delay.
    """
    return format_context(
        context=context,
        project_id=project_id,
        formatter=formatter,
        service=service,
    )
