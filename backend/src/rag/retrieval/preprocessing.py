"""Query Preprocessing stage of the Retrieval pipeline."""

import time
from typing import Any
import unicodedata

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from exceptions.retrieval import (
    EmptyQueryError,
    InvalidQueryError,
    QueryLengthExceededError,
    QueryPreprocessingError,
)
from rag.retrieval.config import QueryPreprocessingConfig
from rag.retrieval.models import ProcessedQuery

logger = get_logger(__name__)


class QueryPreprocessor:
    """Service executing deterministic, conservative query preprocessing for retrieval.

    Operates as the entry point to the Retrieval pipeline. Prepares raw user queries
    by validating inputs, normalizing Unicode representation (NFC), and collapsing
    insignificant whitespace while strictly preserving meaningful casing, punctuation,
    technical identifiers, error codes, and domain terminology without semantic alteration.
    """

    def __init__(
        self,
        config: QueryPreprocessingConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize QueryPreprocessor.

        Args:
            config: Optional preprocessing configuration override. If omitted,
                defaults are initialized from application settings.
            settings: Optional application settings. Defaults to cached app settings.
        """
        if config is not None:
            self._config = config
        else:
            resolved_settings = settings or get_settings()
            max_len = getattr(resolved_settings, "MAX_QUERY_LENGTH", 2000)
            self._config = QueryPreprocessingConfig(max_query_length=max_len)

    @property
    def config(self) -> QueryPreprocessingConfig:
        """Active preprocessing configuration."""
        return self._config

    def preprocess(self, query: Any) -> ProcessedQuery:
        """Preprocess a user query synchronously.

        Executes single-pass deterministic validation, Unicode normalization,
        and whitespace normalization, returning a ProcessedQuery holding both
        the verbatim raw query and the normalized query.

        Args:
            query: Raw user query string.

        Returns:
            ProcessedQuery containing original_query and processed_query.

        Raises:
            InvalidQueryError: If query is not a string.
            EmptyQueryError: If query is empty or contains only whitespace.
            QueryLengthExceededError: If query character length exceeds configured max limit.
            QueryPreprocessingError: If an unexpected error occurs during processing.
        """
        logger.debug("Starting query preprocessing: type=%s", type(query).__name__)

        # 1. Type Validation
        if not isinstance(query, str):
            msg = f"Query must be a string, got '{type(query).__name__}'."
            logger.warning("Query validation failed: %s", msg)
            raise InvalidQueryError(msg)

        # 2. Empty / Whitespace-Only Validation
        if not query or not query.strip():
            msg = "Query cannot be empty or whitespace only."
            logger.warning("Query validation failed: %s", msg)
            raise EmptyQueryError(msg)

        # 3. Maximum Query Length Validation
        query_len = len(query)
        if query_len > self._config.max_query_length:
            msg = (
                f"Query length ({query_len}) exceeds maximum allowed limit "
                f"({self._config.max_query_length})."
            )
            logger.warning("Query validation failed: %s", msg)
            raise QueryLengthExceededError(
                msg,
                length=query_len,
                max_length=self._config.max_query_length,
            )

        try:
            start_time = time.perf_counter()

            # 4. Safe Unicode Normalization (W3C NFC canonical composition)
            normalized = unicodedata.normalize(self._config.unicode_form, query)

            # 5. Insignificant Whitespace Normalization (Trim + Collapse multiple spaces/tabs/newlines)
            processed_text = " ".join(normalized.split())

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(
                "Query preprocessing completed: raw_len=%d, processed_len=%d, elapsed_ms=%.3f",
                query_len,
                len(processed_text),
                elapsed_ms,
            )

            # 6. Produce ProcessedQuery preserving verbatim original
            return ProcessedQuery(
                original_query=query,
                processed_query=processed_text,
            )
        except (InvalidQueryError, EmptyQueryError, QueryLengthExceededError):
            raise
        except Exception as exc:
            logger.error("Unexpected query preprocessing error: %s", exc, exc_info=True)
            raise QueryPreprocessingError(
                f"Query preprocessing failed unexpectedly: {exc}",
                original_error=exc,
            ) from exc


# Service alias maintaining naming consistency across pipeline stages
QueryPreprocessingService = QueryPreprocessor

_default_query_preprocessor: QueryPreprocessor | None = None


def get_query_preprocessor(
    config: QueryPreprocessingConfig | None = None,
    settings: Settings | None = None,
) -> QueryPreprocessor:
    """Get or create the default QueryPreprocessor singleton instance."""
    global _default_query_preprocessor
    if config is not None or settings is not None:
        return QueryPreprocessor(config=config, settings=settings)
    if _default_query_preprocessor is None:
        _default_query_preprocessor = QueryPreprocessor()
    return _default_query_preprocessor


def reset_query_preprocessor() -> None:
    """Reset the singleton query preprocessor instance (for test isolation)."""
    global _default_query_preprocessor
    _default_query_preprocessor = None


def preprocess_query(
    query: str,
    config: QueryPreprocessingConfig | None = None,
) -> ProcessedQuery:
    """Convenience function to preprocess a retrieval query using the active preprocessor."""
    return get_query_preprocessor(config=config).preprocess(query)
