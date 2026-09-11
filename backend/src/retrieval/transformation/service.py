"""Query Transformation service orchestrating adaptive retrieval query transformation."""

import time
from typing import Any, Optional, Union

from app.core.config import Settings
from app.core.logging import get_logger
from exceptions.retrieval import (
    TransformationProviderError,
    TransformationTimeoutError,
    TransformationUnavailableError,
    TransformationValidationError,
)
from retrieval.config import QueryTransformationConfig
from retrieval.models import ProcessedQuery, RetrievalQuerySet
from retrieval.preprocessing import preprocess_query
from retrieval.transformation.policy import (
    AdaptiveTransformationPolicy,
    BaseTransformationPolicy,
)
from retrieval.transformation.provider import (
    BaseLLMClient,
    OpenAICompatibleLLMClient,
)
from retrieval.transformation.strategy import (
    BaseTransformationStrategy,
    LLMQueryRewriteStrategy,
)
from retrieval.transformation.validator import TransformationOutputValidator

logger = get_logger(__name__)


class QueryTransformationService:
    """Service orchestrating adaptive query transformation in the retrieval pipeline.

    Evaluates whether a preprocessed query is context-dependent or an insufficient
    retrieval fallback. If transformation is warranted, rewrites the query using an
    LLM and non-destructively outputs a RetrievalQuerySet containing both the original
    and transformed query representations. Self-contained queries pass through with
    0ms LLM overhead.
    """

    def __init__(
        self,
        config: Optional[QueryTransformationConfig] = None,
        settings: Optional[Settings] = None,
        policy: Optional[BaseTransformationPolicy] = None,
        strategy: Optional[BaseTransformationStrategy] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        """Initialize QueryTransformationService with single-source configuration.

        Args:
            config: Optional explicit transformation configuration override.
            settings: Optional application settings override.
            policy: Optional transformation policy override.
            strategy: Optional transformation strategy override.
            llm_client: Optional LLM client provider override.
        """
        if config is not None:
            self._config = config
        else:
            self._config = QueryTransformationConfig.from_settings(settings)

        self._policy = policy or AdaptiveTransformationPolicy()

        if strategy is not None:
            self._strategy = strategy
        else:
            client = llm_client or OpenAICompatibleLLMClient(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                default_model=self._config.model,
                default_timeout=self._config.timeout_seconds,
                max_retries=self._config.max_retries,
                retry_delay=self._config.retry_delay,
            )
            validator = TransformationOutputValidator(
                max_query_length=self._config.max_query_length,
                preserve_identifiers=self._config.preserve_identifiers,
            )
            self._strategy = LLMQueryRewriteStrategy(
                llm_client=client,
                model=self._config.model,
                validator=validator,
                temperature=self._config.temperature,
                timeout=self._config.timeout_seconds,
            )

    @property
    def config(self) -> QueryTransformationConfig:
        """Active query transformation configuration."""
        return self._config

    @property
    def policy(self) -> BaseTransformationPolicy:
        """Active transformation policy."""
        return self._policy

    @property
    def strategy(self) -> BaseTransformationStrategy:
        """Active transformation strategy."""
        return self._strategy

    async def transform(
        self,
        query: Union[ProcessedQuery, str],
        conversation_context: Optional[Any] = None,
        is_fallback: bool = False,
        attempt: int = 1,
        project_id: Optional[str] = None,
    ) -> RetrievalQuerySet:
        """Transform a preprocessed query adaptively into a RetrievalQuerySet.

        Args:
            query: Preprocessed user query (ProcessedQuery) or raw string to preprocess.
            conversation_context: Optional conversation context to resolve references.
            is_fallback: True if re-entering from an insufficient retrieval fallback.
            attempt: Current retrieval attempt number (for bounding fallbacks).
            project_id: Optional project identifier for auditing / isolation tracking.

        Returns:
            RetrievalQuerySet: Downstream retrieval contract containing original query
                and optionally the transformed query representation.
        """
        start_time = time.perf_counter()

        # 1. Normalize input to ProcessedQuery
        if isinstance(query, str):
            processed_query = preprocess_query(query)
        elif isinstance(query, ProcessedQuery):
            processed_query = query
        else:
            processed_query = preprocess_query(str(query))

        original = processed_query.processed_query

        # 2. Check if transformation is globally disabled via configuration
        if not self._config.enabled:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(
                "Query transformation disabled: passing through original query (%.2fms)",
                elapsed_ms,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=None,
                metadata={
                    "reason": "disabled",
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )

        # 3. Enforce bounded fallback attempts
        max_allowed_attempts = 1 + self._config.max_fallback_attempts
        if is_fallback and attempt > max_allowed_attempts:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "Query transformation fallback bound reached (attempt=%d > max=%d). "
                "Retaining original query to prevent infinite loops.",
                attempt,
                max_allowed_attempts,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=None,
                metadata={
                    "reason": "fallback_limit_reached",
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )

        # 4. Evaluate transformation policy
        decision = self._policy.evaluate(
            query=processed_query,
            conversation_context=conversation_context,
            is_fallback=is_fallback,
        )

        if not decision.should_transform:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(
                "Query transformation skipped by policy: reason=%s, elapsed_ms=%.2f",
                decision.reason,
                elapsed_ms,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=None,
                metadata={
                    "reason": decision.reason,
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )

        # 5. Execute transformation strategy with resilient fallback
        try:
            transformed = await self._strategy.transform(
                query=processed_query,
                conversation_context=conversation_context,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if transformed and transformed != original:
                logger.info(
                    "Query transformed successfully: reason=%s, strategy=%s, elapsed_ms=%.2f, attempt=%d, project_id=%s",
                    decision.reason,
                    self._strategy.strategy_name,
                    elapsed_ms,
                    attempt,
                    project_id,
                )
                return RetrievalQuerySet(
                    original_query=original,
                    transformed_query=transformed,
                    is_transformed=True,
                    strategy_used=self._strategy.strategy_name,
                    metadata={
                        "reason": decision.reason,
                        "is_fallback": is_fallback,
                        "project_id": project_id,
                    },
                )

            # Rewrite produced identical text to original
            logger.info(
                "Query rewrite resulted in identical text; retaining original query (%.2fms)",
                elapsed_ms,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=self._strategy.strategy_name,
                metadata={
                    "reason": "identical_to_original",
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )

        except (
            TransformationProviderError,
            TransformationValidationError,
            TransformationTimeoutError,
            TransformationUnavailableError,
        ) as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "Query transformation failed (%s: %s); falling back to original query (%.2fms)",
                type(exc).__name__,
                exc,
                elapsed_ms,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=self._strategy.strategy_name,
                metadata={
                    "reason": "transformation_failed",
                    "error": str(exc),
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "Unexpected error during query transformation: %s; falling back to original query (%.2fms)",
                exc,
                elapsed_ms,
                exc_info=True,
            )
            return RetrievalQuerySet(
                original_query=original,
                transformed_query=None,
                is_transformed=False,
                strategy_used=self._strategy.strategy_name,
                metadata={
                    "reason": "unexpected_error",
                    "error": str(exc),
                    "is_fallback": is_fallback,
                    "project_id": project_id,
                },
            )


_default_transformation_service: Optional[QueryTransformationService] = None


def get_query_transformation_service(
    config: Optional[QueryTransformationConfig] = None,
    settings: Optional[Settings] = None,
    policy: Optional[BaseTransformationPolicy] = None,
    strategy: Optional[BaseTransformationStrategy] = None,
    llm_client: Optional[BaseLLMClient] = None,
) -> QueryTransformationService:
    """Get or create singleton QueryTransformationService instance."""
    global _default_transformation_service

    if any(arg is not None for arg in (config, settings, policy, strategy, llm_client)):
        return QueryTransformationService(
            config=config,
            settings=settings,
            policy=policy,
            strategy=strategy,
            llm_client=llm_client,
        )

    if _default_transformation_service is None:
        _default_transformation_service = QueryTransformationService()
    return _default_transformation_service


def reset_query_transformation_service() -> None:
    """Reset singleton instance (useful for test isolation)."""
    global _default_transformation_service
    _default_transformation_service = None


async def transform_query(
    query: Union[ProcessedQuery, str],
    conversation_context: Optional[Any] = None,
    is_fallback: bool = False,
    attempt: int = 1,
    project_id: Optional[str] = None,
    config: Optional[QueryTransformationConfig] = None,
) -> RetrievalQuerySet:
    """Convenience async function to transform a query using the active transformation service."""
    service = get_query_transformation_service(config=config)
    return await service.transform(
        query=query,
        conversation_context=conversation_context,
        is_fallback=is_fallback,
        attempt=attempt,
        project_id=project_id,
    )
