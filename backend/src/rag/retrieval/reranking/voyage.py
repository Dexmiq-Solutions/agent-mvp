"""Voyage AI cross-encoder reranker provider implementation."""

import asyncio
import math
from typing import Any, Optional, Sequence

import voyageai
import voyageai.error as voyage_errors

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from rag.embeddings.client import get_async_voyage_client
from exceptions.retrieval import (
    RerankingConfigurationError,
    RerankingProviderError,
    RerankingTimeoutError,
    RerankingValidationError,
)
from rag.retrieval.reranking.base import BaseReranker, ScoredDocument

logger = get_logger(__name__)


class VoyageReranker(BaseReranker):
    """Voyage AI concrete implementation of the BaseReranker interface.

    Executes cross-encoder reranking using Voyage AI's asynchronous Python SDK.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[voyageai.AsyncClient] = None,
        timeout_seconds: Optional[float] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize VoyageReranker.

        Args:
            model: Reranking model name. Defaults to settings.RERANKER_MODEL or 'rerank-2.5'.
            client: Pre-configured AsyncClient instance. If None, resolved from client provider.
            timeout_seconds: Network call timeout in seconds. Defaults to settings.RERANKER_TIMEOUT or 10.0.
            settings: Settings instance. Defaults to application settings.
        """
        self._settings = settings or get_settings()
        self._model = model or getattr(self._settings, "RERANKER_MODEL", "rerank-2.5")
        if not self._model or not self._model.strip():
            raise RerankingConfigurationError("Reranking model name must be a non-empty string.")

        timeout_val = timeout_seconds if timeout_seconds is not None else getattr(self._settings, "RERANKER_TIMEOUT", 10.0)
        if not isinstance(timeout_val, (int, float)) or timeout_val <= 0:
            raise RerankingConfigurationError(f"timeout_seconds must be a positive number, got {timeout_val}.")
        self._timeout_seconds = float(timeout_val)

        self._client = client

    @property
    def model_name(self) -> str:
        """Return the active reranker model name."""
        return self._model

    def _get_client(self) -> voyageai.AsyncClient:
        """Resolve the active asynchronous Voyage AI client."""
        if self._client is not None:
            return self._client
        return get_async_voyage_client(self._settings)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[ScoredDocument]:
        """Score candidate documents for relevance to the query via Voyage AI.

        Args:
            query: Query text string.
            documents: Sequence of document chunk texts.
            top_k: Optional maximum number of scored results to return.

        Returns:
            list[ScoredDocument]: Scored documents preserving original input indices.

        Raises:
            RerankingValidationError: If inputs or provider output are malformed.
            RerankingTimeoutError: If external API call exceeds timeout.
            RerankingProviderError: If external Voyage AI service fails.
        """
        # 1. Input Validation
        if not isinstance(query, str) or not query.strip():
            raise RerankingValidationError("query must be a non-empty string.")

        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise RerankingValidationError(
                f"documents must be a Sequence of strings, got '{type(documents).__name__}'."
            )

        if len(documents) == 0:
            return []

        doc_list: list[str] = []
        for idx, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise RerankingValidationError(
                    f"Document at index {idx} must be a string, got '{type(doc).__name__}'."
                )
            doc_list.append(doc)

        effective_top_k: Optional[int] = None
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise RerankingValidationError(f"top_k must be a positive integer, got {top_k}.")
            effective_top_k = top_k

        client = self._get_client()

        # 2. Batched External Call with Timeout
        try:
            logger.debug(
                "Requesting Voyage reranking: model='%s', docs_count=%d, top_k=%s",
                self._model,
                len(doc_list),
                effective_top_k,
            )

            rerank_coro = client.rerank(
                query=query,
                documents=doc_list,
                model=self._model,
                top_k=effective_top_k,
                truncation=True,
            )

            response = await asyncio.wait_for(rerank_coro, timeout=self._timeout_seconds)

        except (asyncio.TimeoutError, voyage_errors.Timeout) as exc:
            logger.error("Voyage AI reranking timed out after %.2fs", self._timeout_seconds)
            raise RerankingTimeoutError(
                f"Voyage AI reranking request timed out after {self._timeout_seconds}s: {exc}",
                original_error=exc,
            ) from exc
        except voyage_errors.AuthenticationError as exc:
            logger.error("Voyage AI authentication failed during reranking")
            raise RerankingProviderError(
                "Authentication failed with Voyage AI. Please verify VOYAGE_API_KEY.",
                original_error=exc,
            ) from exc
        except voyage_errors.RateLimitError as exc:
            logger.warning("Voyage AI rate limit exceeded during reranking")
            raise RerankingProviderError(
                "Voyage AI rate limit exceeded. Please retry with backoff.",
                original_error=exc,
            ) from exc
        except (voyage_errors.APIConnectionError, voyage_errors.InvalidRequestError, voyage_errors.VoyageError) as exc:
            logger.error("Voyage AI error during reranking: %s", exc)
            raise RerankingProviderError(
                f"Voyage AI error: {exc}",
                original_error=exc,
            ) from exc
        except (RerankingValidationError, RerankingConfigurationError):
            raise
        except Exception as exc:
            logger.error("Unexpected error communicating with Voyage AI reranker: %s", exc)
            raise RerankingProviderError(
                f"Unexpected reranker error: {exc}",
                original_error=exc,
            ) from exc

        # 3. Output Validation & Mapping
        results_data = getattr(response, "results", None)
        if results_data is None:
            raise RerankingValidationError("Voyage response missing 'results' collection.")

        scored_docs: list[ScoredDocument] = []
        num_docs = len(doc_list)

        for item in results_data:
            idx = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)

            if not isinstance(idx, int) or idx < 0 or idx >= num_docs:
                raise RerankingValidationError(
                    f"Voyage returned invalid candidate index '{idx}' for document batch of size {num_docs}."
                )

            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise RerankingValidationError(
                    f"Voyage returned non-finite score '{score}' at index {idx}."
                )

            scored_docs.append(ScoredDocument(index=idx, score=float(score)))

        return scored_docs
