"""Jina AI cross-encoder reranker provider implementation."""

import asyncio
import math
from typing import Any, Optional, Sequence

import httpx

from core.config import Settings, get_settings
from exceptions.retrieval import (
    RerankingConfigurationError,
    RerankingProviderError,
    RerankingTimeoutError,
    RerankingValidationError,
)
from observability.logging import get_logger
from rag.retrieval.reranking.base import BaseReranker, ScoredDocument

logger = get_logger(__name__)

DEFAULT_JINA_BASE_URL = "https://api.jina.ai/v1"
DEFAULT_JINA_MODEL = "jina-reranker-v3.5"


class JinaReranker(BaseReranker):
    """Jina AI concrete implementation of the BaseReranker interface.

    Executes cross-encoder reranking using Jina AI's REST API endpoint (/v1/rerank).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        base_url: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize JinaReranker.

        Args:
            model: Reranking model name. Defaults to settings.RERANKER_MODEL or 'jina-reranker-v3.5'.
            api_key: Jina AI API key. Defaults to settings.JINA_API_KEY.
            timeout_seconds: Network call timeout in seconds. Defaults to settings.RERANKER_TIMEOUT or 15.0.
            base_url: Base URL for Jina API. Defaults to 'https://api.jina.ai/v1'.
            client: Optional pre-configured httpx.AsyncClient instance.
            settings: Settings instance. Defaults to application settings.
        """
        self._settings = settings or get_settings()
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise RerankingConfigurationError("Reranking model name must be a non-empty string.")
            self._model = model.strip()
        else:
            raw_model = getattr(self._settings, "RERANKER_MODEL", DEFAULT_JINA_MODEL)
            if not isinstance(raw_model, str) or not raw_model.strip():
                raw_model = DEFAULT_JINA_MODEL
            self._model = raw_model.strip()

        self._api_key = api_key or getattr(self._settings, "JINA_API_KEY", None)

        timeout_val = timeout_seconds if timeout_seconds is not None else getattr(self._settings, "RERANKER_TIMEOUT", 15.0)
        if not isinstance(timeout_val, (int, float)) or timeout_val <= 0:
            raise RerankingConfigurationError(f"timeout_seconds must be a positive number, got {timeout_val}.")
        self._timeout_seconds = float(timeout_val)

        raw_base_url = base_url or getattr(self._settings, "JINA_BASE_URL", DEFAULT_JINA_BASE_URL) or DEFAULT_JINA_BASE_URL
        self._base_url = raw_base_url.rstrip("/")
        self._client = client

    @property
    def model_name(self) -> str:
        """Return the active reranker model name."""
        return self._model

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[ScoredDocument]:
        """Score candidate documents for relevance to the query via Jina AI.

        Args:
            query: Query text string.
            documents: Sequence of document chunk texts.
            top_k: Optional maximum number of scored results to return.

        Returns:
            list[ScoredDocument]: Scored documents preserving original input indices.

        Raises:
            RerankingValidationError: If inputs or provider output are malformed.
            RerankingConfigurationError: If JINA_API_KEY is not configured.
            RerankingTimeoutError: If external API call exceeds timeout.
            RerankingProviderError: If external Jina AI service fails.
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

        # 2. Check API Key
        api_key = self._api_key
        if not api_key or not str(api_key).strip():
            raise RerankingConfigurationError(
                "JINA_API_KEY is not configured. Cross-encoder reranking via Jina AI requires a valid API key."
            )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(api_key).strip()}",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": doc_list,
            "return_documents": False,
        }
        if effective_top_k is not None:
            payload["top_n"] = effective_top_k

        url = f"{self._base_url}/rerank"

        logger.debug(
            "Requesting Jina AI reranking: url='%s', model='%s', docs_count=%d, top_n=%s",
            url,
            self._model,
            len(doc_list),
            effective_top_k,
        )

        # 3. HTTP Request Execution with Timeout & Error Mapping
        try:
            if self._client is not None:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                    )

            if response.status_code in (401, 403):
                logger.error("Jina AI authentication failed: HTTP %d", response.status_code)
                raise RerankingProviderError(
                    "Authentication failed with Jina AI. Please verify JINA_API_KEY.",
                )

            if response.status_code == 429:
                logger.warning("Jina AI rate limit exceeded: HTTP 429")
                raise RerankingProviderError(
                    "Jina AI rate limit exceeded. Please retry with backoff.",
                )

            if response.status_code >= 500:
                logger.error("Jina AI server error: HTTP %d - %s", response.status_code, response.text[:200])
                raise RerankingProviderError(
                    f"Jina AI server error (HTTP {response.status_code}): {response.text[:200]}",
                )

            if response.status_code >= 400:
                logger.error("Jina AI client error: HTTP %d - %s", response.status_code, response.text[:200])
                raise RerankingProviderError(
                    f"Jina AI error (HTTP {response.status_code}): {response.text[:200]}",
                )

            try:
                data = response.json()
            except Exception as exc:
                raise RerankingValidationError(f"Jina response is not valid JSON: {exc}") from exc

        except httpx.TimeoutException as exc:
            logger.error("Jina AI reranking timed out after %.2fs", self._timeout_seconds)
            raise RerankingTimeoutError(
                f"Jina AI reranking request timed out after {self._timeout_seconds}s: {exc}",
                original_error=exc,
            ) from exc
        except (RerankingValidationError, RerankingConfigurationError, RerankingProviderError):
            raise
        except httpx.RequestError as exc:
            logger.error("Network error communicating with Jina AI: %s", exc)
            raise RerankingProviderError(
                f"Network error communicating with Jina AI: {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error communicating with Jina AI: %s", exc)
            raise RerankingProviderError(
                f"Unexpected reranker error: {exc}",
                original_error=exc,
            ) from exc

        # 4. Output Validation & Mapping
        if not isinstance(data, dict):
            raise RerankingValidationError("Jina response payload must be a JSON object.")

        results_data = data.get("results")
        if results_data is None or not isinstance(results_data, list):
            raise RerankingValidationError("Jina response missing or invalid 'results' array.")

        scored_docs: list[ScoredDocument] = []
        num_docs = len(doc_list)

        for item in results_data:
            if not isinstance(item, dict):
                raise RerankingValidationError(f"Jina result item must be a dictionary, got '{type(item).__name__}'.")

            idx = item.get("index")
            score = item.get("relevance_score")

            if not isinstance(idx, int) or idx < 0 or idx >= num_docs:
                raise RerankingValidationError(
                    f"Jina returned invalid candidate index '{idx}' for document batch of size {num_docs}."
                )

            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise RerankingValidationError(
                    f"Jina returned non-finite score '{score}' at index {idx}."
                )

            scored_docs.append(ScoredDocument(index=idx, score=float(score)))

        return scored_docs
