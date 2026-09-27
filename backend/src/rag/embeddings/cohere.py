"""Cohere embedding provider implementation using Cohere Embed v4."""

import asyncio
from typing import Any, Optional, Sequence

import cohere
import httpx

from core.config import Settings, get_settings
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
    EmbeddingResponseValidationError,
)
from observability.logging import get_logger
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.cohere_client import get_async_cohere_client
from rag.embeddings.models import EmbeddingBatchResult
from rag.embeddings.validation import validate_embedding_batch

logger = get_logger(__name__)

# Cohere Embed v4 supports up to 96 texts per request
DEFAULT_COHERE_MAX_BATCH_SIZE: int = 96
DEFAULT_COHERE_EMBEDDING_DIM: int = 1024


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere Embed v4 implementation of the BaseEmbeddingProvider interface.

    Generates dense text embeddings using the official asynchronous Cohere Python SDK.
    Handles input-type translation ('document' -> 'search_document', 'query' -> 'search_query'),
    batch slicing (max 96 items per call), and 1024-dimensional vector validation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[cohere.AsyncClient] = None,
        settings: Optional[Settings] = None,
        expected_dim: Optional[int] = DEFAULT_COHERE_EMBEDDING_DIM,
        max_batch_size: int = DEFAULT_COHERE_MAX_BATCH_SIZE,
    ) -> None:
        """Initialize the Cohere embedding provider.

        Args:
            model: Optional model name override. Defaults to settings.EMBEDDING_MODEL or 'embed-v4.0'.
            client: Pre-configured AsyncClient instance. If None, resolved from client provider.
            settings: Settings instance. Defaults to application settings.
            expected_dim: Expected output vector dimension for validation (default: 1024).
            max_batch_size: Maximum texts per Cohere API request (default: 96).
        """
        self._settings = settings or get_settings()
        self._model = model or getattr(self._settings, "EMBEDDING_MODEL", None) or "embed-v4.0"
        if not self._model:
            raise EmbeddingConfigurationError(
                "Embedding model name must be provided either directly or via EMBEDDING_MODEL."
            )
        self._client = client
        self._expected_dim = expected_dim
        self._max_batch_size = max(1, max_batch_size)

    @property
    def model_name(self) -> str:
        """Return the active embedding model name."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "cohere"

    @property
    def expected_dim(self) -> Optional[int]:
        """Return expected embedding vector dimension."""
        return self._expected_dim

    # --------------------------------------------------------------------------
    # Synchronous Input Validation and Mapping
    # --------------------------------------------------------------------------

    def _validate_text(self, text: Any) -> str:
        """Validate a single text string synchronously."""
        if not isinstance(text, str):
            raise EmbeddingInputValidationError(
                f"Input text must be a string, got {type(text).__name__}."
            )
        if not text or not text.strip():
            raise EmbeddingInputValidationError("Input text cannot be empty or whitespace only.")
        return text

    def _validate_batch(self, texts: Any) -> list[str]:
        """Validate a batch of text strings synchronously."""
        if not isinstance(texts, (list, tuple)):
            raise EmbeddingInputValidationError(
                f"Input texts must be a list or tuple of strings, got {type(texts).__name__}."
            )
        if len(texts) == 0:
            raise EmbeddingInputValidationError("Input texts batch cannot be empty.")

        validated: list[str] = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingInputValidationError(
                    f"Item at index {i} must be a string, got {type(text).__name__}."
                )
            if not text or not text.strip():
                raise EmbeddingInputValidationError(
                    f"Item at index {i} cannot be empty or whitespace only."
                )
            validated.append(text)
        return validated

    def _map_input_type(self, input_type: Optional[str]) -> str:
        """Map generic application input types to Cohere-specific input_type parameters.

        Translates:
            'document'        -> 'search_document'
            'search_document' -> 'search_document'
            'query'           -> 'search_query'
            'search_query'    -> 'search_query'
            default (None)    -> 'search_document'
        """
        if not input_type:
            return "search_document"
        norm = input_type.lower().strip()
        if norm in ("document", "search_document"):
            return "search_document"
        if norm in ("query", "search_query"):
            return "search_query"
        if norm in ("classification", "clustering", "image"):
            return norm
        logger.debug("Unrecognized input_type '%s', defaulting to 'search_document'", input_type)
        return "search_document"

    # --------------------------------------------------------------------------
    # Client Access
    # --------------------------------------------------------------------------

    def _get_client(self) -> cohere.AsyncClient:
        """Resolve the active asynchronous Cohere client."""
        if self._client is not None:
            return self._client
        return get_async_cohere_client(self._settings)

    # --------------------------------------------------------------------------
    # Sub-batch Execution
    # --------------------------------------------------------------------------

    async def _embed_single_slice(
        self,
        client: cohere.AsyncClient,
        slice_texts: list[str],
        mapped_input_type: str,
        slice_idx: int,
        total_slices: int,
    ) -> tuple[list[list[float]], Optional[int]]:
        """Execute a single bounded API request against Cohere embed API."""
        try:
            logger.debug(
                "Requesting Cohere embeddings: slice %d/%d (%d items, model: '%s', input_type: '%s')",
                slice_idx,
                total_slices,
                len(slice_texts),
                self._model,
                mapped_input_type,
            )
            response = await client.embed(
                texts=slice_texts,
                model=self._model,
                input_type=mapped_input_type,
                embedding_types=["float"],
            )

            raw_embeddings = getattr(response, "embeddings", None)
            if hasattr(raw_embeddings, "float") and raw_embeddings.float is not None:
                embeddings = raw_embeddings.float
            elif isinstance(raw_embeddings, list):
                embeddings = raw_embeddings
            else:
                raise EmbeddingResponseValidationError(
                    f"Unexpected response structure from Cohere: {type(raw_embeddings).__name__}"
                )

            # Validate returned vectors and dimensions
            validated = validate_embedding_batch(
                embeddings,
                expected_count=len(slice_texts),
                expected_dim=self._expected_dim,
            )

            # Extract token usage if provided in response metadata
            tokens_consumed: Optional[int] = None
            meta = getattr(response, "meta", None)
            if meta is not None:
                tokens_obj = getattr(meta, "tokens", None)
                if tokens_obj is not None:
                    tokens_consumed = getattr(tokens_obj, "input_tokens", None)
                if tokens_consumed is None:
                    billed_obj = getattr(meta, "billed_units", None)
                    if billed_obj is not None:
                        tokens_consumed = getattr(billed_obj, "input_tokens", None)

            return validated, tokens_consumed

        except (cohere.UnauthorizedError, cohere.InvalidTokenError, cohere.ForbiddenError) as exc:
            logger.error("Cohere authentication failed")
            raise EmbeddingAuthenticationError(
                "Authentication failed with Cohere. Please verify COHERE_API_KEY.",
                original_error=exc,
            ) from exc
        except cohere.TooManyRequestsError as exc:
            logger.warning("Cohere rate limit exceeded")
            raise EmbeddingRateLimitError(
                "Cohere rate limit exceeded. Please retry with backoff.",
                original_error=exc,
            ) from exc
        except (
            cohere.GatewayTimeoutError,
            cohere.ServiceUnavailableError,
            cohere.ClientClosedRequestError,
            httpx.TimeoutException,
            httpx.ConnectError,
            asyncio.TimeoutError,
        ) as exc:
            logger.error("Connection error communicating with Cohere API: %s", exc)
            raise EmbeddingConnectionError(
                f"Failed to connect to Cohere API: {exc}",
                original_error=exc,
            ) from exc
        except (cohere.BadRequestError, cohere.UnprocessableEntityError) as exc:
            logger.error("Invalid request sent to Cohere: %s", exc)
            raise EmbeddingRequestError(
                f"Invalid request to Cohere: {exc}",
                original_error=exc,
            ) from exc
        except cohere.core.api_error.ApiError as exc:
            logger.error("Cohere API error: %s", exc)
            raise EmbeddingError(
                f"Cohere API error: {exc}",
                original_error=exc,
            ) from exc
        except (
            EmbeddingConfigurationError,
            EmbeddingInputValidationError,
            EmbeddingResponseValidationError,
        ):
            raise
        except Exception as exc:
            logger.error("Unexpected error generating Cohere embeddings: %s", exc)
            raise EmbeddingError(
                f"Unexpected error generating Cohere embeddings: {exc}",
                original_error=exc,
            ) from exc

    # --------------------------------------------------------------------------
    # Public Asynchronous Operations
    # --------------------------------------------------------------------------

    async def embed_text(
        self,
        text: str,
        input_type: Optional[str] = None,
    ) -> list[float]:
        """Generate an embedding vector for a single text input."""
        valid_text = self._validate_text(text)
        batch_result = await self.embed_batch([valid_text], input_type=input_type)
        return batch_result.embeddings[0]

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single retrieval query."""
        return await self.embed_text(query, input_type="query")

    async def embed_queries(self, queries: list[str]) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of retrieval queries preserving order."""
        return await self.embed_batch(queries, input_type="query")

    async def embed_batch(
        self,
        texts: list[str],
        input_type: Optional[str] = None,
    ) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of text inputs preserving order.

        Automatically splits batches exceeding max_batch_size (96) into sequential
        sub-batches to comply with Cohere API limits, preserving 1-to-1 input ordering.
        """
        valid_texts = self._validate_batch(texts)
        mapped_input_type = self._map_input_type(input_type)
        client = self._get_client()

        total_items = len(valid_texts)
        total_slices = (total_items + self._max_batch_size - 1) // self._max_batch_size

        all_embeddings: list[list[float]] = []
        total_tokens_consumed: Optional[int] = None

        for slice_idx in range(total_slices):
            start = slice_idx * self._max_batch_size
            end = min(start + self._max_batch_size, total_items)
            slice_texts = valid_texts[start:end]

            slice_embeddings, slice_tokens = await self._embed_single_slice(
                client=client,
                slice_texts=slice_texts,
                mapped_input_type=mapped_input_type,
                slice_idx=slice_idx + 1,
                total_slices=total_slices,
            )

            all_embeddings.extend(slice_embeddings)
            if slice_tokens is not None:
                total_tokens_consumed = (total_tokens_consumed or 0) + slice_tokens

        logger.info(
            "Successfully generated %d Cohere embeddings (%s tokens consumed, slices: %d)",
            len(all_embeddings),
            total_tokens_consumed,
            total_slices,
        )

        return EmbeddingBatchResult(
            embeddings=all_embeddings,
            model=self._model,
            total_tokens=total_tokens_consumed,
        )
