"""Voyage AI embedding provider implementation."""

from typing import Any, Optional

import voyageai
import voyageai.error as voyage_errors

from core.config import Settings, get_settings
from observability.logging import get_logger
from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.client import get_async_voyage_client
from rag.embeddings.models import EmbeddingBatchResult
from exceptions.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingInputValidationError,
    EmbeddingRateLimitError,
    EmbeddingRequestError,
)

logger = get_logger(__name__)


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI implementation of the BaseEmbeddingProvider interface.
    
    Generates text embeddings using the official asynchronous Voyage AI Python SDK.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[voyageai.AsyncClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the Voyage embedding provider.
        
        Args:
            model: Optional model name override. Defaults to settings.EMBEDDING_MODEL.
            client: Pre-configured AsyncClient instance. If None, resolved from client provider.
            settings: Settings instance. Defaults to application settings.
        """
        self._settings = settings or get_settings()
        self._model = model or self._settings.EMBEDDING_MODEL
        if not self._model:
            raise EmbeddingConfigurationError(
                "Embedding model name must be provided either directly or via EMBEDDING_MODEL."
            )
        self._client = client

    @property
    def model_name(self) -> str:
        """Return the active embedding model name."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "voyage"

    # --------------------------------------------------------------------------
    # Synchronous Input Validation
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

    # --------------------------------------------------------------------------
    # Client Access
    # --------------------------------------------------------------------------

    def _get_client(self) -> voyageai.AsyncClient:
        """Resolve the active asynchronous Voyage AI client."""
        if self._client is not None:
            return self._client
        return get_async_voyage_client(self._settings)

    # --------------------------------------------------------------------------
    # Asynchronous Embedding Operations
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
        """Generate embedding vectors for a batch of text inputs preserving order."""
        valid_texts = self._validate_batch(texts)
        client = self._get_client()

        try:
            logger.debug(
                "Requesting Voyage embeddings for batch of %d items (model: '%s', input_type: %s)",
                len(valid_texts),
                self._model,
                input_type,
            )
            response = await client.embed(
                texts=valid_texts,
                model=self._model,
                input_type=input_type,
            )

            embeddings = getattr(response, "embeddings", None)
            if embeddings is None or len(embeddings) != len(valid_texts):
                raise EmbeddingError(
                    f"Mismatched embedding count: expected {len(valid_texts)}, got {len(embeddings) if embeddings else 0}"
                )

            total_tokens = getattr(response, "total_tokens", None)
            logger.debug(
                "Successfully generated %d embeddings (%s tokens consumed)",
                len(embeddings),
                total_tokens,
            )

            return EmbeddingBatchResult(
                embeddings=embeddings,
                model=self._model,
                total_tokens=total_tokens,
            )
        except voyage_errors.AuthenticationError as exc:
            logger.error("Voyage AI authentication failed")
            raise EmbeddingAuthenticationError(
                "Authentication failed with Voyage AI. Please verify VOYAGE_API_KEY.",
                original_error=exc,
            ) from exc
        except voyage_errors.RateLimitError as exc:
            logger.warning("Voyage AI rate limit exceeded")
            raise EmbeddingRateLimitError(
                "Voyage AI rate limit exceeded. Please retry with backoff.",
                original_error=exc,
            ) from exc
        except (voyage_errors.APIConnectionError, voyage_errors.Timeout) as exc:
            logger.error("Connection error communicating with Voyage AI API: %s", exc)
            raise EmbeddingConnectionError(
                f"Failed to connect to Voyage AI API: {exc}",
                original_error=exc,
            ) from exc
        except (voyage_errors.InvalidRequestError, voyage_errors.MalformedRequestError) as exc:
            logger.error("Invalid request sent to Voyage AI: %s", exc)
            raise EmbeddingRequestError(
                f"Invalid request to Voyage AI: {exc}",
                original_error=exc,
            ) from exc
        except voyage_errors.VoyageError as exc:
            logger.error("Voyage AI API error: %s", exc)
            raise EmbeddingError(
                f"Voyage AI error: {exc}",
                original_error=exc,
            ) from exc
        except (EmbeddingConfigurationError, EmbeddingInputValidationError):
            raise
        except Exception as exc:
            logger.error("Unexpected error generating embeddings: %s", exc)
            raise EmbeddingError(
                f"Unexpected error generating embeddings: {exc}",
                original_error=exc,
            ) from exc
