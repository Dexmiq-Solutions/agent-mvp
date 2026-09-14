"""Abstract base class defining the embedding provider interface."""

from abc import ABC, abstractmethod
from typing import Optional

from rag.embeddings.models import EmbeddingBatchResult


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding providers.
    
    Decouples the application, document ingestion, and retrieval pipelines from specific
    embedding vendor SDKs (such as Voyage AI).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the active embedding model."""

    @abstractmethod
    async def embed_text(
        self,
        text: str,
        input_type: Optional[str] = None,
    ) -> list[float]:
        """Generate an embedding vector for a single text string.
        
        Args:
            text: Input text string to embed.
            input_type: Optional input type hint (e.g., 'document' or 'query').
            
        Returns:
            list[float]: Embedding vector representing the input text.
            
        Raises:
            EmbeddingInputValidationError: If input text is empty or invalid.
            EmbeddingError: If embedding generation fails.
        """

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        input_type: Optional[str] = None,
    ) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of text inputs preserving order.
        
        Args:
            texts: List of input text strings to embed.
            input_type: Optional input type hint (e.g., 'document' or 'query').
            
        Returns:
            EmbeddingBatchResult: Result containing embedding vectors in 1-to-1 order.
            
        Raises:
            EmbeddingInputValidationError: If input list is empty or contains invalid items.
            EmbeddingError: If embedding generation fails.
        """

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single retrieval query.
        
        Args:
            query: Input query text string to embed.
            
        Returns:
            list[float]: Embedding vector representing the retrieval query.
            
        Raises:
            EmbeddingInputValidationError: If input query is empty or invalid.
            EmbeddingError: If embedding generation fails.
        """
        return await self.embed_text(query, input_type="query")

    async def embed_queries(self, queries: list[str]) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of retrieval queries preserving order.
        
        Args:
            queries: List of input query text strings to embed.
            
        Returns:
            EmbeddingBatchResult: Result containing embedding vectors in 1-to-1 order.
            
        Raises:
            EmbeddingInputValidationError: If queries list is empty or contains invalid items.
            EmbeddingError: If embedding generation fails.
        """
        return await self.embed_batch(queries, input_type="query")
