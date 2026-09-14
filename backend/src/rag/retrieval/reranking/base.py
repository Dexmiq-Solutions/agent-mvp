"""Base abstractions for cross-encoder rerankers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ScoredDocument:
    """Scored document returned by a cross-encoder reranker.

    Attributes:
        index: Zero-based index corresponding to the input documents sequence position.
        score: Relevance score produced by the cross-encoder model.
    """

    index: int
    score: float


class BaseReranker(ABC):
    """Minimal abstract interface for cross-encoder reranking providers.

    Decouples the retrieval pipeline from any concrete reranker provider SDK (e.g. Voyage AI,
    Cohere, local cross-encoders).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active reranker model name."""
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[ScoredDocument]:
        """Score candidate documents for relevance to the query.

        Args:
            query: Non-empty query string.
            documents: Ordered sequence of candidate document chunk text strings.
            top_k: Optional maximum number of scored results to return.

        Returns:
            list[ScoredDocument]: Scored documents preserving original input indices.
        """
        ...
