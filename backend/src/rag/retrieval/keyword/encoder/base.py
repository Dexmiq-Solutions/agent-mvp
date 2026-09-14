"""Abstract base class defining the sparse representation encoder interface."""

from abc import ABC, abstractmethod

from rag.retrieval.models import SparseVector


class BaseSparseEncoder(ABC):
    """Abstract interface for sparse vector representations.

    Generates sparse vector representations (dimension indices and positive weights)
    for retrieval queries and document chunks. Implementations may use deterministic
    lexical tokenization, learned sparse neural models (e.g. SPLADE, BGE-M3),
    or external sparse representation services.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the unique strategy identifier for this sparse representation."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version identifier for this sparse representation mechanism."""

    @abstractmethod
    def encode_query(self, query: str) -> SparseVector:
        """Encode a single retrieval query into a SparseVector representation.

        Args:
            query: Verbatim retrieval query text.

        Returns:
            SparseVector: Sorted non-negative dimension indices and positive weights.

        Raises:
            SparseEncodingError: If input is invalid or encoding computation fails.
        """

    def encode_queries(self, queries: list[str]) -> list[SparseVector]:
        """Encode a batch of retrieval queries into SparseVector representations.

        Default implementation processes queries sequentially. Encoders supporting
        native vectorized batch operations should override this method.

        Args:
            queries: Sequence of retrieval query strings.

        Returns:
            list[SparseVector]: SparseVector representations corresponding 1-to-1 with inputs.

        Raises:
            SparseEncodingError: If input is invalid or encoding computation fails.
        """
        return [self.encode_query(q) for q in queries]

    @abstractmethod
    def encode_document(self, text: str) -> SparseVector:
        """Encode a single document chunk text into a SparseVector representation.

        Args:
            text: Document chunk textual content.

        Returns:
            SparseVector: Sorted non-negative dimension indices and positive weights.

        Raises:
            SparseEncodingError: If input is invalid or encoding computation fails.
        """

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode a batch of document chunk texts into SparseVector representations.

        Default implementation processes documents sequentially. Encoders supporting
        native vectorized batch operations should override this method.

        Args:
            texts: Sequence of document chunk textual content strings.

        Returns:
            list[SparseVector]: SparseVector representations corresponding 1-to-1 with inputs.

        Raises:
            SparseEncodingError: If input is invalid or encoding computation fails.
        """
        return [self.encode_document(t) for t in texts]
