"""Data models representing embedding results."""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    """Vector representation and metadata for a single text input."""

    vector: list[float]
    model: str
    token_count: int | None = None

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.vector)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Ordered collection of embedding vectors generated for a batch of text inputs."""

    embeddings: list[list[float]]
    model: str
    total_tokens: int | None = None

    def __len__(self) -> int:
        """Return the number of embeddings in the batch."""
        return len(self.embeddings)

    def __iter__(self) -> Iterator[list[float]]:
        """Iterate over embedding vectors in input order."""
        return iter(self.embeddings)

    def __getitem__(self, index: int) -> list[float]:
        """Get embedding vector by index."""
        return self.embeddings[index]

    @property
    def dimension(self) -> int | None:
        """Return the dimensionality of the embedding vectors if non-empty."""
        return len(self.embeddings[0]) if self.embeddings else None
