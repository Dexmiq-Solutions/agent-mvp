"""Chunk sizing abstractions and implementations.

Provides an extensible measurement strategy decoupled from specific tokenizers
or model providers, ensuring seamless evolution from deterministic character/word
measurement to model-specific tokenizers without service refactoring.
"""

from abc import ABC, abstractmethod


class BaseChunkSizer(ABC):
    """Abstract base class defining the sizing contract for document chunking."""

    @abstractmethod
    def measure(self, text: str) -> int:
        """Measure the size of the given text according to the sizing strategy.

        Args:
            text: Input string to measure.

        Returns:
            Integer measurement (characters, words, tokens, etc.).
        """
        pass

    def fits(self, text: str, max_size: int) -> bool:
        """Check whether the given text satisfies the maximum size constraint.

        Args:
            text: Input string to check.
            max_size: Maximum allowed size constraint.

        Returns:
            True if measured size is less than or equal to max_size, False otherwise.
        """
        return self.measure(text) <= max_size


class CharacterChunkSizer(BaseChunkSizer):
    """Measures chunk size by character count (Python string length).

    Provides fast, deterministic, zero-dependency sizing suitable for production
    RAG pipelines with predictable character boundaries.
    """

    def measure(self, text: str) -> int:
        """Return the length of the string in characters."""
        return len(text)

    def __repr__(self) -> str:
        return "CharacterChunkSizer()"


class WordChunkSizer(BaseChunkSizer):
    """Measures chunk size by whitespace-delimited word count.

    Useful when chunks are specified in word counts (~0.75 tokens/word).
    """

    def measure(self, text: str) -> int:
        """Return the number of whitespace-delimited words."""
        return len(text.split())

    def __repr__(self) -> str:
        return "WordChunkSizer()"
