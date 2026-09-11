"""Domain models for the retrieval and query preprocessing layer."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessedQuery:
    """Structured representation of a preprocessed retrieval query.

    Preserves the verbatim raw query alongside the conservatively normalized
    representation for downstream retrieval, tracing, and future transformation.
    """

    original_query: str
    processed_query: str

    @property
    def is_changed(self) -> bool:
        """Return True if preprocessing modified the query representation."""
        return self.original_query != self.processed_query

    def to_dict(self) -> dict[str, Any]:
        """Serialize the processed query to a standard dictionary."""
        return {
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "is_changed": self.is_changed,
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return (
            f"ProcessedQuery(original_query={self.original_query!r}, "
            f"processed_query={self.processed_query!r})"
        )
