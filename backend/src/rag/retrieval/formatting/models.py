"""Domain models for the Context Formatting stage of the RAG retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FormattedContextItem:
    """Represents an individual formatted context item.

    Captures the verbatim retrieved content alongside structured, model-facing
    metadata and hierarchy, cleanly formatted for downstream consumption.
    """

    index: int
    text: str
    content: str
    chunk_id: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    source: str | None = None
    heading: str | None = None
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize formatted context item to a dictionary."""
        return {
            "index": self.index,
            "text": self.text,
            "content": self.content,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "source": self.source,
            "heading": self.heading,
            "section_path": list(self.section_path),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FormattedContext:
    """Structured, model-readable representation of retrieved context.

    Positioned between Final Context Assembly (retrieval) and the application/Agent boundary.
    Preserves exact retrieval ordering, source provenance, and content integrity,
    while cleanly transforming raw context items into an unambiguous, model-readable
    textual format.
    """

    text: str
    items: tuple[FormattedContextItem, ...] = ()
    item_count: int = 0
    project_id: str | None = None
    strategy: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return True if the formatted context contains zero items."""
        return self.item_count == 0

    def __str__(self) -> str:
        """Return the formatted context text for direct string evaluation."""
        return self.text

    def __len__(self) -> int:
        """Return the number of formatted context items."""
        return self.item_count

    def __iter__(self):
        """Iterate over formatted context items in preserved order."""
        return iter(self.items)

    def __getitem__(self, idx: int) -> FormattedContextItem:
        """Access formatted item by 0-based index."""
        return self.items[idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize formatted context to a standard dictionary representation."""
        return {
            "text": self.text,
            "item_count": self.item_count,
            "project_id": self.project_id,
            "strategy": self.strategy,
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe debug representation avoiding dumping huge context into logs."""
        return (
            f"FormattedContext(project_id={self.project_id!r}, "
            f"item_count={self.item_count}, "
            f"strategy={self.strategy!r}, "
            f"characters={len(self.text)})"
        )
