"""Domain models for the document parsing and extraction layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag.acquisition.formats import DocumentType


class ElementType(str, Enum):
    """Enumeration of structural element types identified during document parsing."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TEXT_BLOCK = "text_block"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST_ITEM = "list_item"


@dataclass(frozen=True)
class ParsedElement:
    """Represents a discrete, ordered structural element extracted from a document.

    Preserves content, document ordering, heading levels, and hierarchical relationships
    (parent identifier and section breadcrumbs) without altering or chunking the content.
    """

    element_id: str
    element_type: ElementType
    content: str
    order: int
    heading_level: int | None = None
    parent_id: str | None = None
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize parsed element to a dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content,
            "order": self.order,
            "heading_level": self.heading_level,
            "parent_id": self.parent_id,
            "section_path": list(self.section_path),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParsedDocument:
    """Standardized internal representation of a parsed document and its structure.

    Provides a clean structural intermediate representation holding ordered extracted
    elements, section hierarchies, and parser metadata for subsequent cleaning and chunking.

    IMPORTANT:
    This model explicitly does NOT contain cleaned/normalized text, text chunks,
    embeddings, or vector database identifiers.
    """

    document_id: str
    project_id: str
    document_type: DocumentType
    elements: list[ParsedElement] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    parser_metadata: dict[str, Any] = field(default_factory=dict)
    document_version_id: str | None = None

    @property
    def total_elements(self) -> int:
        """Return total count of extracted elements."""
        return len(self.elements)

    @property
    def headings(self) -> list[ParsedElement]:
        """Return only heading elements in document order."""
        return [e for e in self.elements if e.element_type == ElementType.HEADING]

    def get_elements_by_type(self, element_type: ElementType) -> list[ParsedElement]:
        """Filter extracted elements by specific element type."""
        return [e for e in self.elements if e.element_type == element_type]

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Concatenate content of all extracted elements using given separator."""
        return separator.join(e.content for e in self.elements if e.content)

    def __repr__(self) -> str:
        """Return safe string representation omitting raw extracted text contents.

        Prevents leaking document content or sensitive payloads into application logs.
        """
        return (
            f"ParsedDocument(document_id={self.document_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_type={self.document_type.value!r}, "
            f"total_elements={self.total_elements}, "
            f"document_version_id={self.document_version_id!r})"
        )

    def to_dict(self, include_elements: bool = True) -> dict[str, Any]:
        """Serialize parsed document metadata and optional elements to a dictionary.

        Args:
            include_elements: Whether to include the list of serialized elements.
        """
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "document_type": self.document_type.value,
            "document_version_id": self.document_version_id,
            "total_elements": self.total_elements,
            "source_metadata": dict(self.source_metadata),
            "parser_metadata": dict(self.parser_metadata),
        }
        if include_elements:
            data["elements"] = [elem.to_dict() for elem in self.elements]
        return data
