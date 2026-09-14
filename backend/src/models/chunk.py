"""SQLAlchemy entity model for stored document chunks in the PostgreSQL Content Store."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from db.base import Base

if TYPE_CHECKING:
    from models.document import DocumentModel, DocumentVersionModel
    from models.project import ProjectModel


class ChunkModel(Base):
    """Authoritative stored chunk record in PostgreSQL Content Store.

    Stores the verbatim chunk text, optional enriched contextual representation,
    structural hierarchy coordinates, and metadata required for downstream Context Assembly.
    """

    __tablename__ = "chunks"

    # Identity and Relational Coordinates
    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_version_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Core Text and Contextual Representations
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Hierarchy and Structural Positioning
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    heading: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    heading_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_path: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)
    parent_element_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    element_types: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)

    # Enriched Metadata (mapped to 'metadata' column in SQL)
    chunk_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSON, nullable=True, default=dict)

    # Timestamps
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_chunks_project_id_chunk_id", "project_id", "chunk_id"),
        Index("ix_chunks_project_id_document_id", "project_id", "document_id"),
    )

    # Relational Hierarchies
    document_version: Mapped[Optional["DocumentVersionModel"]] = relationship(
        "DocumentVersionModel",
        back_populates="chunks",
    )
    document: Mapped[Optional["DocumentModel"]] = relationship(
        "DocumentModel",
        foreign_keys=[document_id],
    )
    project: Mapped[Optional["ProjectModel"]] = relationship(
        "ProjectModel",
        foreign_keys=[project_id],
    )

    @property
    def text(self) -> str:
        """Alias for content returning raw chunk text."""
        return self.content

    @property
    def chunk_text(self) -> str:
        """Alias for content returning raw chunk text."""
        return self.content

    @property
    def index(self) -> int:
        """Alias for chunk_index."""
        return self.chunk_index

    @property
    def meta(self) -> dict[str, Any]:
        """Convenience property returning chunk_metadata dictionary."""
        return self.chunk_metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk model to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "content": self.content,
            "chunk_text": self.content,
            "contextual_content": self.contextual_content,
            "chunk_index": self.chunk_index,
            "index": self.chunk_index,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "section_path": list(self.section_path or []),
            "parent_element_id": self.parent_element_id,
            "parent_chunk_id": self.parent_chunk_id,
            "element_types": list(self.element_types or []),
            "metadata": dict(self.chunk_metadata or {}),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """Safe representation omitting full chunk content."""
        return (
            f"ChunkModel(chunk_id={self.chunk_id!r}, "
            f"project_id={self.project_id!r}, "
            f"document_id={self.document_id!r}, "
            f"chunk_index={self.chunk_index}, "
            f"content_length={len(self.content)})"
        )
