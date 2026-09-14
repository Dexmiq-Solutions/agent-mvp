"""SQLAlchemy entity models for Documents (Sources) and Document Versions.

A Document represents a user-facing source belonging to a Project.
A DocumentVersion represents a concrete immutable/re-indexed version of that file.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import uuid

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from models.chunk import ChunkModel
    from models.project import ProjectModel


class DocumentVersionStatus(str, Enum):
    """Indexing lifecycle status for a document version."""

    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class DocumentModel(Base):
    """Authoritative persistent representation of a user-facing Source document."""

    __tablename__ = "documents"

    # Identity and Tenant Ownership
    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

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
        Index("ix_documents_project_id_created_at", "project_id", "created_at"),
    )

    # Relational Hierarchies
    project: Mapped["ProjectModel"] = relationship(
        "ProjectModel",
        back_populates="documents",
    )
    versions: Mapped[list["DocumentVersionModel"]] = relationship(
        "DocumentVersionModel",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentVersionModel.version_number",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize document model to a dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """Safe representation omitting relational children."""
        return f"DocumentModel(id={self.id!r}, project_id={self.project_id!r}, name={self.name!r})"


class DocumentVersionModel(Base):
    """Authoritative representation of a single physical version of an uploaded source file."""

    __tablename__ = "document_versions"

    # Identity and Foreign References
    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Supabase Object Storage Coordinates
    storage_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # File Attributes
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    etag: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Processing and Indexing State
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=DocumentVersionStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

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
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_document_id_version_number"),
        Index("ix_document_versions_project_id_document_id", "project_id", "document_id"),
    )

    # Relational Hierarchies
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="versions",
    )
    project: Mapped["ProjectModel"] = relationship("ProjectModel")
    chunks: Mapped[list["ChunkModel"]] = relationship(
        "ChunkModel",
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChunkModel.chunk_index",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize document version model to a dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "version_number": self.version_number,
            "storage_bucket": self.storage_bucket,
            "storage_path": self.storage_path,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "etag": self.etag,
            "status": self.status,
            "error_message": self.error_message,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """Safe representation omitting full relational tree."""
        return (
            f"DocumentVersionModel(id={self.id!r}, "
            f"document_id={self.document_id!r}, "
            f"version_number={self.version_number}, "
            f"status={self.status!r})"
        )
