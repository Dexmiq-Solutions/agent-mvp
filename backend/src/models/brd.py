"""SQLAlchemy entity models for canonical BRD, version snapshots, and structured sections.

Enforces:
- Exactly one canonical BRD per project (UNIQUE project_id).
- Strictly acyclic tenant ownership: projects -> brds -> brd_versions -> brd_sections (no redundant project_id).
- Singularity of current working draft via PostgreSQL partial unique index (is_current=True).
- Structured sections with stable identity, ordering, and heading hierarchy.
- Immutable historical snapshots.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from db.base import Base

if TYPE_CHECKING:
    from models.project import ProjectModel


class BRDVersionStatus(str, Enum):
    """Lifecycle status for a BRD version."""

    DRAFT = "draft"
    SNAPSHOT = "snapshot"


class BRDModel(Base):
    """Authoritative canonical BRD entity scoped strictly 1:1 to a Project."""

    __tablename__ = "brds"

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
        unique=True,
        index=True,
    )

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

    # Relational Associations
    project: Mapped["ProjectModel"] = relationship(
        "ProjectModel",
        back_populates="brd",
    )
    versions: Mapped[list["BRDVersionModel"]] = relationship(
        "BRDVersionModel",
        back_populates="brd",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="BRDVersionModel.version_number",
        lazy="selectin",
    )
    current_version: Mapped[Optional["BRDVersionModel"]] = relationship(
        "BRDVersionModel",
        primaryjoin="and_(BRDModel.id == BRDVersionModel.brd_id, BRDVersionModel.is_current.is_(True))",
        uselist=False,
        viewonly=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize canonical BRD entity to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"BRDModel(id={self.id!r}, project_id={self.project_id!r})"


class BRDVersionModel(Base):
    """Authoritative representation of a BRD version (working draft or frozen snapshot)."""

    __tablename__ = "brd_versions"

    # Identity and BRD Association (Acyclic ownership: projects -> brds -> brd_versions)
    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    brd_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("brds.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Version Sequence and Labels
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    version_label: Mapped[str] = mapped_column(String(50), nullable=False, default="0.1")

    # Working Draft vs Frozen Snapshot Flags
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=BRDVersionStatus.DRAFT.value,
        index=True,
    )

    # Derived Document Markdown and Metadata
    assembled_markdown: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=dict,
    )

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
        # Enforce at most one current working draft per BRD at the database level
        Index(
            "uq_brd_versions_one_current",
            "brd_id",
            unique=True,
            postgresql_where=(is_current.is_(True)),
            sqlite_where=(is_current.is_(True)),
        ),
        # Enforce monotonic sequence uniqueness per BRD
        UniqueConstraint("brd_id", "version_number", name="uq_brd_versions_brd_id_version_number"),
        Index("ix_brd_versions_brd_id_status", "brd_id", "status"),
    )

    # Relational Associations
    brd: Mapped["BRDModel"] = relationship(
        "BRDModel",
        back_populates="versions",
    )
    sections: Mapped[list["BRDSectionModel"]] = relationship(
        "BRDSectionModel",
        back_populates="version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="BRDSectionModel.order_index",
        lazy="selectin",
    )

    @property
    def is_snapshot(self) -> bool:
        """True if this version is an immutable historical snapshot."""
        return self.status == BRDVersionStatus.SNAPSHOT.value or not self.is_current

    def to_dict(self) -> dict[str, Any]:
        """Serialize BRD version to dictionary."""
        return {
            "id": self.id,
            "brd_id": self.brd_id,
            "version_number": self.version_number,
            "version_label": self.version_label,
            "is_current": self.is_current,
            "status": self.status,
            "assembled_markdown": self.assembled_markdown,
            "metadata": dict(self.version_metadata or {}),
            "sections_count": len(self.sections) if self.sections else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"BRDVersionModel(id={self.id!r}, brd_id={self.brd_id!r}, "
            f"v={self.version_number}, label={self.version_label!r}, "
            f"current={self.is_current}, status={self.status!r})"
        )


class BRDSectionModel(Base):
    """Authoritative representation of a structured BRD section within a version."""

    __tablename__ = "brd_sections"

    # Identity and Version Association
    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    brd_version_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("brd_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Structure, Identity, and Hierarchy
    section_key: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    heading_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_section_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        ForeignKey("brd_sections.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Content and Status
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="Not Started")
    rework_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Extensible metadata
    section_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=dict,
    )

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
        UniqueConstraint("brd_version_id", "section_key", name="uq_brd_sections_version_id_section_key"),
        Index("ix_brd_sections_version_order", "brd_version_id", "order_index"),
    )

    # Relational Associations
    version: Mapped["BRDVersionModel"] = relationship(
        "BRDVersionModel",
        back_populates="sections",
    )
    parent: Mapped[Optional["BRDSectionModel"]] = relationship(
        "BRDSectionModel",
        remote_side=[id],
        backref="children",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize section model to dictionary."""
        return {
            "id": self.id,
            "brd_version_id": self.brd_version_id,
            "section_key": self.section_key,
            "title": self.title,
            "order_index": self.order_index,
            "heading_level": self.heading_level,
            "parent_section_id": self.parent_section_id,
            "content": self.content,
            "status": self.status,
            "rework_feedback": self.rework_feedback,
            "metadata": dict(self.section_metadata or {}),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"BRDSectionModel(id={self.id!r}, key={self.section_key!r}, "
            f"order={self.order_index}, status={self.status!r}, "
            f"content_length={len(self.content)})"
        )
