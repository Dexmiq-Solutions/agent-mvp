"""SQLAlchemy entity model for Conversations.

A Conversation belongs to a Project and contains an ordered sequence of Messages.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from models.message import MessageModel
    from models.project import ProjectModel


class ConversationModel(Base):
    """Authoritative conversation entity scoped within a project."""

    __tablename__ = "conversations"

    # Identity and Tenant Scoping
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
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        default="New Conversation",
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
        Index("ix_conversations_project_id_updated_at", "project_id", "updated_at"),
    )

    # Relational Hierarchies
    project: Mapped["ProjectModel"] = relationship(
        "ProjectModel",
        back_populates="conversations",
    )
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageModel.created_at",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation model to a dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """Safe representation omitting messages list."""
        return f"ConversationModel(id={self.id!r}, project_id={self.project_id!r}, title={self.title!r})"
