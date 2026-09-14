"""SQLAlchemy entity model for Messages.

A Message represents a single conversational turn (user, assistant, or system) belonging
to a Conversation.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from db.base import Base

if TYPE_CHECKING:
    from models.conversation import ConversationModel


class MessageModel(Base):
    """Authoritative message record within a conversation."""

    __tablename__ = "messages"

    # Identity and Conversation Ownership
    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Structured metadata (mapped to 'metadata' column in SQL)
    message_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        default=dict,
    )

    # Timestamps (Messages are append-only chronological turns)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    # Relational Hierarchies
    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel",
        back_populates="messages",
    )

    @property
    def meta(self) -> dict[str, Any]:
        """Convenience property returning metadata dictionary."""
        return self.message_metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize message model to a dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "metadata": dict(self.message_metadata or {}),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        """Safe representation omitting full message content."""
        return (
            f"MessageModel(id={self.id!r}, "
            f"conversation_id={self.conversation_id!r}, "
            f"role={self.role!r}, "
            f"content_length={len(self.content) if self.content else 0})"
        )
