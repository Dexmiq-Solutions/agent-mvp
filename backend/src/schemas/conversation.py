"""Pydantic request and response schemas for Conversations and Messages."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageRole(str, Enum):
    """Authoritative message roles for conversational turns."""

    USER = "user"
    ASSISTANT = "assistant"


class MessageCreate(BaseModel):
    """Schema for sending and persisting a new message in a conversation."""

    content: str = Field(..., min_length=1, description="Message text content")
    role: str = Field(default="user", description="Message sender role ('user' or 'assistant')")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional structured metadata")
    stream: Optional[bool] = Field(default=None, description="Whether to stream the agent response for user messages")

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure message content is not empty or pure whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message content cannot be empty or whitespace only.")
        return stripped

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is either 'user' or 'assistant'."""
        cleaned = v.strip().lower()
        allowed = {MessageRole.USER.value, MessageRole.ASSISTANT.value}
        if cleaned not in allowed:
            raise ValueError(f"Invalid message role '{v}'. Allowed roles are: {sorted(allowed)}")
        return cleaned


class MessageResponse(BaseModel):
    """Schema for message response."""

    id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Owning conversation identifier")
    role: str = Field(..., description="Message sender role")
    content: str = Field(..., description="Message text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured message metadata")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp with timezone")

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, model: Any) -> "MessageResponse":
        """Build MessageResponse from a MessageModel instance."""
        return cls(
            id=model.id,
            conversation_id=model.conversation_id,
            role=model.role,
            content=model.content,
            metadata=dict(getattr(model, "message_metadata", None) or {}),
            created_at=model.created_at,
        )


class ConversationBase(BaseModel):
    """Base schema containing common conversation attributes."""

    title: Optional[str] = Field(
        default="New Conversation",
        max_length=255,
        description="Conversation display title",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace and enforce sensible title defaults."""
        if v is not None:
            stripped = v.strip()
            if not stripped:
                return "New Conversation"
            return stripped
        return "New Conversation"


class ConversationCreate(ConversationBase):
    """Schema for creating a new conversation."""


class ConversationUpdate(BaseModel):
    """Schema for updating an existing conversation's title."""

    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Updated conversation display title",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        """Ensure updated title is not empty if provided."""
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Conversation title cannot be empty or whitespace only.")
            return stripped
        return v


class ConversationResponse(ConversationBase):
    """Schema for conversation response."""

    id: str = Field(..., description="Unique conversation identifier")
    project_id: str = Field(..., description="Owning project identifier")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp with timezone")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp with timezone")
    messages_count: Optional[int] = Field(default=None, description="Total count of messages in conversation")

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(ConversationResponse):
    """Detailed conversation response including ordered message history."""

    messages: list[MessageResponse] = Field(
        default_factory=list,
        description="Chronological message sequence",
    )
