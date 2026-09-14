"""Pydantic schemas and domain models for LLM Generation Integration."""

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.message import MessageModel
from llm.models import LLMUsage
from schemas.conversation import MessageResponse


@dataclass(frozen=True)
class GenerationResult:
    """Application-level domain result of an end-to-end RAG + LLM generation execution.

    Faithfully encapsulates the accepted assistant response, provenance,
    token usage metrics, retrieval execution telemetry, and evaluation results
    without exposing raw provider SDK objects.
    """

    content: str
    finish_reason: Optional[str]
    usage: Optional[LLMUsage]
    project_id: str
    conversation_id: str
    user_message_id: Optional[str]
    assistant_message_id: Optional[str]
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_metadata: dict[str, Any] = field(default_factory=dict)
    user_message: Optional[MessageModel] = None
    assistant_message: Optional[MessageModel] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize generation result to standard dictionary representation."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "project_id": self.project_id,
            "conversation_id": self.conversation_id,
            "user_message_id": self.user_message_id,
            "assistant_message_id": self.assistant_message_id,
            "retrieval_metadata": dict(self.retrieval_metadata),
            "execution_metadata": dict(self.execution_metadata),
            "evaluation_metadata": dict(self.evaluation_metadata),
        }


class GenerationRequestSchema(BaseModel):
    """Application-facing request schema for triggering RAG-backed LLM generation."""

    user_message: str = Field(..., min_length=1, description="Raw user message text to process and answer.")
    system_instruction: Optional[str] = Field(None, description="Optional system instruction override.")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional user message metadata.")

    @field_validator("user_message")
    @classmethod
    def validate_user_message_not_empty(cls, v: str) -> str:
        """Ensure user message is not empty or whitespace only."""
        clean = v.strip()
        if not clean:
            raise ValueError("user_message cannot be empty or whitespace only.")
        return clean


class GenerationResponseSchema(BaseModel):
    """Application-facing response schema for a completed RAG generation turn."""

    content: str = Field(..., description="Accepted and normalized generated assistant response text.")
    finish_reason: Optional[str] = Field(None, description="Normalized model completion finish reason.")
    usage: Optional[dict[str, Optional[int]]] = Field(None, description="Token usage metrics.")
    project_id: str = Field(..., description="Owning project identifier.")
    conversation_id: str = Field(..., description="Owning conversation identifier.")
    user_message: Optional[MessageResponse] = Field(None, description="Persisted user message turn.")
    assistant_message: Optional[MessageResponse] = Field(None, description="Persisted assistant message turn.")
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict, description="Retrieval execution telemetry.")
    execution_metadata: dict[str, Any] = Field(default_factory=dict, description="End-to-end generation execution telemetry.")
    evaluation_metadata: dict[str, Any] = Field(default_factory=dict, description="Groundedness and safety evaluation details.")

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_result(cls, result: GenerationResult) -> "GenerationResponseSchema":
        """Construct response schema from GenerationResult domain model."""
        return cls(
            content=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage.to_dict() if result.usage else None,
            project_id=result.project_id,
            conversation_id=result.conversation_id,
            user_message=MessageResponse.from_model(result.user_message) if result.user_message else None,
            assistant_message=MessageResponse.from_model(result.assistant_message) if result.assistant_message else None,
            retrieval_metadata=dict(result.retrieval_metadata),
            execution_metadata=dict(result.execution_metadata),
            evaluation_metadata=dict(result.evaluation_metadata),
        )
