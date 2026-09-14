"""Pydantic request and response schemas for RAG retrieval operations."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalRequestSchema(BaseModel):
    """Application-facing request schema to execute a project-scoped retrieval query."""

    query: str = Field(..., min_length=1, description="Raw user query to retrieve project knowledge for.")
    top_k: Optional[int] = Field(None, gt=0, description="Override maximum number of final retrieved chunks.")
    dense_top_k: Optional[int] = Field(None, gt=0, description="Override dense vector search limit.")
    sparse_top_k: Optional[int] = Field(None, gt=0, description="Override sparse keyword search limit.")
    enable_transformation: Optional[bool] = Field(None, description="Toggle query transformation stage.")
    enable_sparse: Optional[bool] = Field(None, description="Toggle sparse keyword retrieval stage.")
    enable_reranking: Optional[bool] = Field(None, description="Toggle cross-encoder reranking stage.")
    enable_relevance_check: Optional[bool] = Field(None, description="Toggle post-retrieval relevance check.")
    metadata_filters: Optional[dict[str, Any]] = Field(None, description="Optional metadata filter constraints.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is non-empty after trimming."""
        clean = v.strip()
        if not clean:
            raise ValueError("query cannot be empty or whitespace only.")
        return clean


class RetrievedChunkSchema(BaseModel):
    """Schema representing an individual retrieved and hydrated chunk."""

    chunk_id: str = Field(..., description="Unique persistent chunk identifier")
    document_id: str = Field(..., description="Logical document identifier")
    document_version_id: Optional[str] = Field(None, description="Physical document version identifier")
    content: str = Field(..., description="Authoritative stored chunk text")
    rank: int = Field(..., description="1-based final ranking position")
    score: float = Field(..., description="Primary relevance score from cross-encoder or fusion")
    heading: Optional[str] = Field(None, description="Extracted heading context")
    section_path: list[str] = Field(default_factory=list, description="Section hierarchy path")
    contextual_content: Optional[str] = Field(None, description="Contextually enriched content if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    model_config = ConfigDict(from_attributes=True)


class RetrievalAttemptMetadataSchema(BaseModel):
    """Telemetry schema for an individual retrieval attempt."""

    attempt: int
    query: str
    is_transformed: bool
    transformed_query: Optional[str] = None
    strategy_used: Optional[str] = None
    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    fused_candidates_count: int = 0
    filtered_candidates_count: int = 0
    reranked_candidates_count: int = 0
    hydrated_candidates_count: int = 0
    is_relevant: Optional[bool] = None
    relevance_reason: Optional[str] = None
    latency_ms: float = 0.0


class RetrievalExecutionMetadataSchema(BaseModel):
    """Telemetry schema for the full retrieval operation."""

    total_duration_ms: float
    attempts_count: int
    fallback_triggered: bool
    attempts: list[RetrievalAttemptMetadataSchema] = Field(default_factory=list)
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalResponseSchema(BaseModel):
    """Application-facing response schema for a completed retrieval operation."""

    project_id: str = Field(..., description="Owning project identifier boundary")
    original_query: str = Field(..., description="Original raw user query")
    retrieval_query: str = Field(..., description="Query representation used for final retrieval")
    chunk_count: int = Field(..., description="Number of retrieved chunks")
    chunks: list[RetrievedChunkSchema] = Field(default_factory=list, description="Ordered retrieved chunks")
    formatted_context: str = Field(..., description="Formatted context string ready for prompt injection")
    execution_metadata: RetrievalExecutionMetadataSchema = Field(..., description="Execution telemetry")

    model_config = ConfigDict(from_attributes=True)
