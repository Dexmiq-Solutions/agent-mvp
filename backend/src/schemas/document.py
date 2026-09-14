"""Pydantic request and response schemas for Documents (Sources) and Document Versions."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentVersionResponse(BaseModel):
    """Schema representing a concrete physical version of an uploaded document."""

    id: str = Field(..., description="Unique document version identifier")
    document_id: str = Field(..., description="Owning logical document identifier")
    project_id: str = Field(..., description="Owning project isolation boundary")
    version_number: int = Field(..., description="Sequential version number")
    storage_bucket: str = Field(..., description="Object storage bucket name")
    storage_path: str = Field(..., description="Object storage path within bucket")
    original_filename: str = Field(..., description="Original uploaded file name")
    content_type: Optional[str] = Field(None, description="MIME content type")
    size_bytes: Optional[int] = Field(None, description="Physical file size in bytes")
    etag: Optional[str] = Field(None, description="Object storage checksum / ETag")
    status: str = Field(..., description="Indexing lifecycle state: pending, indexing, ready, failed")
    error_message: Optional[str] = Field(None, description="Error details if status is failed")
    indexed_at: Optional[datetime] = Field(None, description="Timestamp when indexing completed")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp with timezone")
    updated_at: Optional[datetime] = Field(None, description="Last modification timestamp with timezone")

    model_config = ConfigDict(from_attributes=True)


class DocumentUpdate(BaseModel):
    """Schema for updating mutable fields of a document/source."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated display name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Ensure name is not empty or whitespace only if provided."""
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Document name cannot be empty or whitespace only.")
            return stripped
        return v


class DocumentResponse(BaseModel):
    """Schema representing a logical document / source in list views."""

    id: str = Field(..., description="Unique logical document identifier")
    project_id: str = Field(..., description="Owning project isolation boundary")
    name: str = Field(..., description="User-facing document display name")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp with timezone")
    updated_at: Optional[datetime] = Field(None, description="Last modification timestamp with timezone")
    latest_version: Optional[DocumentVersionResponse] = Field(
        None, description="Most recent document version details"
    )
    versions_count: int = Field(0, description="Total number of physical versions")

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    """Detailed schema representing a logical document including all its versions."""

    versions: list[DocumentVersionResponse] = Field(
        default_factory=list, description="All physical versions in sequential order"
    )

    model_config = ConfigDict(from_attributes=True)
