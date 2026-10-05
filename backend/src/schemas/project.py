"""Pydantic request and response schemas for Projects."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectBase(BaseModel):
    """Base schema containing common project attributes."""

    name: str = Field(..., min_length=1, max_length=255, description="Project display name")
    description: Optional[str] = Field(None, description="Optional project description")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure name is not empty or pure whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Project name cannot be empty or whitespace only.")
        return stripped


class ProjectCreate(ProjectBase):
    """Schema for creating a new project."""


class ProjectUpdate(BaseModel):
    """Schema for updating an existing project."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project display name")
    description: Optional[str] = Field(None, description="Optional project description")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Ensure updated name is not pure whitespace if provided."""
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Project name cannot be empty or whitespace only.")
            return stripped
        return v


class ProjectResponse(ProjectBase):
    """Schema for project responses."""

    id: str = Field(..., description="Unique project identifier")
    user_id: Optional[str] = Field(None, description="Owning user identifier")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp with timezone")
    updated_at: Optional[datetime] = Field(None, description="Last modification timestamp with timezone")

    model_config = ConfigDict(from_attributes=True)
