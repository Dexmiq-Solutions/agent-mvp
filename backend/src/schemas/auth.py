"""Pydantic request and response schemas for Authentication."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserSignupRequest(BaseModel):
    """Schema for user account registration."""

    email: EmailStr = Field(..., description="User's valid email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (minimum 8 characters)",
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email address to lowercase stripped format."""
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password is not pure whitespace."""
        if not v.strip():
            raise ValueError("Password cannot be empty or whitespace only.")
        return v


class UserLoginRequest(BaseModel):
    """Schema for user authentication credentials."""

    email: EmailStr = Field(..., description="User's registered email address")
    password: str = Field(..., min_length=1, description="Account password")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email address to lowercase stripped format."""
        return v.strip().lower()


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token submission."""

    refresh_token: str = Field(..., min_length=1, description="Opaque refresh token string")


class TokenResponse(BaseModel):
    """Schema for authentication token response."""

    access_token: str = Field(..., description="Short-lived JWT access token")
    refresh_token: str = Field(..., description="Long-lived opaque refresh token")
    token_type: str = Field(default="bearer", description="Token scheme type")
    expires_in: int = Field(..., description="Access token lifespan in seconds")


class UserResponse(BaseModel):
    """Schema for user profile responses."""

    id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User's registered email address")
    is_active: bool = Field(..., description="Whether user account is active")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Account last update timestamp")

    model_config = ConfigDict(from_attributes=True)
