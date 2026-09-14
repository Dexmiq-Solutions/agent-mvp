"""Domain exceptions for project management."""

from typing import Optional


class ProjectError(Exception):
    """Base exception for all project-related domain errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class ProjectNotFoundError(ProjectError):
    """Raised when a requested project does not exist."""

    def __init__(self, project_id: str, message: Optional[str] = None) -> None:
        msg = message or f"Project with ID '{project_id}' not found."
        super().__init__(msg)
        self.project_id = project_id


class InvalidProjectDataError(ProjectError):
    """Raised when project data or parameters fail validation."""
