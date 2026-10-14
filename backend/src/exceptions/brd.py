"""Domain exceptions for BRD persistence and version management."""

from typing import Optional


class BRDError(Exception):
    """Base exception for all BRD storage and versioning domain errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class BRDNotFoundError(BRDError):
    """Raised when a canonical BRD does not exist for a project."""

    def __init__(
        self,
        project_id: str,
        message: Optional[str] = None,
    ) -> None:
        msg = message or f"Canonical BRD not found for project '{project_id}'."
        super().__init__(msg)
        self.project_id = project_id


class BRDVersionNotFoundError(BRDError):
    """Raised when a specific BRD version snapshot or working draft cannot be found."""

    def __init__(
        self,
        version_identifier: str | int,
        project_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message:
            msg = message
        elif project_id:
            msg = f"BRD version '{version_identifier}' not found for project '{project_id}'."
        else:
            msg = f"BRD version '{version_identifier}' not found."
        super().__init__(msg)
        self.version_identifier = version_identifier
        self.project_id = project_id


class BRDVersionImmutableError(BRDError):
    """Raised when attempting to modify a frozen/historical version snapshot."""

    def __init__(
        self,
        version_id: str,
        version_number: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        msg = (
            message
            or f"BRD version '{version_id}' (version #{version_number}) is an immutable snapshot and cannot be modified."
        )
        super().__init__(msg)
        self.version_id = version_id
        self.version_number = version_number


class InvalidBRDDataError(BRDError):
    """Raised when BRD, version, or section input data fails validation."""
    pass
