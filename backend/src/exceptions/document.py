"""Domain exceptions for document (source) and document version management."""

from typing import Optional


class DocumentError(Exception):
    """Base exception for all document-related domain errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document does not exist within the project context."""

    def __init__(self, document_id: str, project_id: Optional[str] = None, message: Optional[str] = None) -> None:
        if message:
            msg = message
        elif project_id:
            msg = f"Document with ID '{document_id}' not found in project '{project_id}'."
        else:
            msg = f"Document with ID '{document_id}' not found."
        super().__init__(msg)
        self.document_id = document_id
        self.project_id = project_id


class DocumentVersionNotFoundError(DocumentError):
    """Raised when a requested document version does not exist."""

    def __init__(
        self,
        document_id: str,
        version_number: int,
        project_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message:
            msg = message
        elif project_id:
            msg = f"Version {version_number} of document '{document_id}' not found in project '{project_id}'."
        else:
            msg = f"Version {version_number} of document '{document_id}' not found."
        super().__init__(msg)
        self.document_id = document_id
        self.version_number = version_number
        self.project_id = project_id


class ProjectDocumentMismatchError(DocumentError):
    """Raised when an operation attempts to access a document outside its owning project."""

    def __init__(self, document_id: str, expected_project_id: str, actual_project_id: str) -> None:
        super().__init__(
            f"Document '{document_id}' belongs to project '{actual_project_id}', not '{expected_project_id}'."
        )
        self.document_id = document_id
        self.expected_project_id = expected_project_id
        self.actual_project_id = actual_project_id


class InvalidDocumentDataError(DocumentError):
    """Raised when document, version, or file payload fails validation."""
