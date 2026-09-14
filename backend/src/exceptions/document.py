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


class DocumentProcessingError(DocumentError):
    """Base exception for all document processing lifecycle errors."""


class InvalidDocumentStateTransitionError(DocumentProcessingError):
    """Raised when an invalid lifecycle state transition is attempted on a document version."""

    def __init__(
        self,
        version_id: str,
        current_status: str,
        target_status: str,
        message: Optional[str] = None,
    ) -> None:
        msg = (
            message
            or f"Invalid state transition for document version '{version_id}': cannot transition from '{current_status}' to '{target_status}'."
        )
        super().__init__(msg)
        self.version_id = version_id
        self.current_status = current_status
        self.target_status = target_status


class DocumentAlreadyProcessingError(InvalidDocumentStateTransitionError):
    """Raised when processing is initiated for a document version already actively in indexing state."""

    def __init__(self, version_id: str, message: Optional[str] = None) -> None:
        msg = message or f"Document version '{version_id}' is already actively processing (indexing)."
        super().__init__(
            version_id=version_id,
            current_status="indexing",
            target_status="indexing",
            message=msg,
        )


class DocumentStorageSourceError(DocumentProcessingError):
    """Raised when required object storage source metadata or coordinates are missing or corrupt."""

    def __init__(self, version_id: str, message: Optional[str] = None) -> None:
        msg = message or f"Missing or invalid object storage coordinates for document version '{version_id}'."
        super().__init__(msg)
        self.version_id = version_id


class DocumentVersionMismatchError(DocumentProcessingError):
    """Raised when a document version does not belong to the claimed project or document hierarchy."""

    def __init__(
        self,
        version_id: str,
        expected_parent: str,
        actual_parent: str,
        entity_type: str = "document",
    ) -> None:
        super().__init__(
            f"Document version '{version_id}' belongs to {entity_type} '{actual_parent}', not '{expected_parent}'."
        )
        self.version_id = version_id
        self.expected_parent = expected_parent
        self.actual_parent = actual_parent

