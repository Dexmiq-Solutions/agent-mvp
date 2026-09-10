"""Indexing and storage domain exceptions."""

from typing import Any


class IndexingError(Exception):
    """Base exception for all document indexing and vector storage errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class InvalidIndexingInputError(IndexingError):
    """Raised when indexing input, vectors, chunk IDs, or project associations are invalid or missing."""


class IndexingConfigurationError(IndexingError):
    """Raised when an invalid indexing configuration or collection parameter is supplied."""


class IndexingConnectionError(IndexingError):
    """Raised when unable to establish a connection to the vector store during indexing."""


class IndexingOperationError(IndexingError):
    """Raised when an operation to index or upsert vectors into the vector store fails."""


class IndexingPartialFailureError(IndexingError):
    """Raised when a multi-batch indexing operation partially fails."""

    def __init__(
        self,
        message: str,
        report: Any | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, original_error=original_error)
        self.report = report
