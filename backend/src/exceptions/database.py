"""Database-specific domain exceptions."""


class DatabaseError(Exception):
    """Base exception for all database-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is missing or invalid."""


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to establish a connection to the database."""


class DatabaseSessionError(DatabaseError):
    """Raised when an error occurs during a database session operation or transaction."""
