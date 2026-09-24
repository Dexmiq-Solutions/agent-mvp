"""Agent domain and runtime infrastructure exceptions."""

from typing import Optional


class AgentError(Exception):
    """Base exception for all Agent execution and runtime errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class AgentConfigurationError(AgentError):
    """Raised when Agent configuration is invalid or missing required credentials/settings."""


class AgentInitializationError(AgentError):
    """Raised when the Agent runtime harness or graph compilation fails to initialize."""


class AgentModelError(AgentError):
    """Raised when the underlying model invocation fails within the Agent runtime."""


class AgentToolExecutionError(AgentError):
    """Raised when an Agent tool fails during execution."""


class AgentExecutionError(AgentError):
    """Raised when the Agent execution cycle fails or encounters an unhandled runtime error."""
