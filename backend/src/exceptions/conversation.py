"""Domain exceptions for conversation and message flow management."""

from typing import Optional, Sequence


class ConversationError(Exception):
    """Base exception for all conversation-related domain errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class ConversationNotFoundError(ConversationError):
    """Raised when a requested conversation does not exist."""

    def __init__(
        self,
        conversation_id: str,
        project_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message:
            msg = message
        elif project_id:
            msg = f"Conversation with ID '{conversation_id}' not found in project '{project_id}'."
        else:
            msg = f"Conversation with ID '{conversation_id}' not found."
        super().__init__(msg)
        self.conversation_id = conversation_id
        self.project_id = project_id


class ProjectConversationMismatchError(ConversationError):
    """Raised when an operation attempts to access a conversation outside its owning project."""

    def __init__(
        self,
        conversation_id: str,
        expected_project_id: str,
        actual_project_id: str,
    ) -> None:
        super().__init__(
            f"Conversation '{conversation_id}' belongs to project '{actual_project_id}', not '{expected_project_id}'."
        )
        self.conversation_id = conversation_id
        self.expected_project_id = expected_project_id
        self.actual_project_id = actual_project_id


class InvalidConversationDataError(ConversationError):
    """Raised when conversation input data or parameters fail validation."""


class MessageNotFoundError(ConversationError):
    """Raised when a requested message does not exist within the conversation context."""

    def __init__(
        self,
        message_id: str,
        conversation_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message:
            msg = message
        elif conversation_id:
            msg = f"Message with ID '{message_id}' not found in conversation '{conversation_id}'."
        else:
            msg = f"Message with ID '{message_id}' not found."
        super().__init__(msg)
        self.message_id = message_id
        self.conversation_id = conversation_id


class ConversationMessageMismatchError(ConversationError):
    """Raised when an operation attempts to access a message outside its owning conversation."""

    def __init__(
        self,
        message_id: str,
        expected_conversation_id: str,
        actual_conversation_id: str,
    ) -> None:
        super().__init__(
            f"Message '{message_id}' belongs to conversation '{actual_conversation_id}', not '{expected_conversation_id}'."
        )
        self.message_id = message_id
        self.expected_conversation_id = expected_conversation_id
        self.actual_conversation_id = actual_conversation_id


class InvalidMessageDataError(ConversationError):
    """Raised when message payload or content fails validation."""


class InvalidMessageRoleError(ConversationError):
    """Raised when an unsupported or invalid message role is provided."""

    def __init__(
        self,
        role: str,
        allowed_roles: Sequence[str] = ("user", "assistant"),
    ) -> None:
        super().__init__(
            f"Invalid message role '{role}'. Allowed roles: {list(allowed_roles)}"
        )
        self.role = role
        self.allowed_roles = tuple(allowed_roles)
