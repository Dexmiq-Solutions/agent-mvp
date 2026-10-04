"""Application service for Conversation and Message lifecycle management."""

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from observability.logging import get_logger
from exceptions.conversation import (
    ConversationNotFoundError,
    InvalidConversationDataError,
    InvalidMessageDataError,
    InvalidMessageRoleError,
    MessageNotFoundError,
    ProjectConversationMismatchError,
)
from models.conversation import ConversationModel
from models.message import MessageModel
from services.project_service import ProjectService

logger = get_logger(__name__)

ALLOWED_MESSAGE_ROLES = ("user", "assistant")


class ConversationService:
    """Service orchestrating project-scoped conversations and message persistence."""

    def __init__(
        self,
        session: AsyncSession,
        project_service: Optional[ProjectService] = None,
    ) -> None:
        """Initialize ConversationService.

        Args:
            session: Active asynchronous SQLAlchemy session.
            project_service: Optional ProjectService for validating project existence.
        """
        self._session = session
        self._project_service = project_service or ProjectService(session=session)

    async def create_conversation(
        self,
        project_id: str,
        title: Optional[str] = None,
    ) -> ConversationModel:
        """Create and persist a new conversation within a project boundary.

        Args:
            project_id: Owning project identifier.
            title: Optional display title for the conversation.

        Returns:
            Newly created and flushed ConversationModel.

        Raises:
            ProjectNotFoundError: If owning project does not exist.
            InvalidConversationDataError: If provided title is invalid.
        """
        # 1. Enforce Project Isolation & Existence
        await self._project_service.get_project(project_id)

        clean_title = "New Conversation"
        if title is not None:
            stripped = title.strip()
            if not stripped:
                clean_title = "New Conversation"
            else:
                clean_title = stripped

        conversation = ConversationModel(
            project_id=project_id,
            title=clean_title,
        )
        self._session.add(conversation)
        await self._session.flush()

        logger.info(
            "Created conversation id: %s for project id: %s (title: '%s')",
            conversation.id,
            project_id,
            conversation.title,
        )
        return conversation

    async def get_project_context(self, project_id: str) -> dict[str, Any]:
        """Retrieve project metadata and available source document inventory for agent context."""
        return await self._project_service.get_project_context(project_id)

    async def get_conversation(
        self,
        project_id: str,
        conversation_id: str,
    ) -> ConversationModel:
        """Retrieve a conversation while strictly enforcing project ownership isolation.

        Args:
            project_id: Expected project tenant identifier.
            conversation_id: Conversation primary key identifier.

        Returns:
            ConversationModel instance.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ProjectConversationMismatchError: If conversation belongs to another project.
        """
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.id == conversation_id)
        )
        result = await self._session.execute(stmt)
        conversation = result.scalar_one_or_none()

        if conversation is None:
            logger.warning(
                "Conversation '%s' not found for project '%s'",
                conversation_id,
                project_id,
            )
            raise ConversationNotFoundError(
                conversation_id=conversation_id,
                project_id=project_id,
            )

        # Enforce strict Project Isolation
        if conversation.project_id != project_id:
            logger.warning(
                "Project boundary violation: conversation '%s' belongs to project '%s', not '%s'",
                conversation_id,
                conversation.project_id,
                project_id,
            )
            raise ProjectConversationMismatchError(
                conversation_id=conversation_id,
                expected_project_id=project_id,
                actual_project_id=conversation.project_id,
            )

        return conversation

    async def list_conversations(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConversationModel]:
        """List all conversations belonging to a project, ordered by last update descending.

        Args:
            project_id: Owning project identifier.
            limit: Maximum conversations to return (1..100).
            offset: Number of records to skip.

        Returns:
            List of ConversationModel instances scoped strictly to the project.

        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        await self._project_service.get_project(project_id)

        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.project_id == project_id)
            .order_by(
                ConversationModel.updated_at.desc().nullslast(),
                ConversationModel.created_at.desc().nullslast(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_conversation(
        self,
        project_id: str,
        conversation_id: str,
        title: Optional[str] = None,
    ) -> ConversationModel:
        """Update mutable fields of a conversation under project boundary validation.

        Args:
            project_id: Owning project identifier.
            conversation_id: Conversation identifier.
            title: Optional updated title string.

        Returns:
            Updated ConversationModel instance.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
            InvalidConversationDataError: If title is empty or invalid.
        """
        conversation = await self.get_conversation(
            project_id=project_id,
            conversation_id=conversation_id,
        )

        if title is not None:
            stripped = title.strip()
            if not stripped:
                raise InvalidConversationDataError("Conversation title cannot be empty or whitespace only.")
            conversation.title = stripped

        conversation.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

        logger.info(
            "Updated conversation id: %s in project id: %s (new title: '%s')",
            conversation_id,
            project_id,
            conversation.title,
        )
        return conversation

    async def delete_conversation(
        self,
        project_id: str,
        conversation_id: str,
    ) -> None:
        """Delete a conversation and cascade deletion to all contained messages.

        Args:
            project_id: Owning project identifier.
            conversation_id: Conversation identifier.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
        """
        conversation = await self.get_conversation(
            project_id=project_id,
            conversation_id=conversation_id,
        )

        await self._session.delete(conversation)
        await self._session.flush()

        logger.info(
            "Deleted conversation id: %s from project id: %s",
            conversation_id,
            project_id,
        )

    async def create_message(
        self,
        project_id: str,
        conversation_id: str,
        content: str,
        role: str = "user",
        metadata: Optional[dict[str, Any]] = None,
    ) -> MessageModel:
        """Persist a new message turn within a conversation.

        Args:
            project_id: Owning project identifier.
            conversation_id: Owning conversation identifier.
            content: Message textual content.
            role: Turn role ('user' or 'assistant').
            metadata: Optional structured metadata dictionary.

        Returns:
            Newly created and persisted MessageModel instance.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
            InvalidMessageRoleError: If role is not 'user' or 'assistant'.
            InvalidMessageDataError: If content is empty or whitespace only.
        """
        # 1. Enforce Conversation and Project Ownership
        conversation = await self.get_conversation(
            project_id=project_id,
            conversation_id=conversation_id,
        )

        # 2. Validate Message Role
        cleaned_role = (role or "").strip().lower()
        if cleaned_role not in ALLOWED_MESSAGE_ROLES:
            logger.warning(
                "Invalid message role '%s' rejected for conversation '%s'",
                role,
                conversation_id,
            )
            raise InvalidMessageRoleError(role=role, allowed_roles=ALLOWED_MESSAGE_ROLES)

        # 3. Validate Message Content
        if not content or not content.strip():
            logger.warning("Empty message content rejected for conversation '%s'", conversation_id)
            raise InvalidMessageDataError("Message content cannot be empty or whitespace only.")

        clean_content = content.strip()

        # 4. Instantiate and Persist Message Turn
        message = MessageModel(
            conversation_id=conversation_id,
            role=cleaned_role,
            content=clean_content,
            message_metadata=dict(metadata or {}),
        )
        self._session.add(message)

        # 5. Update Conversation Touch Timestamp
        conversation.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

        logger.info(
            "Created message id: %s in conversation id: %s (project: %s, role: '%s', content_length: %d)",
            message.id,
            conversation_id,
            project_id,
            cleaned_role,
            len(clean_content),
        )
        return message

    async def list_messages(
        self,
        project_id: str,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MessageModel]:
        """Retrieve ordered message history for a conversation in deterministic chronological order.

        Args:
            project_id: Owning project identifier.
            conversation_id: Owning conversation identifier.
            limit: Maximum messages to return (1..100).
            offset: Number of records to skip.

        Returns:
            List of MessageModel records ordered by created_at ascending, then id ascending.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
        """
        # Verify conversation and project isolation
        await self.get_conversation(project_id=project_id, conversation_id=conversation_id)

        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(
                MessageModel.created_at.asc().nullslast(),
                MessageModel.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_message(
        self,
        project_id: str,
        conversation_id: str,
        message_id: str,
    ) -> MessageModel:
        """Retrieve a specific message within a conversation.

        Args:
            project_id: Owning project identifier.
            conversation_id: Owning conversation identifier.
            message_id: Target message identifier.

        Returns:
            MessageModel instance.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
            MessageNotFoundError: If message does not exist within conversation.
        """
        await self.get_conversation(project_id=project_id, conversation_id=conversation_id)

        stmt = select(MessageModel).where(
            MessageModel.id == message_id,
            MessageModel.conversation_id == conversation_id,
        )
        result = await self._session.execute(stmt)
        message = result.scalar_one_or_none()

        if message is None:
            logger.warning(
                "Message '%s' not found in conversation '%s'",
                message_id,
                conversation_id,
            )
            raise MessageNotFoundError(
                message_id=message_id,
                conversation_id=conversation_id,
            )

        return message

    async def get_messages_count(
        self,
        project_id: str,
        conversation_id: str,
    ) -> int:
        """Count total messages belonging to a conversation.

        Args:
            project_id: Owning project identifier.
            conversation_id: Owning conversation identifier.

        Returns:
            Integer total count of messages.
        """
        await self.get_conversation(project_id=project_id, conversation_id=conversation_id)

        stmt = (
            select(func.count(MessageModel.id))
            .where(MessageModel.conversation_id == conversation_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one() or 0

    async def get_conversation_context(
        self,
        project_id: str,
        conversation_id: str,
        max_messages: int = 10,
    ) -> list[MessageModel]:
        """RAG Integration Boundary: Retrieve recent conversation turns for context assembly.

        Fetches up to `max_messages` recent turns in chronological order to supply conversation
        history to future generation and retrieval stages.

        Args:
            project_id: Owning project identifier.
            conversation_id: Owning conversation identifier.
            max_messages: Maximum recent messages to assemble.

        Returns:
            List of MessageModel items ordered chronologically.
        """
        await self.get_conversation(project_id=project_id, conversation_id=conversation_id)

        # Fetch recent turns ordered descending, then reverse to chronological order
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(
                MessageModel.created_at.desc().nullslast(),
                MessageModel.id.desc(),
            )
            .limit(max_messages)
        )
        result = await self._session.execute(stmt)
        recent = list(result.scalars().all())
        recent.reverse()
        return recent
