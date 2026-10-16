"""Application service for Conversation and Message lifecycle management."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from observability.logging import get_logger
from exceptions.conversation import (
    ConversationNotFoundError,
    ExecutionLeaseConflictError,
    InvalidConversationDataError,
    InvalidMessageDataError,
    InvalidMessageRoleError,
    MessageNotFoundError,
    ProjectConversationMismatchError,
    StaleExecutionFencedError,
    WorkflowMismatchError,
    WorkflowSwitchBlockedError,
)
from models.conversation import ConversationModel
from models.message import MessageModel
from services.project_service import ProjectService

logger = get_logger(__name__)

ALLOWED_MESSAGE_ROLES = ("user", "assistant")


def _get_db_clock_expr(session: AsyncSession) -> Any:
    """Return dialect-aware database real-time clock expression.

    Uses PostgreSQL `clock_timestamp()` which reflects real-time progression during a
    transaction (unlike `now()` which remains fixed at transaction start time).
    Falls back to `func.now()` for SQLite.
    """
    try:
        bind = getattr(session, "bind", None)
        if bind is None and hasattr(session, "get_bind"):
            bind = session.get_bind()
        if bind is not None and getattr(bind.dialect, "name", "") == "postgresql":
            return func.clock_timestamp()
    except Exception:
        pass
    return func.now()


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
        active_workflow: Optional[str] = "brd",
    ) -> ConversationModel:
        """Create and persist a new conversation within a project boundary.

        Args:
            project_id: Owning project identifier.
            title: Optional display title for the conversation.
            active_workflow: Initial active workflow ('brd' or 'recommendation'). Defaults to 'brd'.

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

        cleaned_workflow = (active_workflow or "brd").strip().lower()
        if cleaned_workflow not in ("brd", "recommendation"):
            cleaned_workflow = "brd"

        conversation = ConversationModel(
            project_id=project_id,
            title=clean_title,
            active_workflow=cleaned_workflow,
        )
        self._session.add(conversation)
        await self._session.flush()

        logger.info(
            "Created conversation id: %s for project id: %s (title: '%s', workflow: '%s')",
            conversation.id,
            project_id,
            conversation.title,
            conversation.active_workflow,
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
    ) -> list[tuple[ConversationModel, int]]:
        """List all conversations belonging to a project, ordered by last update descending.

        Args:
            project_id: Owning project identifier.
            limit: Maximum conversations to return (1..100).
            offset: Number of records to skip.

        Returns:
            List of tuples (ConversationModel, messages_count) scoped strictly to the project.

        Raises:
            ProjectNotFoundError: If project does not exist.
        """
        await self._project_service.get_project(project_id)

        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        stmt = (
            select(
                ConversationModel,
                func.count(MessageModel.id).label("messages_count")
            )
            .outerjoin(MessageModel, MessageModel.conversation_id == ConversationModel.id)
            .where(ConversationModel.project_id == project_id)
            .group_by(ConversationModel.id)
            .order_by(
                ConversationModel.updated_at.desc().nullslast(),
                ConversationModel.created_at.desc().nullslast(),
            )
            .limit(limit)
            .offset(offset)
        )

        result = await self._session.execute(stmt)
        return list(result.all())

    async def update_conversation(
        self,
        project_id: str,
        conversation_id: str,
        title: Optional[str] = None,
        active_workflow: Optional[str] = None,
    ) -> ConversationModel:
        """Update mutable fields of a conversation under project boundary validation.

        Args:
            project_id: Owning project identifier.
            conversation_id: Conversation identifier.
            title: Optional updated title string.
            active_workflow: Optional target workflow ('brd' or 'recommendation').

        Returns:
            Updated ConversationModel instance.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ProjectConversationMismatchError: If conversation belongs to another project.
            InvalidConversationDataError: If title or workflow is empty or invalid.
            WorkflowSwitchBlockedError: If switching workflow while execution is running.
        """
        if active_workflow is not None:
            conversation = await self.switch_workflow(
                project_id=project_id,
                conversation_id=conversation_id,
                target_workflow=active_workflow,
            )
        else:
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
            "Updated conversation id: %s in project id: %s (title: '%s', workflow: '%s')",
            conversation_id,
            project_id,
            conversation.title,
            conversation.active_workflow,
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

    async def switch_workflow(
        self,
        project_id: str,
        conversation_id: str,
        target_workflow: str,
    ) -> ConversationModel:
        """Switch the active workflow for a conversation under pessimistic row lock.

        Args:
            project_id: Owning project tenant identifier.
            conversation_id: Target conversation identifier.
            target_workflow: Desired active workflow ('brd' or 'recommendation').

        Returns:
            Updated ConversationModel instance.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ProjectConversationMismatchError: If conversation belongs to another project.
            InvalidConversationDataError: If target_workflow is invalid.
            WorkflowSwitchBlockedError: If an execution run is currently in progress.
        """
        cleaned_workflow = (target_workflow or "").strip().lower()
        if cleaned_workflow not in ("brd", "recommendation"):
            raise InvalidConversationDataError(
                f"Invalid workflow '{target_workflow}'. Allowed workflows are 'brd' and 'recommendation'."
            )

        stmt = (
            select(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.project_id == project_id,
            )
            .with_for_update()
        )
        res = await self._session.execute(stmt)
        conversation = res.scalar_one_or_none()
        if conversation is None:
            existing = await self._session.get(ConversationModel, conversation_id)
            if existing is not None:
                raise ProjectConversationMismatchError(
                    conversation_id=conversation_id,
                    expected_project_id=project_id,
                    actual_project_id=existing.project_id,
                )
            raise ConversationNotFoundError(
                conversation_id=conversation_id,
                project_id=project_id,
            )

        now_utc = datetime.now(timezone.utc)
        expires_at = conversation.lease_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if (
            conversation.execution_status == "running"
            and expires_at is not None
            and expires_at > now_utc
        ):
            raise WorkflowSwitchBlockedError(
                conversation_id=conversation_id,
                active_run_id=conversation.active_run_id,
            )

        conversation.active_workflow = cleaned_workflow
        conversation.execution_status = "idle"
        conversation.active_run_id = None
        conversation.lease_expires_at = None
        conversation.updated_at = now_utc
        await self._session.flush()

        logger.info(
            "Switched active workflow for conversation '%s' (project '%s') to '%s'",
            conversation_id,
            project_id,
            cleaned_workflow,
        )
        return conversation

    async def acquire_execution_lease(
        self,
        project_id: str,
        conversation_id: str,
        run_id: str,
        expected_workflow: str,
        ttl_seconds: int = 45,
    ) -> ConversationModel:
        """Atomically validate active workflow and acquire run-ownership execution lease under row lock.

        Args:
            project_id: Owning project identifier.
            conversation_id: Target conversation identifier.
            run_id: Unique execution run identifier.
            expected_workflow: Expected active workflow ('brd' or 'recommendation').
            ttl_seconds: Lease duration in seconds. Defaults to 45s.

        Returns:
            Locked and updated ConversationModel instance with held lease.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ProjectConversationMismatchError: If conversation belongs to another project.
            WorkflowMismatchError: If conversation active_workflow does not match expected_workflow.
            ExecutionLeaseConflictError: If another run currently holds a valid execution lease.
        """
        stmt = (
            select(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.project_id == project_id,
            )
            .with_for_update()
        )
        res = await self._session.execute(stmt)
        conversation = res.scalar_one_or_none()
        if conversation is None:
            existing = await self._session.get(ConversationModel, conversation_id)
            if existing is not None:
                raise ProjectConversationMismatchError(
                    conversation_id=conversation_id,
                    expected_project_id=project_id,
                    actual_project_id=existing.project_id,
                )
            raise ConversationNotFoundError(
                conversation_id=conversation_id,
                project_id=project_id,
            )

        # 1. Enforce workflow alignment
        if conversation.active_workflow != expected_workflow:
            raise WorkflowMismatchError(
                conversation_id=conversation_id,
                expected_workflow=expected_workflow,
                actual_workflow=conversation.active_workflow,
            )

        # 2. Check lease availability (active non-expired lease by another run)
        now_utc = datetime.now(timezone.utc)
        expires_at = conversation.lease_expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if (
            conversation.execution_status == "running"
            and expires_at is not None
            and expires_at > now_utc
            and conversation.active_run_id != run_id
        ):
            raise ExecutionLeaseConflictError(
                conversation_id=conversation_id,
                active_run_id=conversation.active_run_id,
            )

        # 3. Grant/renew lease to this run
        conversation.execution_status = "running"
        conversation.active_run_id = run_id
        conversation.lease_expires_at = now_utc + timedelta(seconds=ttl_seconds)
        conversation.updated_at = now_utc
        await self._session.flush()

        logger.info(
            "Acquired execution lease for run '%s' on conversation '%s' (workflow: '%s', expires in %ds)",
            run_id,
            conversation_id,
            expected_workflow,
            ttl_seconds,
        )
        return conversation

    async def heartbeat_execution_lease(
        self,
        conversation_id: str,
        run_id: str,
        ttl_seconds: int = 45,
    ) -> bool:
        """Extend lease expiry for an active run. Fails closed and cannot revive an expired lease.

        Uses real-time DB clock check to ensure heartbeats cannot revive a lease that already
        expired or was overtaken by another run.

        Args:
            conversation_id: Target conversation identifier.
            run_id: Run identifier expected to own the lease.
            ttl_seconds: Extension duration from current time.

        Returns:
            True if lease was renewed, False if expired or ownership lost.
        """
        clock_expr = _get_db_clock_expr(self._session)
        now_utc = datetime.now(timezone.utc)
        new_expiry = now_utc + timedelta(seconds=ttl_seconds)

        stmt = (
            update(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.active_run_id == run_id,
                ConversationModel.execution_status == "running",
                ConversationModel.lease_expires_at >= clock_expr,
            )
            .values(
                lease_expires_at=new_expiry,
                updated_at=now_utc,
            )
        )
        res = await self._session.execute(stmt)
        await self._session.commit()
        renewed = bool(res.rowcount and res.rowcount > 0)
        if renewed:
            logger.debug(
                "Heartbeat renewed execution lease for run '%s' on conversation '%s' (+%ds)",
                run_id,
                conversation_id,
                ttl_seconds,
            )
        else:
            logger.warning(
                "Heartbeat failed for run '%s' on conversation '%s': lease expired or superseded",
                run_id,
                conversation_id,
            )
        return renewed

    async def verify_and_renew_lease(
        self,
        conversation_id: str,
        expected_run_id: str,
        ttl_seconds: int = 45,
    ) -> ConversationModel:
        """Verify run ownership and renew lease under row lock before executing consequential writes.

        Args:
            conversation_id: Target conversation identifier.
            expected_run_id: Run identifier expected to hold the lease.
            ttl_seconds: Lease duration to renew.

        Returns:
            Locked ConversationModel instance with renewed lease.

        Raises:
            StaleExecutionFencedError: If lease expired or owned by another run.
        """
        clock_expr = _get_db_clock_expr(self._session)
        stmt = (
            select(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.active_run_id == expected_run_id,
                ConversationModel.execution_status == "running",
                ConversationModel.lease_expires_at >= clock_expr,
            )
            .with_for_update()
        )
        res = await self._session.execute(stmt)
        conversation = res.scalar_one_or_none()
        if conversation is None:
            raise StaleExecutionFencedError(
                run_id=expected_run_id,
                conversation_id=conversation_id,
                message=(
                    f"Lease verification failed: Run '{expected_run_id}' has expired or lost ownership "
                    f"of conversation '{conversation_id}'."
                ),
            )

        now_utc = datetime.now(timezone.utc)
        conversation.lease_expires_at = now_utc + timedelta(seconds=ttl_seconds)
        conversation.updated_at = now_utc
        await self._session.flush()
        return conversation

    async def pre_commit_fencing_assertion(
        self,
        conversation_id: str,
        expected_run_id: str,
    ) -> None:
        """Final pre-commit boundary assertion ensuring lease has not expired during persistence.

        Mandatory Technical Correction:
        Uses actual database time (e.g. PostgreSQL `clock_timestamp()` rather than `now()`),
        because `now()` reflects the transaction start time and cannot detect expiry during
        long-running persistence transactions.

        Args:
            conversation_id: Target conversation identifier.
            expected_run_id: Run identifier expected to hold valid lease.

        Raises:
            StaleExecutionFencedError: If lease expired or active_run_id changed.
        """
        clock_expr = _get_db_clock_expr(self._session)
        stmt = (
            select(ConversationModel.id)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.active_run_id == expected_run_id,
                ConversationModel.execution_status == "running",
                ConversationModel.lease_expires_at >= clock_expr,
            )
        )
        res = await self._session.execute(stmt)
        if res.scalar_one_or_none() is None:
            raise StaleExecutionFencedError(
                run_id=expected_run_id,
                conversation_id=conversation_id,
                message=(
                    f"Pre-commit fencing assertion failed: Run '{expected_run_id}' expired or lost ownership "
                    f"before commit on conversation '{conversation_id}'."
                ),
            )

    async def release_execution_lease(
        self,
        conversation_id: str,
        run_id: str,
    ) -> bool:
        """Safely release execution lease via Compare-And-Swap (CAS) only if run_id matches.

        Args:
            conversation_id: Target conversation identifier.
            run_id: Run identifier releasing the lease.

        Returns:
            True if released by this run, False if already released or superseded.
        """
        stmt = (
            update(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.active_run_id == run_id,
            )
            .values(
                execution_status="idle",
                active_run_id=None,
                lease_expires_at=None,
                updated_at=datetime.now(timezone.utc),
            )
        )
        res = await self._session.execute(stmt)
        await self._session.commit()
        released = bool(res.rowcount and res.rowcount > 0)
        if released:
            logger.info(
                "Released execution lease for run '%s' on conversation '%s'",
                run_id,
                conversation_id,
            )
        else:
            logger.debug(
                "Lease for run '%s' on conversation '%s' already released or superseded",
                run_id,
                conversation_id,
            )
        return released
