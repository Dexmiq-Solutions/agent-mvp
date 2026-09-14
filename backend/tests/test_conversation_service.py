"""Unit and integration tests for ConversationService."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from exceptions.conversation import (
    ConversationNotFoundError,
    InvalidConversationDataError,
    InvalidMessageDataError,
    InvalidMessageRoleError,
    MessageNotFoundError,
    ProjectConversationMismatchError,
)
from exceptions.project import ProjectNotFoundError
from models.base import Base
from models.conversation import ConversationModel
from models.message import MessageModel
from models.project import ProjectModel
from services.conversation_service import ConversationService
from services.project_service import ProjectService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def create_test_session() -> AsyncIterator[AsyncSession]:
    """Create an in-memory SQLite database and yield an active AsyncSession."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.anyio
async def test_create_and_get_conversation():
    """Verify creating and retrieving a conversation within a project."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Alpha Project")

        # Create conversation
        conv = await conv_service.create_conversation(
            project_id=project.id,
            title="Design Discussion",
        )
        assert conv.id is not None
        assert conv.project_id == project.id
        assert conv.title == "Design Discussion"
        assert conv.created_at is not None
        assert conv.updated_at is not None

        # Retrieve conversation
        fetched = await conv_service.get_conversation(
            project_id=project.id,
            conversation_id=conv.id,
        )
        assert fetched.id == conv.id
        assert fetched.title == "Design Discussion"


@pytest.mark.anyio
async def test_create_conversation_default_title():
    """Verify conversation creation applies default title when none or empty provided."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Beta Project")

        conv_default = await conv_service.create_conversation(project_id=project.id)
        assert conv_default.title == "New Conversation"

        conv_empty = await conv_service.create_conversation(
            project_id=project.id,
            title="   ",
        )
        assert conv_empty.title == "New Conversation"


@pytest.mark.anyio
async def test_create_conversation_nonexistent_project():
    """Verify creating a conversation under an unknown project raises ProjectNotFoundError."""
    async with create_test_session() as session:
        conv_service = ConversationService(session=session)
        with pytest.raises(ProjectNotFoundError):
            await conv_service.create_conversation(
                project_id="nonexistent-project-id",
                title="Ghost Chat",
            )


@pytest.mark.anyio
async def test_list_conversations_project_isolation():
    """Verify list_conversations returns only conversations for the target project."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        proj_a = await proj_service.create_project(name="Project A")
        proj_b = await proj_service.create_project(name="Project B")

        # Create 2 conversations in A, 1 in B
        c_a1 = await conv_service.create_conversation(project_id=proj_a.id, title="A1")
        c_a2 = await conv_service.create_conversation(project_id=proj_a.id, title="A2")
        c_b1 = await conv_service.create_conversation(project_id=proj_b.id, title="B1")

        list_a = await conv_service.list_conversations(project_id=proj_a.id)
        assert len(list_a) == 2
        a_ids = {c.id for c in list_a}
        assert c_a1.id in a_ids
        assert c_a2.id in a_ids
        assert c_b1.id not in a_ids

        list_b = await conv_service.list_conversations(project_id=proj_b.id)
        assert len(list_b) == 1
        assert list_b[0].id == c_b1.id


@pytest.mark.anyio
async def test_get_conversation_project_isolation_mismatch():
    """Verify accessing another project's conversation raises ProjectConversationMismatchError."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        proj_a = await proj_service.create_project(name="Project A")
        proj_b = await proj_service.create_project(name="Project B")

        conv_a = await conv_service.create_conversation(project_id=proj_a.id, title="Secret A")

        # Accessing conv_a with proj_b
        with pytest.raises(ProjectConversationMismatchError) as exc_info:
            await conv_service.get_conversation(
                project_id=proj_b.id,
                conversation_id=conv_a.id,
            )
        assert exc_info.value.conversation_id == conv_a.id
        assert exc_info.value.expected_project_id == proj_b.id
        assert exc_info.value.actual_project_id == proj_a.id


@pytest.mark.anyio
async def test_update_and_delete_conversation():
    """Verify updating conversation title and deleting a conversation."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Gamma Project")
        conv = await conv_service.create_conversation(project_id=project.id, title="Old Title")

        # Update title
        updated = await conv_service.update_conversation(
            project_id=project.id,
            conversation_id=conv.id,
            title="New Title",
        )
        assert updated.title == "New Title"

        # Reject empty title update
        with pytest.raises(InvalidConversationDataError):
            await conv_service.update_conversation(
                project_id=project.id,
                conversation_id=conv.id,
                title="   ",
            )

        # Delete conversation
        await conv_service.delete_conversation(
            project_id=project.id,
            conversation_id=conv.id,
        )

        with pytest.raises(ConversationNotFoundError):
            await conv_service.get_conversation(
                project_id=project.id,
                conversation_id=conv.id,
            )


@pytest.mark.anyio
async def test_message_creation_and_ordering():
    """Verify persisting messages and retrieving them in deterministic chronological order."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Message Project")
        conv = await conv_service.create_conversation(project_id=project.id, title="Chat 1")

        # Initially 0 messages
        messages = await conv_service.list_messages(project_id=project.id, conversation_id=conv.id)
        assert len(messages) == 0
        assert await conv_service.get_messages_count(project.id, conv.id) == 0

        # Create user message
        m1 = await conv_service.create_message(
            project_id=project.id,
            conversation_id=conv.id,
            role="user",
            content="Hello RAG assistant!",
            metadata={"source": "web"},
        )
        assert m1.id is not None
        assert m1.conversation_id == conv.id
        assert m1.role == "user"
        assert m1.content == "Hello RAG assistant!"
        assert m1.meta == {"source": "web"}

        # Create assistant message
        m2 = await conv_service.create_message(
            project_id=project.id,
            conversation_id=conv.id,
            role="assistant",
            content="Hello! How can I help with your project documents?",
            metadata={"confidence": 0.99},
        )
        assert m2.role == "assistant"

        # List messages
        history = await conv_service.list_messages(project_id=project.id, conversation_id=conv.id)
        assert len(history) == 2
        assert history[0].id == m1.id
        assert history[1].id == m2.id
        assert await conv_service.get_messages_count(project.id, conv.id) == 2

        # Get specific message
        fetched_m1 = await conv_service.get_message(
            project_id=project.id,
            conversation_id=conv.id,
            message_id=m1.id,
        )
        assert fetched_m1.id == m1.id
        assert fetched_m1.content == "Hello RAG assistant!"


@pytest.mark.anyio
async def test_message_validation_errors():
    """Verify validation for message roles, empty contents, and non-existent records."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Validation Project")
        conv = await conv_service.create_conversation(project_id=project.id)

        # Invalid role
        with pytest.raises(InvalidMessageRoleError):
            await conv_service.create_message(
                project_id=project.id,
                conversation_id=conv.id,
                role="agent",  # Disallowed
                content="I am an agent",
            )

        # Empty content
        with pytest.raises(InvalidMessageDataError):
            await conv_service.create_message(
                project_id=project.id,
                conversation_id=conv.id,
                role="user",
                content="   ",
            )

        # Message not found
        with pytest.raises(MessageNotFoundError):
            await conv_service.get_message(
                project_id=project.id,
                conversation_id=conv.id,
                message_id="unknown-msg-id",
            )


@pytest.mark.anyio
async def test_conversation_cascade_deletes_messages():
    """Verify deleting a conversation cascades to delete all child messages."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Cascade Project")
        conv = await conv_service.create_conversation(project_id=project.id)

        await conv_service.create_message(
            project_id=project.id,
            conversation_id=conv.id,
            role="user",
            content="Message 1",
        )
        await conv_service.create_message(
            project_id=project.id,
            conversation_id=conv.id,
            role="assistant",
            content="Message 2",
        )

        assert await conv_service.get_messages_count(project.id, conv.id) == 2

        # Delete conversation
        await conv_service.delete_conversation(project_id=project.id, conversation_id=conv.id)

        # Confirm conversation is gone
        with pytest.raises(ConversationNotFoundError):
            await conv_service.get_conversation(project_id=project.id, conversation_id=conv.id)


@pytest.mark.anyio
async def test_get_conversation_context_rag_boundary():
    """Verify get_conversation_context helper returns chronological recent turns with limit."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="RAG Context Project")
        conv = await conv_service.create_conversation(project_id=project.id)

        # Add 5 messages
        for i in range(5):
            await conv_service.create_message(
                project_id=project.id,
                conversation_id=conv.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i}",
            )

        # Fetch last 3 messages
        context = await conv_service.get_conversation_context(
            project_id=project.id,
            conversation_id=conv.id,
            max_messages=3,
        )
        assert len(context) == 3
        # Chronological order
        assert context[0].content == "Turn 2"
        assert context[1].content == "Turn 3"
        assert context[2].content == "Turn 4"


@pytest.mark.anyio
async def test_message_pagination():
    """Verify limit and offset pagination on message listing."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Pagination Project")
        conv = await conv_service.create_conversation(project_id=project.id)

        for i in range(10):
            await conv_service.create_message(
                project_id=project.id,
                conversation_id=conv.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        # Page 1: limit 4, offset 0
        page1 = await conv_service.list_messages(
            project_id=project.id,
            conversation_id=conv.id,
            limit=4,
            offset=0,
        )
        assert len(page1) == 4
        assert page1[0].content == "Message 0"
        assert page1[3].content == "Message 3"

        # Page 2: limit 4, offset 4
        page2 = await conv_service.list_messages(
            project_id=project.id,
            conversation_id=conv.id,
            limit=4,
            offset=4,
        )
        assert len(page2) == 4
        assert page2[0].content == "Message 4"
        assert page2[3].content == "Message 7"

        # Page 3: limit 4, offset 8
        page3 = await conv_service.list_messages(
            project_id=project.id,
            conversation_id=conv.id,
            limit=4,
            offset=8,
        )
        assert len(page3) == 2
        assert page3[0].content == "Message 8"
        assert page3[1].content == "Message 9"


@pytest.mark.anyio
async def test_message_project_isolation():
    """Verify cannot create or list messages across project boundary."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        proj_a = await proj_service.create_project(name="Tenant A")
        proj_b = await proj_service.create_project(name="Tenant B")

        conv_a = await conv_service.create_conversation(project_id=proj_a.id)

        # Tenant B cannot create message in Tenant A's conversation
        with pytest.raises(ProjectConversationMismatchError):
            await conv_service.create_message(
                project_id=proj_b.id,
                conversation_id=conv_a.id,
                role="user",
                content="Unauthorized",
            )

        # Tenant B cannot list messages in Tenant A's conversation
        with pytest.raises(ProjectConversationMismatchError):
            await conv_service.list_messages(
                project_id=proj_b.id,
                conversation_id=conv_a.id,
            )

        # Tenant B cannot get context of Tenant A's conversation
        with pytest.raises(ProjectConversationMismatchError):
            await conv_service.get_conversation_context(
                project_id=proj_b.id,
                conversation_id=conv_a.id,
            )


@pytest.mark.anyio
async def test_disallowed_message_roles():
    """Verify agent, tool, system, and retriever roles are rejected."""
    async with create_test_session() as session:
        proj_service = ProjectService(session=session)
        conv_service = ConversationService(session=session, project_service=proj_service)

        project = await proj_service.create_project(name="Roles Project")
        conv = await conv_service.create_conversation(project_id=project.id)

        for disallowed in ["agent", "tool", "system", "retriever", "bot"]:
            with pytest.raises(InvalidMessageRoleError):
                await conv_service.create_message(
                    project_id=project.id,
                    conversation_id=conv.id,
                    role=disallowed,
                    content="Hello",
                )

