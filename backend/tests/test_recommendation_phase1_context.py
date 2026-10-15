"""Unit tests for deterministic context loading (Step 1A).

Covers:
- Successful context assembly from project, BRD, and conversation services.
- Tenant isolation and failing closed when project ownership check fails.
- Preserving pending clarification, focus entity, and decision history from message metadata.
- Merging user clarification answers into user_provided_facts.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.recommendation.context_loader import load_context
from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    UserContext,
)
from exceptions.project import ProjectNotFoundError
from models.brd import BRDModel, BRDSectionModel, BRDVersionModel
from models.message import MessageModel
from models.project import ProjectModel


@pytest.mark.asyncio
async def test_load_context_success():
    session = AsyncMock()

    # Mock ProjectService
    project_service = AsyncMock()
    project_service.get_project.return_value = ProjectModel(
        id="proj-101",
        user_id="user-101",
        name="Test Project",
    )

    # Mock BRDService
    brd_service = AsyncMock()
    section = BRDSectionModel(
        id="sec-1",
        brd_version_id="ver-1",
        section_key="6.0",
        title="6. Security",
        order_index=6,
        content="Security details",
    )
    draft_version = BRDVersionModel(
        id="ver-1",
        brd_id="brd-1",
        version_number=1,
        version_label="0.1",
        sections=[section],
    )
    brd_model = BRDModel(id="brd-1", project_id="proj-101", versions=[draft_version])
    brd_service.get_or_create_canonical_brd.return_value = (brd_model, draft_version)

    # Mock ConversationService
    conversation_service = AsyncMock()
    prior_msg1 = MessageModel(
        id="m-1",
        conversation_id="conv-101",
        role="user",
        content="Hello",
    )
    prior_msg2 = MessageModel(
        id="m-2",
        conversation_id="conv-101",
        role="assistant",
        content="Hi there",
        message_metadata={
            "focus_entity": "6.0",
            "decision_record": {"id": "d-1", "pick": "Option A"},
        },
    )
    conversation_service.list_messages.return_value = [prior_msg1, prior_msg2]

    request = AgentRequest(
        thread_id="conv-101",
        brd_id="proj-101",
        message="What is section 6 about?",
        user=UserContext(user_id="user-101", tenant_id="user-101"),
    )

    ctx, sections = await load_context(
        request=request,
        session=session,
        conversation_service=conversation_service,
        brd_service=brd_service,
        project_service=project_service,
    )

    assert ctx.request.brd_id == "proj-101"
    assert ctx.brd_version == "0.1"
    assert len(sections) == 1
    assert sections[0].section_key == "6.0"
    assert ctx.focus_entity == "6.0"
    assert len(ctx.decision_history) == 1
    assert "User: Hello" in ctx.history_summary
    assert "Assistant: Hi there" in ctx.history_summary


@pytest.mark.asyncio
async def test_load_context_fails_closed_on_unauthorized_tenant():
    session = AsyncMock()
    project_service = AsyncMock()
    # Simulate tenant access failure (e.g. user does not own project)
    project_service.get_project.side_effect = ProjectNotFoundError("Project not found")

    brd_service = AsyncMock()
    conversation_service = AsyncMock()

    request = AgentRequest(
        thread_id="conv-intruder",
        brd_id="proj-forbidden",
        message="Show me confidential data",
        user=UserContext(user_id="intruder-user", tenant_id="intruder-user"),
    )

    with pytest.raises(ProjectNotFoundError):
        await load_context(
            request=request,
            session=session,
            conversation_service=conversation_service,
            brd_service=brd_service,
            project_service=project_service,
        )

    # Assert that no BRD or conversation retrieval was performed
    brd_service.get_or_create_canonical_brd.assert_not_called()
    conversation_service.list_messages.assert_not_called()


@pytest.mark.asyncio
async def test_load_context_resumes_clarification_merging():
    session = AsyncMock()
    project_service = AsyncMock()
    project_service.get_project.return_value = ProjectModel(id="proj-clarify", user_id="user-1")

    brd_service = AsyncMock()
    draft_version = BRDVersionModel(id="ver-1", brd_id="brd-1", version_number=1, version_label="0.1", sections=[])
    brd_service.get_or_create_canonical_brd.return_value = (BRDModel(id="brd-1"), draft_version)

    clarification_data = {
        "questions": ["What is the revised date?"],
        "missing_facts": ["new_date"],
        "reason": "Missing date",
        "original_message": "We are delayed",
        "profile": "communication",
    }

    conversation_service = AsyncMock()
    assistant_clarification_msg = MessageModel(
        id="m-clarify",
        conversation_id="conv-clarify",
        role="assistant",
        content="Please provide the revised date.",
        message_metadata={"pending_clarification": clarification_data},
    )
    conversation_service.list_messages.return_value = [assistant_clarification_msg]

    request = AgentRequest(
        thread_id="conv-clarify",
        brd_id="proj-clarify",
        message="The new target date is November 15.",
        user=UserContext(user_id="user-1", tenant_id="user-1"),
    )

    ctx, _ = await load_context(
        request=request,
        session=session,
        conversation_service=conversation_service,
        brd_service=brd_service,
        project_service=project_service,
    )

    assert ctx.pending_clarification is not None
    assert ctx.pending_clarification.profile == "communication"
    assert "new_date" in ctx.user_provided_facts
    assert ctx.user_provided_facts["new_date"] == "The new target date is November 15."
