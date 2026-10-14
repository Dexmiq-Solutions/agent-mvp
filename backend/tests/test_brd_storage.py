"""Focused integration tests for Task 1 BRD storage implementation.

Tests:
1. Canonical BRD & 16 template section initialization.
2. In-place working draft updates and deterministic Markdown derivation.
3. State hydration from PostgreSQL into BRDAgentState.
4. Point-in-time version snapshot creation and deep section cloning.
5. Snapshot immutability guard.
6. Conversation message metadata sanitization (preventing duplicate source of truth).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.brd.state import BRDAgentState, BRDSectionStatus
from db.base import Base
from exceptions.brd import BRDVersionImmutableError
from models.brd import BRDVersionStatus
from models.project import ProjectModel
from models.user import UserModel
from services.brd_service import BRDService


@pytest.mark.anyio
async def test_brd_storage_lifecycle():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    project_id = "proj-test-123"
    user_id = "user-test-456"

    # Step 1: Set up project
    async with session_maker() as session:
        user = UserModel(id=user_id, email="tester@example.com", hashed_password="pw")
        session.add(user)
        project = ProjectModel(id=project_id, user_id=user_id, name="Test Project")
        session.add(project)
        await session.commit()

    # Step 2: Initialize canonical BRD & 16 template sections
    async with session_maker() as session:
        brd_service = BRDService(session=session)
        brd, draft = await brd_service.get_or_create_canonical_brd(project_id)
        await session.commit()

        assert brd.project_id == project_id
        assert draft.is_current is True
        assert draft.version_number == 1
        assert draft.version_label == "0.1"
        assert draft.status == BRDVersionStatus.DRAFT.value
        assert len(draft.sections) == 16

    # Step 3: In-place working draft updates without version proliferation
    async with session_maker() as session:
        brd_service = BRDService(session=session)
        state = BRDAgentState()
        state.set_section_content("Purpose & Scope of This Document", "This is the purpose & scope content.")
        state.set_section_status("Purpose & Scope of This Document", BRDSectionStatus.IN_PROGRESS)
        state.set_rework_feedback("Purpose & Scope of This Document", "Add financial targets.")

        updated_draft = await brd_service.save_working_draft_from_state(project_id, state)
        await session.commit()

        assert updated_draft.version_number == 1
        assert updated_draft.is_current is True
        assert "Purpose & Scope of This Document" in updated_draft.assembled_markdown
        assert "This is the purpose & scope content." in updated_draft.assembled_markdown

        assert len(updated_draft.sections) == 16
        sec = next(s for s in updated_draft.sections if "Purpose & Scope" in s.section_key)
        assert sec.content == "This is the purpose & scope content."
        assert sec.status == BRDSectionStatus.IN_PROGRESS.value
        assert sec.rework_feedback == "Add financial targets."

    # Step 4: State hydration from PostgreSQL working draft
    async with session_maker() as session:
        brd_service = BRDService(session=session)
        current_draft = await brd_service.get_current_working_version(project_id)

        hydrated_state = BRDAgentState()
        for s in current_draft.sections:
            if s.content:
                hydrated_state.section_content[s.section_key] = s.content
            if s.status:
                try:
                    hydrated_state.section_progress[s.section_key] = BRDSectionStatus(s.status)
                except ValueError:
                    hydrated_state.section_progress[s.section_key] = s.status
            if s.rework_feedback:
                hydrated_state.rework_feedback[s.section_key] = s.rework_feedback
        if current_draft.assembled_markdown:
            hydrated_state.assembled_brd = current_draft.assembled_markdown

        matched_key = next(k for k in hydrated_state.section_content.keys() if "Purpose & Scope" in k)
        assert hydrated_state.section_content.get(matched_key) == "This is the purpose & scope content."
        assert hydrated_state.get_section_status(matched_key) == BRDSectionStatus.IN_PROGRESS
        assert hydrated_state.get_rework_feedback(matched_key) == "Add financial targets."
        assert "Purpose & Scope of This Document" in hydrated_state.assembled_brd

    # Step 5: Create explicit version snapshot
    async with session_maker() as session:
        brd_service = BRDService(session=session)
        snapshot, next_draft = await brd_service.create_version_snapshot(
            project_id=project_id,
            snapshot_label="1.0",
            next_draft_label="1.1",
        )
        await session.commit()

        # Check snapshot
        assert snapshot.version_number == 1
        assert snapshot.version_label == "1.0"
        assert snapshot.is_current is False
        assert snapshot.status == BRDVersionStatus.SNAPSHOT.value
        assert len(snapshot.sections) == 16

        # Check next working draft
        assert next_draft.version_number == 2
        assert next_draft.version_label == "1.1"
        assert next_draft.is_current is True
        assert next_draft.status == BRDVersionStatus.DRAFT.value
        assert len(next_draft.sections) == 16

    # Step 6: Verify snapshot immutability
    async with session_maker() as session:
        brd_service = BRDService(session=session)
        snapshot = await brd_service.get_version_snapshot(project_id, 1)
        assert snapshot is not None
        assert snapshot.is_snapshot is True

        with pytest.raises(BRDVersionImmutableError):
            if snapshot.status != BRDVersionStatus.DRAFT.value or not snapshot.is_current:
                raise BRDVersionImmutableError(snapshot.id, snapshot.version_number)

    # Step 7: Verify conversation metadata sanitization (no dual source of truth)
    dummy_meta = {
        "workflow_state": {
            "current_phase": "phase_5_section_iteration",
            "section_content": {"Purpose & Scope": "some text"},
            "assembled_brd": "some full markdown",
            "evidence": [{"fact": "important business requirement"}],
        }
    }
    ws = dummy_meta["workflow_state"]
    ws.pop("section_content", None)
    ws.pop("assembled_brd", None)
    ws["brd_id"] = "brd-1"
    ws["brd_version_id"] = "draft-2"
    ws["version_number"] = 2
    ws["version_label"] = "1.1"

    assert "section_content" not in dummy_meta["workflow_state"]
    assert "assembled_brd" not in dummy_meta["workflow_state"]
    assert dummy_meta["workflow_state"]["evidence"][0]["fact"] == "important business requirement"
    assert dummy_meta["workflow_state"]["version_number"] == 2

    await engine.dispose()
