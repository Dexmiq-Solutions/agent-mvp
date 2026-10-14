"""Application service for persistent, versioned BRD lifecycle management.

Handles:
- One canonical BRD per project (PostgreSQL tenant isolation).
- In-place working draft updates during conversation turns without version proliferation.
- Structured section persistence and deterministic Markdown derivation.
- Point-in-time immutable version snapshots.
- Single database transaction consistency.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
import uuid

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agents.brd.assembly import assemble_markdown_from_sections, format_section_for_assembly
from agents.brd.state import BRDAgentState, BRDSectionStatus, _match_section_name
from agents.brd.template import extract_brd_sections, load_brd_template
from exceptions.brd import (
    BRDNotFoundError,
    BRDSectionNotFoundError,
    BRDVersionImmutableError,
    BRDVersionNotFoundError,
    InvalidBRDDataError,
)
from models.brd import BRDModel, BRDSectionModel, BRDVersionModel, BRDVersionStatus
from observability.logging import get_logger
from services.project_service import ProjectService

logger = get_logger(__name__)


class BRDService:
    """Service orchestrating canonical BRD persistence and version snapshots."""

    def __init__(
        self,
        session: AsyncSession,
        project_service: Optional[ProjectService] = None,
    ) -> None:
        """Initialize BRDService with active asynchronous session."""
        self._session = session
        self._project_service = project_service or ProjectService(session=session)

    async def get_or_create_canonical_brd(
        self,
        project_id: str,
    ) -> tuple[BRDModel, BRDVersionModel]:
        """Retrieve or initialize the canonical BRD and its active working draft.

        Enforces project existence and idempotent initialization. If no canonical
        BRD exists for the project, initializes BRDModel, an initial working draft
        (version 1, label '0.1', is_current=True), and 16 template sections.

        Args:
            project_id: Owning project identifier.

        Returns:
            Tuple of (BRDModel, current working BRDVersionModel).
        """
        # 1. Enforce project isolation and existence
        await self._project_service.get_project(project_id)

        # 2. Query existing canonical BRD
        stmt = (
            select(BRDModel)
            .options(
                selectinload(BRDModel.versions).selectinload(BRDVersionModel.sections)
            )
            .where(BRDModel.project_id == project_id)
        )
        result = await self._session.execute(stmt)
        brd = result.scalar_one_or_none()

        if brd is not None:
            # Locate active working draft
            working_version = next(
                (v for v in brd.versions if v.is_current),
                None,
            )
            if working_version is not None:
                return brd, working_version

            # Fallback: if all versions were snapshotted without creating a next draft
            max_v = max((v.version_number for v in brd.versions), default=0)
            next_num = max_v + 1
            working_version = BRDVersionModel(
                brd_id=brd.id,
                version_number=next_num,
                version_label=f"0.{next_num}",
                is_current=True,
                status=BRDVersionStatus.DRAFT.value,
            )
            self._session.add(working_version)
            await self._session.flush()
            return brd, working_version

        # 3. Create canonical BRD
        brd = BRDModel(project_id=project_id)
        self._session.add(brd)

        # 4. Create initial working draft
        working_version = BRDVersionModel(
            brd=brd,
            version_number=1,
            version_label="0.1",
            is_current=True,
            status=BRDVersionStatus.DRAFT.value,
        )
        self._session.add(working_version)

        # 5. Initialize 16 template sections in exact template order
        template_content = load_brd_template()
        template_sections = extract_brd_sections(template_content)

        for idx, sec_name in enumerate(template_sections):
            clean_name = sec_name.strip()
            section = BRDSectionModel(
                version=working_version,
                section_key=clean_name,
                title=clean_name,
                order_index=idx + 1,
                heading_level=1,
                content="",
                status="Not Started",
            )
            self._session.add(section)

        await self._session.flush()

        # Re-query with selectinload to ensure everything is cleanly loaded and attached
        stmt = (
            select(BRDModel)
            .options(
                selectinload(BRDModel.versions).selectinload(BRDVersionModel.sections)
            )
            .where(BRDModel.id == brd.id)
        )
        result = await self._session.execute(stmt)
        brd = result.scalar_one()
        working_version = next(v for v in brd.versions if v.is_current)

        logger.info(
            "Initialized canonical BRD (id: %s) and working draft (id: %s) for project '%s'",
            brd.id,
            working_version.id,
            project_id,
        )
        return brd, working_version

    async def get_current_working_version(
        self,
        project_id: str,
    ) -> Optional[BRDVersionModel]:
        """Retrieve the active working draft and all its structured sections.

        Args:
            project_id: Owning project identifier.

        Returns:
            Active working BRDVersionModel with sections loaded, or None.
        """
        stmt = (
            select(BRDVersionModel)
            .join(BRDModel, BRDModel.id == BRDVersionModel.brd_id)
            .options(selectinload(BRDVersionModel.sections))
            .where(
                BRDModel.project_id == project_id,
                BRDVersionModel.is_current.is_(True),
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def save_working_draft_from_state(
        self,
        project_id: str,
        state: BRDAgentState,
    ) -> BRDVersionModel:
        """Update the existing active working draft in-place from agent operational state.

        Does NOT create a new version snapshot. Updates structured sections and
        derives assembled Markdown consistently.

        Args:
            project_id: Owning project identifier.
            state: Active BRDAgentState containing section content and progress.

        Returns:
            Updated BRDVersionModel instance.

        Raises:
            BRDVersionImmutableError: If target version is an immutable snapshot.
        """
        brd, working_version = await self.get_or_create_canonical_brd(project_id)

        # Enforce draft mutability guard
        if not working_version.is_current or working_version.status != BRDVersionStatus.DRAFT.value:
            raise BRDVersionImmutableError(
                version_id=working_version.id,
                version_number=working_version.version_number,
                message=f"Cannot edit BRD version '{working_version.id}': version is not an active draft.",
            )

        # 1. Update structured sections
        existing_sections = {s.section_key: s for s in working_version.sections}
        available_keys = list(existing_sections.keys())

        # If sections are missing from working draft, populate them
        if not existing_sections:
            template_sections = extract_brd_sections(load_brd_template())
            for idx, sec_name in enumerate(template_sections):
                clean_name = sec_name.strip()
                new_sec = BRDSectionModel(
                    brd_version_id=working_version.id,
                    section_key=clean_name,
                    title=clean_name,
                    order_index=idx + 1,
                    heading_level=1,
                    content="",
                    status="Not Started",
                )
                self._session.add(new_sec)
                existing_sections[clean_name] = new_sec
            await self._session.flush()
            available_keys = list(existing_sections.keys())

        for sec_name, content in state.section_content.items():
            canonical_key = _match_section_name(sec_name, available_keys)
            sec_model = existing_sections.get(canonical_key)

            if sec_model is None:
                # Add section if not already present in draft
                order_num = len(existing_sections) + 1
                sec_model = BRDSectionModel(
                    brd_version_id=working_version.id,
                    section_key=canonical_key,
                    title=canonical_key,
                    order_index=order_num,
                    heading_level=1,
                    content=content,
                    status="In Progress",
                )
                self._session.add(sec_model)
                existing_sections[canonical_key] = sec_model
                available_keys.append(canonical_key)
            else:
                sec_model.content = content or ""
                sec_model.updated_at = datetime.now(timezone.utc)

            # Update status if present in state
            status_val = state.get_section_status(sec_name)
            if status_val is not None:
                sec_model.status = status_val.value if isinstance(status_val, BRDSectionStatus) else str(status_val)

            # Update rework feedback if present
            rework = state.get_rework_feedback(sec_name)
            sec_model.rework_feedback = rework

        # Update remaining section statuses from state
        for sec_name, status_val in state.section_progress.items():
            canonical_key = _match_section_name(sec_name, available_keys)
            sec_model = existing_sections.get(canonical_key)
            if sec_model is not None:
                sec_model.status = status_val.value if isinstance(status_val, BRDSectionStatus) else str(status_val)

        # 2. Derive Markdown snapshot deterministically from structured sections
        # PostgreSQL sections are the single source of truth (no state.assembled_brd fallback)
        sorted_sections = sorted(existing_sections.values(), key=lambda s: s.order_index)
        working_version.assembled_markdown = assemble_markdown_from_sections(sorted_sections)

        # 3. Update telemetry and touch timestamps
        metadata_dict = dict(working_version.version_metadata or {})
        if state.latest_assembly_result is not None:
            metadata_dict["assembly_result"] = state.latest_assembly_result.to_dict()
        if "llm_execution_summary" in state.metadata:
            metadata_dict["llm_execution_summary"] = state.metadata["llm_execution_summary"]

        working_version.version_metadata = metadata_dict
        working_version.updated_at = datetime.now(timezone.utc)
        brd.updated_at = datetime.now(timezone.utc)

        await self._session.flush()

        # Re-query working_version with selectinload so sections and derived markdown are freshly loaded
        stmt = (
            select(BRDVersionModel)
            .options(selectinload(BRDVersionModel.sections))
            .where(BRDVersionModel.id == working_version.id)
        )
        res = await self._session.execute(stmt)
        refreshed_version = res.scalar_one()

        logger.info(
            "Persisted working draft update for BRD '%s' (version: %d, sections: %d)",
            brd.id,
            refreshed_version.version_number,
            len(refreshed_version.sections),
        )
        return refreshed_version

    async def create_version_snapshot(
        self,
        project_id: str,
        snapshot_label: str,
        next_draft_label: Optional[str] = None,
    ) -> tuple[BRDVersionModel, BRDVersionModel]:
        """Atomically create a point-in-time immutable version snapshot and initialize the next draft.

        Executes under an exclusive row lock on the canonical BRD:
        1. Freezes the active working draft: is_current=False, status='snapshot', version_label=snapshot_label.
        2. Creates the next working draft: version_number=N+1, is_current=True, status='draft'.
        3. Deep-copies all section records to the new draft.

        Args:
            project_id: Owning project identifier.
            snapshot_label: Human-facing label for the frozen snapshot (e.g. '1.0').
            next_draft_label: Optional label for the next working draft (defaults to 'X.0-draft').

        Returns:
            Tuple of (frozen BRDVersionModel snapshot, newly active BRDVersionModel draft).

        Raises:
            BRDNotFoundError: If canonical BRD does not exist.
            BRDVersionNotFoundError: If active working draft does not exist.
        """
        # 1. Acquire exclusive lock on canonical BRD
        stmt = (
            select(BRDModel)
            .where(BRDModel.project_id == project_id)
            .with_for_update()
        )
        result = await self._session.execute(stmt)
        brd = result.scalar_one_or_none()

        if brd is None:
            raise BRDNotFoundError(project_id)

        # 2. Retrieve active working draft with sections
        v_stmt = (
            select(BRDVersionModel)
            .options(selectinload(BRDVersionModel.sections))
            .where(
                BRDVersionModel.brd_id == brd.id,
                BRDVersionModel.is_current.is_(True),
            )
            .with_for_update()
        )
        v_result = await self._session.execute(v_stmt)
        current_draft = v_result.scalar_one_or_none()

        if current_draft is None:
            raise BRDVersionNotFoundError(
                version_identifier="active_draft",
                project_id=project_id,
                message=f"No active working draft found for project '{project_id}' to snapshot.",
            )

        # 3. Freeze current working draft
        current_draft.is_current = False
        current_draft.status = BRDVersionStatus.SNAPSHOT.value
        current_draft.version_label = snapshot_label.strip() or current_draft.version_label
        current_draft.updated_at = datetime.now(timezone.utc)

        # 4. Insert next working draft
        next_num = current_draft.version_number + 1
        resolved_next_label = (
            next_draft_label.strip()
            if next_draft_label and next_draft_label.strip()
            else f"{next_num}.0-draft"
        )
        new_draft = BRDVersionModel(
            brd_id=brd.id,
            version_number=next_num,
            version_label=resolved_next_label,
            is_current=True,
            status=BRDVersionStatus.DRAFT.value,
            assembled_markdown=current_draft.assembled_markdown,
            version_metadata=dict(current_draft.version_metadata or {}),
        )
        self._session.add(new_draft)

        # 5. Deep-copy section records to isolate the new draft from the frozen snapshot
        for sec in current_draft.sections:
            cloned_sec = BRDSectionModel(
                version=new_draft,
                section_key=sec.section_key,
                title=sec.title,
                order_index=sec.order_index,
                heading_level=sec.heading_level,
                parent_section_id=sec.parent_section_id,
                content=sec.content,
                status=sec.status,
                rework_feedback=sec.rework_feedback,
                section_metadata=dict(sec.section_metadata or {}),
            )
            self._session.add(cloned_sec)

        brd.updated_at = datetime.now(timezone.utc)
        await self._session.flush()

        # Re-query both snapshot and new_draft with selectinload
        q_stmt = (
            select(BRDVersionModel)
            .options(selectinload(BRDVersionModel.sections))
            .where(BRDVersionModel.id.in_([current_draft.id, new_draft.id]))
        )
        res = await self._session.execute(q_stmt)
        loaded = {v.id: v for v in res.scalars().all()}

        frozen_snapshot = loaded[current_draft.id]
        active_new_draft = loaded[new_draft.id]

        logger.info(
            "Created BRD version snapshot '%s' (v%d) and initialized new working draft '%s' (v%d) for project '%s'",
            frozen_snapshot.version_label,
            frozen_snapshot.version_number,
            active_new_draft.version_label,
            active_new_draft.version_number,
            project_id,
        )
        return frozen_snapshot, active_new_draft

    async def list_version_snapshots(
        self,
        project_id: str,
    ) -> list[BRDVersionModel]:
        """List all frozen historical version snapshots for a project ordered by version_number descending."""
        stmt = (
            select(BRDVersionModel)
            .join(BRDModel, BRDModel.id == BRDVersionModel.brd_id)
            .where(
                BRDModel.project_id == project_id,
                BRDVersionModel.status == BRDVersionStatus.SNAPSHOT.value,
            )
            .order_by(BRDVersionModel.version_number.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_version_snapshot(
        self,
        project_id: str,
        version_number: int,
    ) -> BRDVersionModel:
        """Retrieve a specific historical version snapshot and its sections."""
        stmt = (
            select(BRDVersionModel)
            .join(BRDModel, BRDModel.id == BRDVersionModel.brd_id)
            .options(selectinload(BRDVersionModel.sections))
            .where(
                BRDModel.project_id == project_id,
                BRDVersionModel.version_number == version_number,
            )
        )
        result = await self._session.execute(stmt)
        version = result.scalar_one_or_none()
        if version is None:
            raise BRDVersionNotFoundError(
                version_identifier=version_number,
                project_id=project_id,
            )
        return version

    async def get_canonical_brd_with_sections(
        self,
        project_id: str,
    ) -> tuple[BRDModel, BRDVersionModel, list[BRDSectionModel]]:
        """Retrieve canonical BRD, active working draft, and all structured sections ordered by order_index.

        Initializes canonical BRD and 16 template sections idempotently if not present.
        Shared across all agents and services.
        """
        brd, working_version = await self.get_or_create_canonical_brd(project_id)
        sorted_sections = sorted(working_version.sections, key=lambda s: s.order_index)
        return brd, working_version, sorted_sections

    async def get_sections(
        self,
        project_id: str,
        version_number: Optional[int] = None,
    ) -> list[BRDSectionModel]:
        """Retrieve all structured sections for a project ordered by order_index.

        Args:
            project_id: Owning project identifier.
            version_number: Optional historical version number. If None, retrieves
                sections from the active working draft.

        Returns:
            List of BRDSectionModel ordered by order_index.
        """
        if version_number is not None:
            version = await self.get_version_snapshot(project_id, version_number)
            return sorted(version.sections, key=lambda s: s.order_index)

        _, _, sections = await self.get_canonical_brd_with_sections(project_id)
        return sections

    async def get_section(
        self,
        project_id: str,
        section_key: str,
        version_number: Optional[int] = None,
    ) -> BRDSectionModel:
        """Retrieve a specific section by section key or title.

        Uses tolerant matching (_match_section_name) so callers can pass e.g.
        "3. Target Audience" or "Target Audience" or "audience".

        Args:
            project_id: Owning project identifier.
            section_key: Key, title, or substring identifier of the section.
            version_number: Optional historical version number.

        Returns:
            BRDSectionModel matching the key.

        Raises:
            BRDSectionNotFoundError: If section cannot be found.
        """
        sections = await self.get_sections(project_id, version_number=version_number)
        available_keys = [s.section_key for s in sections]
        matched_key = _match_section_name(section_key, available_keys)

        for sec in sections:
            if sec.section_key == matched_key or sec.title == matched_key:
                return sec

        raise BRDSectionNotFoundError(
            section_key=section_key,
            project_id=project_id,
            version_number=version_number,
        )

    async def update_section_content(
        self,
        project_id: str,
        section_key: str,
        content: str,
        status: Optional[str] = None,
        rework_feedback: Optional[str] = None,
    ) -> BRDSectionModel:
        """Update a specific section's content in the canonical working draft.

        Enforces draft mutability, updates section content, deterministically
        re-derives the working draft's assembled Markdown document, and flushes
        the session. Accessible by any agent or backend service.

        Args:
            project_id: Owning project identifier.
            section_key: Section key or title to update.
            content: New content for the section.
            status: Optional new status for the section.
            rework_feedback: Optional rework feedback to record.

        Returns:
            Updated BRDSectionModel.

        Raises:
            BRDVersionImmutableError: If active version is not a draft.
            BRDSectionNotFoundError: If target section is not in the draft.
        """
        brd, working_version = await self.get_or_create_canonical_brd(project_id)

        if not working_version.is_current or working_version.status != BRDVersionStatus.DRAFT.value:
            raise BRDVersionImmutableError(
                version_id=working_version.id,
                version_number=working_version.version_number,
                message=f"Cannot edit BRD version '{working_version.id}': version is not an active draft.",
            )

        available_keys = [s.section_key for s in working_version.sections]
        matched_key = _match_section_name(section_key, available_keys)

        target_sec = next(
            (s for s in working_version.sections if s.section_key == matched_key or s.title == matched_key),
            None,
        )
        if target_sec is None:
            raise BRDSectionNotFoundError(
                section_key=section_key,
                project_id=project_id,
            )

        target_sec.content = content or ""
        target_sec.updated_at = datetime.now(timezone.utc)
        if status is not None:
            target_sec.status = status.value if isinstance(status, BRDSectionStatus) else str(status)
        if rework_feedback is not None:
            target_sec.rework_feedback = rework_feedback

        # Deterministically re-derive assembled Markdown from all sections
        sorted_sections = sorted(working_version.sections, key=lambda s: s.order_index)
        working_version.assembled_markdown = assemble_markdown_from_sections(sorted_sections)
        working_version.updated_at = datetime.now(timezone.utc)
        brd.updated_at = datetime.now(timezone.utc)

        await self._session.flush()
        return target_sec
