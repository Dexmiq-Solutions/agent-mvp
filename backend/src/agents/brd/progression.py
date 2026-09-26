"""Deterministic BRD section progression and completion detection.

Provides authoritative, template-driven section progression without LLM intervention:
- Determines next section strictly based on template order
- Tracks section completion via BRDSectionStatus.COMPLETED
- Activates next section as IN_PROGRESS
- Detects when all top-level sections have completed
- Derives remaining sections dynamically from template order and section progress
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from agents.brd.state import BRDAgentState

from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SectionProgressionResult:
    """Lightweight result representing a deterministic section progression transition.

    Attributes:
        current_section: Current active section after progression (None if all completed).
        completed_section: Section that was just completed and transitioned from.
        next_section: Next section in template order (None if all completed).
        has_remaining_sections: Whether additional sections remain to be processed.
        section_processing_complete: True when every top-level template section is COMPLETED.
        remaining_sections: Dynamically derived list of uncompleted template sections.
        error: Optional error message if progression could not be performed.
    """

    current_section: Optional[str] = None
    completed_section: Optional[str] = None
    next_section: Optional[str] = None
    has_remaining_sections: bool = False
    section_processing_complete: bool = False
    remaining_sections: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize progression result to dictionary."""
        return {
            "current_section": self.current_section,
            "completed_section": self.completed_section,
            "next_section": self.next_section,
            "has_remaining_sections": self.has_remaining_sections,
            "section_processing_complete": self.section_processing_complete,
            "remaining_sections": list(self.remaining_sections),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionProgressionResult":
        """Deserialize dictionary into SectionProgressionResult."""
        return cls(
            current_section=data.get("current_section"),
            completed_section=data.get("completed_section"),
            next_section=data.get("next_section"),
            has_remaining_sections=data.get("has_remaining_sections", False),
            section_processing_complete=data.get("section_processing_complete", False),
            remaining_sections=list(data.get("remaining_sections", [])),
            error=data.get("error"),
        )


from agents.brd.state import BRDSectionStatus, _match_section_name


def get_remaining_sections(
    state: BRDAgentState,
    template_sections: Optional[Sequence[str]] = None,
) -> list[str]:
    """Dynamically derive remaining (uncompleted) sections from template order and state.

    Args:
        state: Active BRDAgentState working state.
        template_sections: Optional ordered template sections override. Defaults to state.template_sections.

    Returns:
        list[str]: Ordered list of sections in template order that have not yet reached COMPLETED status.
    """
    sections = (
        list(template_sections)
        if template_sections is not None
        else list(state.template_sections)
    )
    remaining: list[str] = []
    for sec in sections:
        status = state.get_section_status(sec)
        if status != BRDSectionStatus.COMPLETED:
            remaining.append(sec)
    return remaining


def is_section_processing_complete(
    state: BRDAgentState,
    template_sections: Optional[Sequence[str]] = None,
) -> bool:
    """Deterministically check whether every top-level template section has reached COMPLETED.

    Args:
        state: Active BRDAgentState working state.
        template_sections: Optional ordered template sections override. Defaults to state.template_sections.

    Returns:
        bool: True if template has sections and all sections are COMPLETED, False otherwise.
    """
    sections = (
        list(template_sections)
        if template_sections is not None
        else list(state.template_sections)
    )
    if not sections:
        return False
    return all(
        state.get_section_status(s) == BRDSectionStatus.COMPLETED
        for s in sections
    )


def determine_next_section(
    current_section: str,
    template_sections: Sequence[str],
) -> Optional[str]:
    """Locate the current section in template order and return the next section.

    Args:
        current_section: Name of the current section.
        template_sections: Authoritative ordered list of top-level sections from template.

    Returns:
        Optional[str]: The next section name in template order, or None if current_section is the final section.

    Raises:
        ValueError: If template_sections is empty or current_section is not found in template_sections.
    """
    if not template_sections:
        raise ValueError("Template contains no top-level sections.")

    canonical = _match_section_name(current_section, template_sections)
    if canonical not in template_sections:
        raise ValueError(
            f"Current section '{current_section}' does not exist in template sections: {list(template_sections)}"
        )

    idx = list(template_sections).index(canonical)
    if idx + 1 < len(template_sections):
        return template_sections[idx + 1]
    return None


def progress_to_next_section(
    state: BRDAgentState,
    template_sections: Optional[Sequence[str]] = None,
    project_id: Optional[str] = None,
) -> SectionProgressionResult:
    """Deterministically progress from the completed current section to the next section.

    Workflow:
    1. Validate ordered template sections exist.
    2. Check if the BRD is already completely finished (idempotent, safe).
    3. Identify current section and verify it exists in template.
    4. Verify current section is COMPLETED (must not advance uncompleted sections).
    5. Determine next section in authoritative template order.
    6. If another section exists: set as current_section, mark IN_PROGRESS, return result.
    7. If no section remains: mark section processing complete, set current_section to None, return result.

    Args:
        state: Active BRDAgentState working context.
        template_sections: Optional ordered template sections override. Defaults to state.template_sections.
        project_id: Optional project identifier for structured logging.

    Returns:
        SectionProgressionResult: Progression outcome detailing transition and completion state.

    Raises:
        ValueError: If template contains no sections, current_section is missing,
            current_section is not in template, or current_section is not COMPLETED.
    """
    resolved_project_id = project_id or state.metadata.get("project_id", "unknown")
    sections = (
        list(template_sections)
        if template_sections is not None
        else list(state.template_sections)
    )

    # Error case 1: Template contains no top-level sections
    if not sections:
        error_msg = "Template contains no top-level sections."
        logger.error(
            "BRD section progression error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # Check if current section is set
    current_sec = state.current_section
    if not current_sec or not current_sec.strip():
        # Safe case: if already completely finished and no active section, return complete safely
        if is_section_processing_complete(state, sections):
            logger.info(
                "BRD section processing already completed (total_sections: %d, project_id: %s)",
                len(sections),
                resolved_project_id,
            )
            return SectionProgressionResult(
                current_section=None,
                completed_section=None,
                next_section=None,
                has_remaining_sections=False,
                section_processing_complete=True,
                remaining_sections=[],
            )
        error_msg = "Cannot progress section: current_section is not set in agent state."
        logger.error(
            "BRD section progression error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # Error case 3: Current section is not in template
    canonical_current = _match_section_name(current_sec, sections)
    if canonical_current not in sections:
        error_msg = f"Current section '{current_sec}' does not exist in template sections: {sections}"
        logger.error(
            "BRD section progression error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # Error case 4: Current section is not completed
    current_status = state.get_section_status(canonical_current)
    if current_status != BRDSectionStatus.COMPLETED:
        error_msg = (
            f"Cannot advance progression: current section '{canonical_current}' "
            f"is not completed (status: {current_status.value if current_status else 'None'})."
        )
        logger.error(
            "BRD section progression error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # Lifecycle Event: Section completed & Progression started
    logger.info(
        "BRD section completed: %s (project_id: %s)",
        canonical_current,
        resolved_project_id,
    )
    logger.info(
        "BRD section progression started (current: %s, total_sections: %d, project_id: %s)",
        canonical_current,
        len(sections),
        resolved_project_id,
    )

    current_idx = sections.index(canonical_current)

    # Determine if next section exists in template order
    if current_idx + 1 < len(sections):
        next_sec = sections[current_idx + 1]
        state.set_current_section(next_sec, auto_in_progress=True)
        remaining = get_remaining_sections(state, sections)

        # Lifecycle Event: Next section selected
        logger.info(
            "BRD progressing to next section: %s completed; progressing to %s (index: %d/%d, project_id: %s)",
            canonical_current,
            next_sec,
            current_idx + 2,
            len(sections),
            resolved_project_id,
        )

        result = SectionProgressionResult(
            current_section=next_sec,
            completed_section=canonical_current,
            next_section=next_sec,
            has_remaining_sections=True,
            section_processing_complete=False,
            remaining_sections=remaining,
        )
    else:
        # Final section completed: no remaining sections
        state.set_current_section(None)
        remaining = get_remaining_sections(state, sections)
        complete = is_section_processing_complete(state, sections)

        # Lifecycle Event: Section processing completed
        logger.info(
            "BRD section processing completed (final_section: %s, total_sections: %d, project_id: %s)",
            canonical_current,
            len(sections),
            resolved_project_id,
        )

        result = SectionProgressionResult(
            current_section=None,
            completed_section=canonical_current,
            next_section=None,
            has_remaining_sections=len(remaining) > 0,
            section_processing_complete=complete,
            remaining_sections=remaining,
        )

    return result


def initialize_progression(
    state: BRDAgentState,
    template_sections: Optional[Sequence[str]] = None,
    project_id: Optional[str] = None,
) -> SectionProgressionResult:
    """Initialize or resume progression by identifying the active or next unstarted section.

    If current_section is already set and not COMPLETED, keeps it active.
    Otherwise finds the first uncompleted section in template order, marks it IN_PROGRESS,
    and returns the progression result.

    Args:
        state: Active BRDAgentState working context.
        template_sections: Optional ordered template sections override. Defaults to state.template_sections.
        project_id: Optional project identifier for logging.

    Returns:
        SectionProgressionResult: Current progression state.
    """
    resolved_project_id = project_id or state.metadata.get("project_id", "unknown")
    sections = (
        list(template_sections)
        if template_sections is not None
        else list(state.template_sections)
    )
    if not sections:
        raise ValueError("Template contains no top-level sections.")

    if is_section_processing_complete(state, sections):
        return SectionProgressionResult(
            current_section=None,
            completed_section=state.current_section,
            next_section=None,
            has_remaining_sections=False,
            section_processing_complete=True,
            remaining_sections=[],
        )

    # If current_section is set and not completed, maintain it
    if state.current_section:
        canonical = _match_section_name(state.current_section, sections)
        if canonical in sections and state.get_section_status(canonical) != BRDSectionStatus.COMPLETED:
            if state.get_section_status(canonical) == BRDSectionStatus.NOT_STARTED:
                state.update_section_status(canonical, BRDSectionStatus.IN_PROGRESS)
            return SectionProgressionResult(
                current_section=canonical,
                completed_section=None,
                next_section=canonical,
                has_remaining_sections=True,
                section_processing_complete=False,
                remaining_sections=get_remaining_sections(state, sections),
            )

    # Find the first uncompleted section in template order
    remaining = get_remaining_sections(state, sections)
    if remaining:
        first_uncompleted = remaining[0]
        state.set_current_section(first_uncompleted, auto_in_progress=True)
        logger.info(
            "BRD section progression initialized: starting with %s (project_id: %s)",
            first_uncompleted,
            resolved_project_id,
        )
        return SectionProgressionResult(
            current_section=first_uncompleted,
            completed_section=None,
            next_section=first_uncompleted,
            has_remaining_sections=True,
            section_processing_complete=False,
            remaining_sections=remaining,
        )

    return SectionProgressionResult(
        current_section=None,
        completed_section=None,
        next_section=None,
        has_remaining_sections=False,
        section_processing_complete=True,
        remaining_sections=[],
    )
