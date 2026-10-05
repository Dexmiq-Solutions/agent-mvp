"""Deterministic BRD Document Assembly.

Provides authoritative, template-driven document assembly without LLM intervention:
- Verifies all required top-level template sections are COMPLETED before assembly
- Combines sections in exact authoritative template order
- Preserves validated section content, formatting, tables, lists, and requirement IDs
- Formats section boundaries and prevents duplicate headings
- Produces a single complete BRD document stored in Agent State
- Implements safe, idempotent re-assembly
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from agents.brd.state import BRDAgentState

from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BRDAssemblyResult:
    """Structured result representing deterministic BRD document assembly.

    Attributes:
        assembled_document: Complete Markdown BRD document assembled from completed sections.
        sections_assembled: Ordered list of section names included in the assembly.
        section_count: Total number of sections assembled.
        assembly_complete: Whether assembly completed successfully.
        metadata: Extensible metadata dictionary (e.g. project_id, assembled_at, length).
        error: Optional error message if assembly could not be performed.
    """

    assembled_document: str = ""
    sections_assembled: list[str] = field(default_factory=list)
    section_count: int = 0
    assembly_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Alias for assembly_complete."""
        return self.assembly_complete

    def to_dict(self) -> dict[str, Any]:
        """Serialize assembly result to dictionary."""
        return {
            "assembled_document": self.assembled_document,
            "sections_assembled": list(self.sections_assembled),
            "section_count": self.section_count,
            "assembly_complete": self.assembly_complete,
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BRDAssemblyResult":
        """Deserialize dictionary into BRDAssemblyResult."""
        return cls(
            assembled_document=data.get("assembled_document", ""),
            sections_assembled=list(data.get("sections_assembled", [])),
            section_count=data.get("section_count", 0),
            assembly_complete=data.get("assembly_complete", False),
            metadata=dict(data.get("metadata", {})),
            error=data.get("error"),
        )


from agents.brd.state import BRDSectionStatus, _match_section_name
from agents.brd.template import extract_brd_sections, load_brd_template


def get_heading_prefix_for_section(
    section_name: str,
    template_content: Optional[str] = None,
) -> str:
    """Determine the Markdown heading prefix ('#' or '##') for a section from the template.

    Args:
        section_name: Name of the section.
        template_content: Optional raw template Markdown string.

    Returns:
        str: Heading prefix, defaults to '#' if not explicitly found in template.
    """
    tmpl = template_content
    if tmpl is None:
        try:
            tmpl = load_brd_template()
        except Exception:
            tmpl = None

    if tmpl:
        pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        cleaned_sec = section_name.strip()
        sec_stripped = re.sub(r"^\d+[\.\)]\s*", "", cleaned_sec).strip().lower()

        for match in pattern.finditer(tmpl):
            h_prefix = match.group(1)
            h_title = match.group(2).strip()
            h_stripped = re.sub(r"^\d+[\.\)]\s*", "", h_title).strip().lower()

            if (
                h_title.lower() == cleaned_sec.lower()
                or (sec_stripped and h_stripped == sec_stripped)
            ):
                return h_prefix

    return "#"


def format_section_for_assembly(
    section_name: str,
    content: str,
    template_content: Optional[str] = None,
) -> str:
    """Format and preserve section content for document assembly.

    Ensures section boundaries and headings are preserved:
    - If content already starts with the top-level section heading (e.g. '# 1. Introduction'
      or '# Introduction'), preserves it as-is without duplicating headings.
    - If content lacks the top-level heading (e.g. raw text or starting with a subsection
      like '## 5.1 Personas'), deterministically prepends the appropriate template heading.
    - Preserves all tables, bullets, markdown formatting, and inner text exactly as generated.

    Args:
        section_name: Canonical section name.
        content: Section content to format.
        template_content: Optional template content for resolving heading levels.

    Returns:
        str: Cleanly formatted section content ready for assembly.
    """
    cleaned = content.strip()
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    first_line = lines[0].strip() if lines else ""

    # Check if first line is a Markdown header matching the section
    m = re.match(r"^(#{1,6})\s+(.+)$", first_line)
    if m:
        h_text = m.group(2).strip()

        # Exact match
        if h_text.lower() == section_name.strip().lower():
            return cleaned

        # Number-stripped match
        sec_stripped = re.sub(r"^\d+[\.\)]\s*", "", section_name.strip()).strip().lower()
        h_stripped = re.sub(r"^\d+[\.\)]\s*", "", h_text).strip().lower()

        if sec_stripped and (h_stripped == sec_stripped or sec_stripped in h_stripped or h_stripped in sec_stripped):
            # Check if this is a nested subsection (e.g. "5.1" when target is "5")
            sec_num_match = re.match(r"^(\d+)[\.\)]", section_name.strip())
            h_num_match = re.match(r"^(\d+(?:\.\d+)+)[\.\)]", h_text)
            if not (sec_num_match and h_num_match and h_num_match.group(1) != sec_num_match.group(1)):
                return cleaned

    # Top-level heading is not present; prepend authoritative heading
    prefix = get_heading_prefix_for_section(section_name, template_content=template_content)
    return f"{prefix} {section_name}\n\n{cleaned}"


def assemble_brd_document(
    state: BRDAgentState,
    template_sections: Optional[Sequence[str]] = None,
    template_content: Optional[str] = None,
    project_id: Optional[str] = None,
) -> BRDAssemblyResult:
    """Deterministically assemble all completed BRD sections into one complete document.

    Workflow:
    1. Resolve authoritative ordered top-level template sections.
    2. Verify template has valid sections (explicit error if empty).
    3. Verify all required sections have reached COMPLETED status (explicit error if incomplete).
    4. For each section in template order, retrieve validated content from state (explicit error if missing/empty).
    5. Format and concatenate sections preserving boundaries and formatting.
    6. Store assembled document and result in Agent State (idempotent, no appending).
    7. Emit lifecycle logging events.

    Args:
        state: Active BRDAgentState working state.
        template_sections: Optional ordered template sections override. Defaults to state.template_sections.
        template_content: Optional raw template Markdown. Defaults to load_brd_template().
        project_id: Optional project identifier for structured logging.

    Returns:
        BRDAssemblyResult: Assembled document and metadata.

    Raises:
        ValueError: If template has no sections, section processing is incomplete,
            or any completed section has missing/empty content.
    """
    resolved_project_id = project_id or state.metadata.get("project_id", "unknown")

    # 1. Resolve ordered sections
    if template_sections is not None:
        sections = list(template_sections)
    elif state.template_sections:
        sections = list(state.template_sections)
    else:
        sections = extract_brd_sections(template_content)

    # 2. Template sections check
    if not sections:
        error_msg = "Cannot assemble BRD: template contains no top-level sections."
        logger.error(
            "BRD assembly error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # 3. Assembly readiness check (Incomplete Section Processing)
    from agents.brd.progression import is_section_processing_complete
    if not is_section_processing_complete(state, sections):
        incomplete = [
            s for s in sections
            if state.get_section_status(s) != BRDSectionStatus.COMPLETED
        ]
        error_msg = (
            f"Cannot assemble BRD: section processing is incomplete. "
            f"Uncompleted sections ({len(incomplete)}/{len(sections)}): {incomplete}"
        )
        logger.error(
            "BRD assembly error: %s (project_id: %s)",
            error_msg,
            resolved_project_id,
        )
        raise ValueError(error_msg)

    # Lifecycle Event: Assembly started
    logger.info(
        "BRD assembly started (project_id: %s, total_sections: %d, completed_sections: %d)",
        resolved_project_id,
        len(sections),
        len(sections),
    )

    assembled_parts: list[str] = []
    sections_assembled: list[str] = []

    # 4. Retrieve and format section content in exact template order
    for idx, sec in enumerate(sections):
        order_num = idx + 1
        canonical_sec = _match_section_name(sec, state._available_section_keys())
        content = state.get_section_content(canonical_sec)

        # Missing or empty section content check
        if content is None or not content.strip():
            error_msg = (
                f"Cannot assemble BRD: completed section '{sec}' has missing or empty content."
            )
            logger.error(
                "BRD assembly error: %s (project_id: %s)",
                error_msg,
                resolved_project_id,
            )
            raise ValueError(error_msg)

        formatted_section = format_section_for_assembly(
            section_name=sec,
            content=content,
            template_content=template_content,
        )
        assembled_parts.append(formatted_section)
        sections_assembled.append(sec)

        # Lifecycle Event: Section included
        logger.info(
            "BRD section included in assembly (section: %s, order: %d/%d, project_id: %s)",
            sec,
            order_num,
            len(sections),
            resolved_project_id,
        )

    # 5. Concatenate sections into the complete BRD document
    assembled_document = "\n\n".join(assembled_parts).strip()

    # Lifecycle Event: Assembly completed
    logger.info(
        "BRD assembly completed (project_id: %s, sections_assembled: %d, document_length: %d)",
        resolved_project_id,
        len(sections_assembled),
        len(assembled_document),
    )

    result = BRDAssemblyResult(
        assembled_document=assembled_document,
        sections_assembled=sections_assembled,
        section_count=len(sections_assembled),
        assembly_complete=True,
        metadata={
            "project_id": resolved_project_id,
            "character_count": len(assembled_document),
            "assembled_at": time.time(),
        },
    )

    # 6. Store in Agent State (replaces prior assembled document deterministically)
    state.set_assembled_brd(assembled_document)
    state.set_assembly_result(result)

    return result
