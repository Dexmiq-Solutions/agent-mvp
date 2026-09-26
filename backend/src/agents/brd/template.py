"""Authoritative BRD template loading and section extraction utilities.

Single source of truth for loading, parsing, and extracting section structures
and requirements from brd_template.md. Decouples template parsing from individual
sub-agents (Evaluation, Section Generation, Validation).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

from observability.logging import get_logger

logger = get_logger(__name__)

BRD_TEMPLATE_FILE = "brd_template.md"


def get_brd_template_path() -> Path:
    """Resolve the absolute path to the authoritative BRD template Markdown file."""
    return Path(__file__).resolve().parent / BRD_TEMPLATE_FILE


def load_brd_template() -> str:
    """Load the authoritative BRD template from Markdown.

    Returns:
        str: Raw Markdown content of the BRD template.

    Raises:
        FileNotFoundError: If the BRD template Markdown file does not exist.
        ValueError: If the BRD template Markdown file is empty.
    """
    import sys
    agent_mod = sys.modules.get("agents.brd.agent")
    template_path = (
        agent_mod.get_brd_template_path()
        if agent_mod is not None and hasattr(agent_mod, "get_brd_template_path")
        else get_brd_template_path()
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Required BRD template file not found: {template_path}"
        )

    try:
        content = template_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read BRD template from %s: %s", template_path, exc)
        raise

    if not content:
        raise ValueError(
            f"BRD template file is empty: {template_path}"
        )

    return content


def extract_brd_sections(content: Optional[str] = None) -> list[str]:
    """Extract required top-level BRD sections dynamically from the BRD template Markdown.

    The Markdown template is the single source of truth for the required BRD structure.
    Top-level section headings define the required sections in exact document order
    without duplicating them in Python code.

    Supports:
    1. Templates where sections are level-1 headings (# Section 1, # Section 2, ...),
       with child subsections at level-2 (## 5.1 Subsection).
    2. Templates where document title is level-1 (# Doc Title) and sections are
       level-2 headings (## Section 1, ## Section 2, ...).
    3. Templates with only level-2 headings.

    Args:
        content: Optional raw Markdown string. If omitted, loaded from load_brd_template().

    Returns:
        list[str]: Ordered list of section headings extracted from the template.
    """
    raw_content = load_brd_template() if content is None else content
    if not raw_content or not raw_content.strip():
        return []

    h1_matches = [m.strip() for m in re.findall(r"^#\s+(.+)$", raw_content, re.MULTILINE) if m.strip()]
    h2_matches = [m.strip() for m in re.findall(r"^##\s+(.+)$", raw_content, re.MULTILINE) if m.strip()]

    # If there are multiple level-1 headers (e.g. Dexmiq brd_template.md),
    # they represent the authoritative top-level sections.
    if len(h1_matches) > 1:
        return h1_matches

    # If there is at most one level-1 header (e.g. document title) and level-2 headers exist,
    # the level-2 headers are the top-level sections.
    if h2_matches:
        return h2_matches

    if h1_matches:
        return h1_matches

    return []


def extract_section_template(
    section_name: str,
    template_content: Optional[str] = None,
) -> str:
    """Extract authoritative template structure for a given section from brd_template.md.

    Accurately extracts the target section and any nested child subsections while preserving
    exact Markdown formatting, tables, lists, and headings.

    Args:
        section_name: Target section name (e.g. "3. In-Scope Business Modules & Feature Groups",
            "Personas", "5. Stakeholders & Personas").
        template_content: Optional raw template Markdown. If omitted, loaded from load_brd_template().

    Returns:
        str: Exact Markdown structure for the target section.
    """
    content = load_brd_template() if template_content is None else template_content
    cleaned_name = section_name.strip()
    name_stripped = re.sub(r"^\d+[\.\)]\s*", "", cleaned_name).strip().lower()

    # Find all Markdown heading lines (# ... , ## ... , ### ...)
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    target_start = -1
    target_level = 1
    target_index = -1

    for i, m in enumerate(matches):
        heading_level = len(m.group(1))
        heading_text = m.group(2).strip()
        heading_stripped = re.sub(r"^\d+[\.\)]\s*", "", heading_text).strip().lower()

        if (
            heading_stripped == name_stripped
            or name_stripped in heading_stripped
            or heading_text.lower() == cleaned_name.lower()
        ):
            target_start = m.start()
            target_level = heading_level
            target_index = i
            break

    if target_start == -1:
        return f"# {section_name}\n\n[Content requirements for {section_name}]"

    # Target section ends at the next heading with level <= target_level
    target_end = len(content)
    for next_idx in range(target_index + 1, len(matches)):
        m_next = matches[next_idx]
        next_level = len(m_next.group(1))
        if next_level <= target_level:
            target_end = m_next.start()
            break

    section_slice = content[target_start:target_end].strip()
    return section_slice


def extract_section_requirements(
    section_name: str,
    template_content: Optional[str] = None,
) -> list[str]:
    """Extract required items / criteria for a given section from the authoritative BRD template.

    Reuses the existing template structure as the single source of truth without
    creating a parallel document schema.

    Args:
        section_name: Target section name (e.g. "Personas", "1. Purpose & Scope of This Document").
        template_content: Optional raw Markdown content of the BRD template.

    Returns:
        list[str]: Extracted requirement points for the section.
    """
    content = load_brd_template() if template_content is None else template_content
    cleaned_name = section_name.strip()
    name_stripped = re.sub(r"^\d+[\.\)]\s*", "", cleaned_name).strip().lower()

    # Find section header lines (# ... or ## ...)
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    target_start = -1
    target_end = len(content)

    for i, m in enumerate(matches):
        heading_text = m.group(2).strip()
        heading_stripped = re.sub(r"^\d+[\.\)]\s*", "", heading_text).strip().lower()

        # Check match against exact or stripped name
        if (
            heading_stripped == name_stripped
            or name_stripped in heading_stripped
            or heading_text.lower() == cleaned_name.lower()
        ):
            target_start = m.end()
            if i + 1 < len(matches):
                target_end = matches[i + 1].start()
            break

    if target_start == -1:
        # Fallback: general requirements for the section
        return [f"Complete requirements and specifications for {section_name}"]

    section_body = content[target_start:target_end].strip()

    # Extract bullet points
    bullets = re.findall(r"^[*-]\s+(.+)$", section_body, re.MULTILINE)
    if bullets:
        return [b.strip() for b in bullets if b.strip()]

    # If table exists, extract table header columns
    table_headers = re.findall(r"^\|(.+)\|$", section_body, re.MULTILINE)
    if table_headers:
        columns = [c.strip() for c in table_headers[0].split("|") if c.strip() and "---" not in c]
        if columns:
            return [f"Structured table containing: {', '.join(columns)}"]

    # Fallback to non-empty lines
    lines = [
        line.strip()
        for line in section_body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if lines:
        return lines

    return [f"Requirements for {section_name}"]
