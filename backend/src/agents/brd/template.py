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


def is_administrative_section(section_name: str) -> bool:
    """Determine whether a section is purely administrative metadata or document mechanics.

    Administrative and document-mechanics sections do not represent project-domain knowledge
    and should not trigger external RAG retrieval or block workflow execution on missing fields.
    """
    if not section_name:
        return False
    norm = re.sub(r"^\d+[\.\)]\s*", "", section_name.strip()).strip().lower()
    return (
        "document header" in norm
        or norm == "version history"
        or "quality gate" in norm
    )


def is_metadata_or_role_item(item: str) -> bool:
    """Check whether a requirement or finding item corresponds to personnel, roles, or administrative metadata."""
    if not item:
        return False
    norm = item.strip().lower()
    metadata_markers = [
        "prepared by",
        "reviewed by",
        "approved by",
        "tech lead",
        "stakeholder",
        "business owner",
        "sme",
        "author",
        "approver",
        "client tier",
        "lifecycle phase",
        "confidentiality level",
        "distribution list",
        "document title",
        "document id",
        "engagement id",
        "date created",
        "last updated",
        "document type",
        "version history",
        "version number",
        "approval date",
    ]
    return any(re.search(r"\b" + re.escape(marker) + r"\b", norm) for marker in metadata_markers)


def is_tbd_value(val: str) -> bool:
    """Check if a string represents an acceptable TBD or pending placeholder."""
    if not val:
        return True
    norm = val.strip().lower()
    return (
        norm == "tbd"
        or norm.startswith("tbd ")
        or "tbd (" in norm
        or "suggested:" in norm
        or "pending" in norm
        or "to be determined" in norm
        or "unspecified" in norm
        or "none specified" in norm
    )


def classify_requirement_item(item: str) -> str:
    """Classify a template requirement into METADATA_ROLE, DOCUMENT_MECHANIC, or BUSINESS_REQUIREMENT."""
    if not item:
        return "BUSINESS_REQUIREMENT"
    norm = item.strip().lower()
    if is_metadata_or_role_item(norm):
        if any(m in norm for m in ["version", "date", "document id", "engagement id"]):
            return "DOCUMENT_MECHANIC"
        return "METADATA_ROLE"
    return "BUSINESS_REQUIREMENT"


def is_substantive_requirement(item: str) -> bool:
    """Check if an item is a genuine business or technical requirement rather than administrative metadata or mechanic."""
    return classify_requirement_item(item) in ("BUSINESS_REQUIREMENT", "substantive_requirement")


def is_benign_administrative_metadata_finding(finding: Any, section_name: str) -> bool:
    """Check if a validation finding is a benign administrative metadata TBD that should not block validation."""
    if finding is None:
        return False

    issue = getattr(finding, "issue", "") or ""
    explanation = getattr(finding, "explanation", "") or ""
    category = getattr(finding, "category", "") or ""
    cat_str = category if isinstance(category, str) else getattr(category, "value", str(category))

    full_text = f"{issue} {explanation}".lower()

    # Structure / template compliance findings (missing tables, missing headings) are never benign metadata TBDs
    if "template compliance" in cat_str.lower() or "structure" in cat_str.lower():
        if "tbd" not in full_text and "suggested" not in full_text:
            return False

    # 1. In administrative sections (e.g. Document Header, Version History, Quality Gate)
    if is_administrative_section(section_name):
        if is_metadata_or_role_item(issue) or any(
            re.search(r"\b" + re.escape(w) + r"\b", full_text)
            for w in ["tbd", "placeholder", "pending", "unknown", "version", "date", "suggested", "metadata"]
        ):
            return True

    # 2. In any section: only filter if specifically about an administrative role/metadata field being TBD or suggested
    meta_fields = ["prepared by", "reviewed by", "approved by", "tech lead", "document id", "engagement id", "client tier", "confidentiality level"]
    has_meta_field = any(field in full_text for field in meta_fields)
    has_tbd_concept = any(
        re.search(r"\b" + re.escape(w) + r"\b", full_text)
        for w in ["tbd", "placeholder", "pending", "unknown", "unspecified", "suggested"]
    )
    if has_meta_field and has_tbd_concept:
        return True

    return False


def is_honest_uncertainty_or_non_provided_text(text: str) -> bool:
    """Check if a string represents an honest statement of uncertainty or non-provided information."""
    if not text:
        return False
    norm = text.strip().lower()
    phrases = [
        "not provided",
        "not discussed",
        "not defined",
        "not specified",
        "to be clarified",
        "to be determined",
        "pending formal definition",
        "pending clarification",
        "not available in the available",
        "not available in available",
        "was not defined",
        "were not defined",
        "were not provided",
        "was not provided",
    ]
    return any(p in norm for p in phrases) or is_tbd_value(norm)


def is_documentation_quality_finding(finding: Any, section_name: str = "") -> bool:
    """Check if a validation finding is an ordinary documentation-quality observation
    or an observation about non-provided detail that should not block initial BRD generation.
    """
    if finding is None:
        return False

    if is_benign_administrative_metadata_finding(finding, section_name):
        return True

    issue = getattr(finding, "issue", "") if not isinstance(finding, str) else finding
    explanation = getattr(finding, "explanation", "") if not isinstance(finding, str) else ""

    full_text = f"{issue} {explanation}".lower()

    # Never treat fabricated claims, hallucinations, or explicit contradictions as benign documentation findings
    if any(k in full_text for k in ["hallucinat", "fabricated", "contradict", "unsupported claim presented as fact"]):
        return False

    # Check if finding is observing that the section honestly noted uncertainty/TBD/unprovided detail
    if any(k in full_text for k in [
        "honestly notes", "honestly states", "marked as tbd", "identified as potential capability",
        "honestly represented", "correctly notes that", "noted as not provided",
        "stated as not provided", "represented as unknown",
    ]):
        return True

    # Optional detail indicators and observations that should not block initial BRD
    doc_quality_indicators = [
        "conceptual workflow", "workflow diagram", "persona priority",
        "persona-to-module", "unresolved assumption", "product naming",
        "inquiry routing", "operational process", "detailed ai",
        "ai functionality", "ai behavior",
    ]
    if any(k in full_text for k in doc_quality_indicators):
        return True

    return False


def is_substantive_business_clarification_item(item: str) -> bool:
    """Check if an unresolved item is a genuine business requirement that warrants user clarification,
    rather than administrative metadata, document mechanic, or ordinary documentation-quality observation.
    """
    if not item or not is_substantive_requirement(item):
        return False

    norm = item.strip().lower()

    # Exclude raw placeholders
    if is_tbd_value(norm) or norm in ("not provided", "tbd", "to be clarified", "unknown", "none", "n/a"):
        return False

    # Exclude optional detail findings
    if is_documentation_quality_finding(item):
        return False

    # Exclude writing and formatting quality feedback from becoming user questions
    writing_quality_terms = ["clarity", "vague", "precision", "formatting", "grammar", "style", "readability", "structured content"]
    if any(t in norm for t in writing_quality_terms):
        return False

    return True


