"""Deterministic helpers and data structures for BRD Final Validation recovery loop.

Provides:
- Maximum recovery cycle limits (MAX_FINAL_VALIDATION_RECOVERY_CYCLES = 3).
- Structured FinalValidationRecoveryResult subclassing FinalValidationResult.
- Candidate affected section resolution to authoritative template sections.
- Tailored, actionable rework guidance synthesis from document-level findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Optional, Sequence

from agents.brd.final_validation.agent import (
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
)
from agents.brd.state import _match_section_name
from observability.logging import get_logger

logger = get_logger(__name__)

# Bounded limit: maximum 3 final validation recovery cycles
MAX_FINAL_VALIDATION_RECOVERY_CYCLES: int = 3


@dataclass
class FinalValidationRecoveryResult(FinalValidationResult):
    """Encapsulates the final outcome of the document-level final validation recovery loop.

    Subclasses FinalValidationResult to preserve full backward and forward compatibility
    with all existing FinalValidationResult consumers, while exposing recovery cycle
    metrics and state.
    """

    recovery_cycles: int = 0
    exhausted: bool = False
    reworked_sections: list[str] = field(default_factory=list)

    @classmethod
    def from_validation_result(
        cls,
        result: FinalValidationResult,
        recovery_cycles: int = 0,
        exhausted: bool = False,
        reworked_sections: Optional[Sequence[str]] = None,
    ) -> "FinalValidationRecoveryResult":
        """Construct a FinalValidationRecoveryResult from an underlying FinalValidationResult."""
        meta = dict(result.metadata)
        meta["recovery_cycles"] = recovery_cycles
        meta["recovery_exhausted"] = exhausted
        meta["reworked_sections"] = list(reworked_sections or [])

        return cls(
            outcome=result.outcome,
            summary=result.summary,
            findings=list(result.findings),
            rework_feedback=result.rework_feedback,
            metadata=meta,
            recovery_cycles=recovery_cycles,
            exhausted=exhausted,
            reworked_sections=list(reworked_sections or []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize result including recovery fields."""
        data = super().to_dict()
        data["recovery_cycles"] = self.recovery_cycles
        data["exhausted"] = self.exhausted
        data["reworked_sections"] = list(self.reworked_sections)
        return data


def resolve_affected_sections(
    affected_candidates: Sequence[str],
    available_sections: Sequence[str],
) -> list[str]:
    """Resolve candidate section references from validation findings to canonical template section names.

    Supports:
    - Exact matching (case-insensitive)
    - Stripped numeric matching (e.g. "Introduction" -> "1. Introduction")
    - "Section X" patterns (e.g. "Section 3" -> "3. In-Scope Business Modules & Feature Groups")
    - Substring matching against section titles
    - Order preservation (following template section ordering)

    Args:
        affected_candidates: Raw section names or references extracted from findings.
        available_sections: Authoritative ordered list of template section names.

    Returns:
        list[str]: Resolved list of canonical section names in template order.
    """
    resolved: list[str] = []
    avail_list = list(available_sections)

    for cand in affected_candidates:
        cand_str = str(cand).strip()
        if not cand_str:
            continue

        matched: Optional[str] = None

        # 1. Exact match (case-insensitive)
        for sec in avail_list:
            if sec.lower() == cand_str.lower():
                matched = sec
                break

        # 2. _match_section_name match (handles numbering differences)
        if not matched:
            candidate_match = _match_section_name(cand_str, avail_list)
            if candidate_match in avail_list:
                matched = candidate_match

        # 3. "Section <N>" pattern
        if not matched:
            m = re.search(r"section\s*(\d+)", cand_str, re.IGNORECASE)
            if m:
                sec_num = m.group(1)
                for sec in avail_list:
                    if (
                        sec.startswith(f"{sec_num}.")
                        or sec.startswith(f"{sec_num} ")
                        or sec == sec_num
                    ):
                        matched = sec
                        break

        # 4. Substring / keyword match (minimum 3 chars to avoid noise)
        if not matched and len(cand_str) >= 3:
            for sec in avail_list:
                if cand_str.lower() in sec.lower():
                    matched = sec
                    break

        if matched and matched not in resolved:
            resolved.append(matched)

    # Return in template order
    return [s for s in avail_list if s in resolved]


def format_section_rework_guidance(
    section_name: str,
    validation_result: FinalValidationResult,
) -> str:
    """Format tailored, actionable rework guidance for an affected section from validation findings.

    Synthesizes document-level validation feedback and specific findings into a clear,
    actionable set of instructions for the Section Generation/Update Sub-Agent.

    Args:
        section_name: Canonical section name to format guidance for.
        validation_result: The FinalValidationResult containing findings and document feedback.

    Returns:
        str: Consolidated rework feedback text.
    """
    sec_lower = section_name.lower()

    # Find findings that mention this section
    specific_findings: list[FinalValidationFinding] = []
    for f in validation_result.findings:
        for aff in f.affected_sections:
            aff_str = str(aff).strip().lower()
            if (
                aff_str == sec_lower
                or aff_str in sec_lower
                or sec_lower in aff_str
                or _match_section_name(aff, [section_name]) == section_name
            ):
                if f not in specific_findings:
                    specific_findings.append(f)
                break

    # If no findings specifically mention this section, treat all findings as document-level context
    target_findings = specific_findings if specific_findings else list(validation_result.findings)

    lines = [
        f"[Final Validation Rework Guidance for Section: {section_name}]",
    ]
    if validation_result.summary:
        lines.append(f"Document Validation Summary: {validation_result.summary}")
    if validation_result.rework_feedback:
        lines.append(f"Overall Document Rework Instructions: {validation_result.rework_feedback}")

    lines.append("\nSpecific Rework Items:")
    for idx, f in enumerate(target_findings, start=1):
        cat_str = f.category.value if hasattr(f.category, "value") else str(f.category)
        sev_str = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
        lines.append(f"{idx}. [{sev_str}] {cat_str}: {f.issue}")
        if f.explanation:
            lines.append(f"   Explanation: {f.explanation}")
        if f.evidence:
            lines.append(f"   Evidence / Excerpt: {f.evidence}")
        if f.required_change:
            lines.append(f"   Required Change: {f.required_change}")

    return "\n".join(lines)
