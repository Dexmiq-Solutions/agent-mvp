"""Deterministic reference resolution implementation for Step 1B of Phase 1.

Identifies and resolves section references, numbered sub-points, requirement identifiers,
and contextual pronouns against the canonical BRD section tree and conversation focus entity.
Categorizes resolutions strictly into:
- exact: unambiguous direct match to a section key, order_index, or requirement id.
- inferred: single resolution derived from conversation focus_entity or context.
- ambiguous: multiple plausible candidates, or references to nonexistent entities.
"""

from __future__ import annotations

import re
from typing import Optional

from agents.recommendation.contracts import (
    RequestContext,
    ResolvedReference,
)
from models.brd import BRDSectionModel
from observability.logging import get_logger

logger = get_logger(__name__)

# Regular expression patterns for reference extraction
POINT_OF_SECTION_PATTERN = re.compile(
    r"(?i)\b(?:point|item)\s+(\d+)\s+of\s+sec(?:tion)?\s*(\d+(?:\.\d+)*)\b"
)
SECTION_NUM_PATTERN = re.compile(
    r"(?i)\bsec(?:tion)?\s*(\d+(?:\.\d+)*)\b"
)
REQUIREMENT_ID_PATTERN = re.compile(
    r"(?i)\b(REQ-\d+|FR-\d+(?:\.\d+)*|BRD-\d+)\b"
)
PRONOUN_PATTERN = re.compile(
    r"(?i)\b(that requirement|the previous decision|this requirement|the current requirement|that section|the above|it|this|these)\b"
)


def _find_section_by_number(
    num_str: str,
    sections: list[BRDSectionModel],
) -> list[BRDSectionModel]:
    """Find matching sections by numeric identifier, order index, or section key."""
    matches: list[BRDSectionModel] = []
    clean_num = num_str.strip()

    for sec in sections:
        # Match order index (e.g. '6' -> 6th section)
        if clean_num.isdigit() and sec.order_index == int(clean_num):
            matches.append(sec)
            continue

        # Match section key exact or prefix
        sec_key = (sec.section_key or "").strip()
        if sec_key == clean_num or sec_key.startswith(f"{clean_num}.") or sec_key.startswith(f"section_{clean_num}"):
            matches.append(sec)
            continue

        # Match title leading numbers (e.g. '6. Security and Compliance')
        sec_title = (sec.title or "").strip()
        if sec_title.startswith(f"{clean_num}.") or sec_title.startswith(f"{clean_num} "):
            matches.append(sec)
            continue

    # Deduplicate matches by id
    seen = set()
    deduped = []
    for m in matches:
        if m.id not in seen:
            seen.add(m.id)
            deduped.append(m)
    return deduped


def _find_section_by_keyword(
    keyword: str,
    sections: list[BRDSectionModel],
) -> list[BRDSectionModel]:
    """Find matching sections by title keyword."""
    clean_kw = keyword.strip().lower()
    matches: list[BRDSectionModel] = []
    for sec in sections:
        title_lower = (sec.title or "").lower()
        key_lower = (sec.section_key or "").lower()
        if clean_kw in title_lower or clean_kw in key_lower:
            matches.append(sec)
    return matches


def resolve_references(
    context: RequestContext,
    sections: list[BRDSectionModel],
) -> RequestContext:
    """Execute Step 1B: Deterministically resolve references in the user message.

    Args:
        context: Active RequestContext from Step 1A.
        sections: Canonical list of BRDSectionModel records.

    Returns:
        Updated RequestContext with populated resolved_references and updated focus_entity.
    """
    message = context.request.message
    resolved: list[ResolvedReference] = []
    updated_focus_entity = context.focus_entity

    # 1. Check for 'point X of section Y'
    point_matches = list(POINT_OF_SECTION_PATTERN.finditer(message))
    for pm in point_matches:
        raw_text = pm.group(0)
        point_num = pm.group(1)
        sec_num = pm.group(2)

        matched_secs = _find_section_by_number(sec_num, sections)
        if len(matched_secs) == 1:
            target_sec = matched_secs[0]
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=target_sec.section_key,
                    confidence="exact",
                    candidates=[target_sec.section_key],
                )
            )
            updated_focus_entity = target_sec.section_key
        elif len(matched_secs) > 1:
            cands = [s.section_key for s in matched_secs]
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=cands,
                )
            )
        else:
            # Nonexistent section reference (Option 1 of Open Decisions)
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=[],
                )
            )

    # 2. Check for standalone 'section Y' (excluding spans already matched by point of section)
    sec_matches = list(SECTION_NUM_PATTERN.finditer(message))
    for sm in sec_matches:
        raw_text = sm.group(0)
        start_idx = sm.start()
        # Skip if within an already processed point_of_section span
        if any(pm.start() <= start_idx <= pm.end() for pm in point_matches):
            continue

        sec_num = sm.group(1)
        matched_secs = _find_section_by_number(sec_num, sections)
        if len(matched_secs) == 1:
            target_sec = matched_secs[0]
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=target_sec.section_key,
                    confidence="exact",
                    candidates=[target_sec.section_key],
                )
            )
            updated_focus_entity = target_sec.section_key
        elif len(matched_secs) > 1:
            cands = [s.section_key for s in matched_secs]
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=cands,
                )
            )
        else:
            # Nonexistent section reference
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=[],
                )
            )

    # 3. Check for requirement IDs (e.g. REQ-01, FR-2.1)
    req_matches = list(REQUIREMENT_ID_PATTERN.finditer(message))
    for rm in req_matches:
        raw_text = rm.group(0)
        # Search for requirement in section content or keys
        matched_secs = []
        for sec in sections:
            if raw_text.lower() in (sec.content or "").lower() or raw_text.lower() in (sec.section_key or "").lower():
                matched_secs.append(sec)

        if len(matched_secs) == 1:
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=matched_secs[0].section_key,
                    confidence="exact",
                    candidates=[matched_secs[0].section_key],
                )
            )
            updated_focus_entity = matched_secs[0].section_key
        elif len(matched_secs) > 1:
            cands = [s.section_key for s in matched_secs]
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=cands,
                )
            )
        else:
            resolved.append(
                ResolvedReference(
                    raw_text=raw_text,
                    section_id=None,
                    confidence="ambiguous",
                    candidates=[],
                )
            )

    # 4. Check for pronouns or deictic references if no explicit section reference was found
    if not resolved:
        pronoun_match = PRONOUN_PATTERN.search(message)
        if pronoun_match:
            raw_text = pronoun_match.group(0)
            if context.focus_entity:
                # Find matching section for focus entity
                matched_focus = [
                    s for s in sections
                    if s.section_key == context.focus_entity or (s.title and context.focus_entity in s.title)
                ]
                sec_id = matched_focus[0].section_key if matched_focus else context.focus_entity
                resolved.append(
                    ResolvedReference(
                        raw_text=raw_text,
                        section_id=sec_id,
                        confidence="inferred",
                        candidates=[sec_id],
                    )
                )
            else:
                resolved.append(
                    ResolvedReference(
                        raw_text=raw_text,
                        section_id=None,
                        confidence="ambiguous",
                        candidates=[],
                    )
                )

    context.resolved_references = resolved
    if updated_focus_entity:
        context.focus_entity = updated_focus_entity

    return context
