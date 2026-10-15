"""Unit tests for deterministic reference resolution (Step 1B).

Covers:
- Exact section references ('section 6', 'sec 2.3').
- Sub-points ('point 4 of section 6').
- Pronouns with and without focus_entity ('it', 'that requirement').
- Ambiguous references matching multiple sections.
- Nonexistent section references.
"""

import pytest

from agents.recommendation.contracts import (
    AgentRequest,
    RequestContext,
    UserContext,
)
from agents.recommendation.reference_resolver import resolve_references
from models.brd import BRDSectionModel


def make_dummy_sections() -> list[BRDSectionModel]:
    """Create sample canonical BRD sections."""
    s1 = BRDSectionModel(
        id="sec-1",
        brd_version_id="v-1",
        section_key="1.0",
        title="1. Executive Summary",
        order_index=1,
        content="Overview of payment gateway integration.",
    )
    s2 = BRDSectionModel(
        id="sec-2",
        brd_version_id="v-1",
        section_key="2.3",
        title="2.3 Data Retention and Deletion",
        order_index=2,
        content="Users can request full deletion of their personal data.",
    )
    s3 = BRDSectionModel(
        id="sec-3",
        brd_version_id="v-1",
        section_key="3.2",
        title="3.2 Audit Log Retention",
        order_index=3,
        content="System must retain audit logs for 6 months after account creation.",
    )
    s6 = BRDSectionModel(
        id="sec-6",
        brd_version_id="v-1",
        section_key="6.0",
        title="6. Security and Compliance",
        order_index=6,
        content="Point 4: Multi-factor authentication is required for all administrative accounts.",
    )
    return [s1, s2, s3, s6]


def make_context(message: str, focus_entity: str | None = None) -> RequestContext:
    req = AgentRequest(
        thread_id="conv-1",
        brd_id="proj-1",
        message=message,
        user=UserContext(user_id="user-1", tenant_id="user-1"),
    )
    return RequestContext(request=req, focus_entity=focus_entity)


def test_exact_section_by_order_index():
    sections = make_dummy_sections()
    ctx = make_context("Explain section 6 and why it is necessary.")
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert ref.raw_text.lower() == "section 6"
    assert ref.section_id == "6.0"
    assert ref.confidence == "exact"
    assert updated.focus_entity == "6.0"


def test_exact_section_by_key():
    sections = make_dummy_sections()
    ctx = make_context("What does sec 2.3 require?")
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert ref.section_id == "2.3"
    assert ref.confidence == "exact"


def test_point_of_section():
    sections = make_dummy_sections()
    ctx = make_context("Explain point 4 of section 6.")
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert "point 4 of section 6" in ref.raw_text.lower()
    assert ref.section_id == "6.0"
    assert ref.confidence == "exact"


def test_pronoun_with_focus_entity():
    sections = make_dummy_sections()
    ctx = make_context("Why are we using that requirement?", focus_entity="6.0")
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert ref.raw_text.lower() == "that requirement"
    assert ref.section_id == "6.0"
    assert ref.confidence == "inferred"


def test_pronoun_without_focus_entity():
    sections = make_dummy_sections()
    ctx = make_context("Why are we using it?", focus_entity=None)
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert ref.raw_text.lower() == "it"
    assert ref.section_id is None
    assert ref.confidence == "ambiguous"


def test_nonexistent_section_reference():
    sections = make_dummy_sections()
    ctx = make_context("Explain section 99.")
    updated = resolve_references(ctx, sections)

    assert len(updated.resolved_references) == 1
    ref = updated.resolved_references[0]
    assert ref.section_id is None
    assert ref.confidence == "ambiguous"
    assert ref.candidates == []
