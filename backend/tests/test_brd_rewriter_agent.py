"""Tests for BRDRewriterAgent sub-agent and associated data structures.

Verifies:
1. Architectural Boundary: Tools list is strictly empty (tools=[]: no RAG, no DB, no search).
2. Narrowed Evidence Authority: Context accepts only findings and findings-referenced evidence.
3. Minimal Change Semantics: Output represents targeted replacements (DocumentEdit).
4. Synchronous and Asynchronous execution parity.
5. Robust JSON parsing and Markdown fence stripping.
6. Graceful degradation on malformed LLM responses.
7. Unapplied findings tracking for open questions and missing evidence.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd.final_validation.agent import (
    FinalValidationCategory,
    FinalValidationFinding,
    FinalValidationSeverity,
    FindingResolutionStatus,
)
from agents.brd.rewriter.agent import (
    BRDRewriterAgent,
    BRDRewriterContext,
    BRDRewriterResult,
    DocumentEdit,
    _build_rewriter_prompt,
    _parse_rewriter_response,
    get_rewriter_system_instruction_path,
    load_rewriter_system_instruction,
)


class MockLLM(BaseChatModel):
    """Deterministic mock chat model for testing."""

    response_content: str = "{}"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response_content))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-llm"


def test_system_instruction_loading():
    """Verify system instruction file exists and loads non-empty content."""
    path = get_rewriter_system_instruction_path()
    assert path.is_file()
    instruction = load_rewriter_system_instruction()
    assert len(instruction) > 50
    assert "BRD Rewriter Sub-Agent" in instruction


def test_rewriter_has_strictly_no_tools():
    """Verify rewriter sub-agent is instantiated with empty tools list (tools=[])."""
    model = MockLLM()
    rewriter = BRDRewriterAgent(model=model)
    assert rewriter.tools == []


def test_document_edit_serialization():
    """Verify DocumentEdit serialization and deserialization."""
    edit = DocumentEdit(
        finding_id="FV-001",
        target_location="Section 3.1",
        original_fragment="Old text fragment",
        corrected_fragment="New corrected text fragment",
        explanation="Fixed discrepancy",
    )
    d = edit.to_dict()
    assert d["finding_id"] == "FV-001"
    assert d["original_fragment"] == "Old text fragment"
    assert d["corrected_fragment"] == "New corrected text fragment"

    restored = DocumentEdit.from_dict(d)
    assert restored.finding_id == edit.finding_id
    assert restored.original_fragment == edit.original_fragment
    assert restored.corrected_fragment == edit.corrected_fragment
    assert restored.explanation == edit.explanation


def test_rewriter_context_serialization():
    """Verify BRDRewriterContext serialization and deserialization."""
    finding = FinalValidationFinding(
        category=FinalValidationCategory.CONSISTENCY,
        severity=FinalValidationSeverity.ERROR,
        issue="Terminology conflict",
        finding_id="FV-101",
        location="Section 2.1",
        problematic_content="Tenant vs Workspace",
        evidence="Glossary specifies Tenant",
        required_correction="Use Tenant consistently",
    )
    ctx = BRDRewriterContext(
        assembled_document="# BRD Document\n\nContent here",
        findings=[finding],
        finding_evidence=[{"finding_id": "FV-101", "evidence": "Glossary specifies Tenant"}],
        metadata={"project_id": "proj-99"},
    )
    d = ctx.to_dict()
    assert d["assembled_document"] == ctx.assembled_document
    assert len(d["findings"]) == 1
    assert len(d["finding_evidence"]) == 1
    assert d["metadata"]["project_id"] == "proj-99"

    restored = BRDRewriterContext.from_dict(d)
    assert restored.assembled_document == ctx.assembled_document
    assert len(restored.findings) == 1
    assert restored.findings[0].finding_id == "FV-101"
    assert restored.findings[0].evidence == "Glossary specifies Tenant"


def test_rewriter_prompt_construction_narrows_evidence():
    """Verify prompt only contains findings and referenced evidence."""
    finding = FinalValidationFinding(
        category=FinalValidationCategory.GROUNDING,
        severity=FinalValidationSeverity.ERROR,
        issue="Feature scope ungrounded",
        finding_id="FV-201",
        location="Section 4.2",
        problematic_content="Feature Z is included",
        evidence="PRD Section 1: Feature Z deferred to v2",
        required_correction="Remove Feature Z or mark Out of Scope",
    )
    ctx = BRDRewriterContext(
        assembled_document="# Complete Assembled BRD\n\n## 4. Scope\nFeature Z is included.",
        findings=[finding],
        finding_evidence=[{"finding_id": "FV-201", "evidence": "PRD Section 1: Feature Z deferred to v2"}],
    )
    prompt = _build_rewriter_prompt(ctx)
    assert "FV-201" in prompt
    assert "Feature Z deferred to v2" in prompt
    assert "# Complete Assembled BRD" in prompt


def test_parse_rewriter_response_valid_json():
    """Verify parsing valid JSON response into BRDRewriterResult."""
    data = {
        "summary": "Applied 1 targeted correction for FV-001.",
        "edits": [
            {
                "finding_id": "FV-001",
                "target_location": "Section 4.1",
                "original_fragment": "Latency target is 500ms",
                "corrected_fragment": "Latency target is 200ms",
                "explanation": "Updated SLA per SLA agreement evidence",
            }
        ],
        "unapplied_findings": [],
    }
    raw = json.dumps(data)
    res = _parse_rewriter_response(raw)
    assert res.summary == data["summary"]
    assert len(res.edits) == 1
    assert res.edits[0].finding_id == "FV-001"
    assert res.edits[0].original_fragment == "Latency target is 500ms"
    assert res.edits[0].corrected_fragment == "Latency target is 200ms"
    assert res.unapplied_findings == []


def test_parse_rewriter_response_with_markdown_fences():
    """Verify parsing JSON wrapped in markdown fences."""
    data = {
        "summary": "Targeted update.",
        "edits": [
            {
                "finding_id": "FV-002",
                "target_location": "Section 2",
                "original_fragment": "foo",
                "corrected_fragment": "bar",
                "explanation": "Replaced foo with bar",
            }
        ],
        "unapplied_findings": ["FV-003"],
    }
    raw = f"```json\n{json.dumps(data)}\n```"
    res = _parse_rewriter_response(raw)
    assert len(res.edits) == 1
    assert res.edits[0].finding_id == "FV-002"
    assert res.unapplied_findings == ["FV-003"]


def test_parse_rewriter_response_fallback_on_invalid():
    """Verify graceful handling when LLM returns non-JSON text."""
    raw = "I was unable to format as JSON, but please check the document."
    res = _parse_rewriter_response(raw)
    assert len(res.edits) == 0
    assert "Failed to parse structured JSON" in res.summary


def test_rewriter_sync_execution():
    """Verify synchronous rewrite execution."""
    expected_data = {
        "summary": "Corrected payment flow SLA",
        "edits": [
            {
                "finding_id": "FV-301",
                "target_location": "Section 5.3",
                "original_fragment": "Timeout is 60s",
                "corrected_fragment": "Timeout is 30s",
                "explanation": "Aligned timeout with gateway contract",
            }
        ],
        "unapplied_findings": [],
    }
    model = MockLLM(response_content=json.dumps(expected_data))
    rewriter = BRDRewriterAgent(model=model)

    ctx = BRDRewriterContext(
        assembled_document="# Payment BRD\n\n## 5. Non-Functional Requirements\nTimeout is 60s",
        findings=[
            FinalValidationFinding(
                finding_id="FV-301",
                category=FinalValidationCategory.REQUIREMENT_CONSISTENCY,
                severity=FinalValidationSeverity.ERROR,
                issue="Timeout inconsistency",
                location="Section 5.3",
                evidence="Gateway contract specifies 30s",
                required_correction="Change timeout to 30s",
            )
        ],
    )
    result = rewriter.rewrite(ctx)
    assert len(result.edits) == 1
    assert result.edits[0].finding_id == "FV-301"
    assert result.edits[0].corrected_fragment == "Timeout is 30s"


@pytest.mark.asyncio
async def test_rewriter_async_execution():
    """Verify asynchronous rewrite_async execution."""
    expected_data = {
        "summary": "Corrected data retention period",
        "edits": [
            {
                "finding_id": "FV-401",
                "target_location": "Section 8.1",
                "original_fragment": "Retain for 90 days",
                "corrected_fragment": "Retain for 365 days",
                "explanation": "Updated per GDPR and legal compliance evidence",
            }
        ],
        "unapplied_findings": [],
    }
    model = MockLLM(response_content=json.dumps(expected_data))
    rewriter = BRDRewriterAgent(model=model)

    ctx = BRDRewriterContext(
        assembled_document="# Security BRD\n\n## 8. Compliance\nRetain for 90 days",
        findings=[
            FinalValidationFinding(
                finding_id="FV-401",
                category=FinalValidationCategory.GROUNDING,
                severity=FinalValidationSeverity.ERROR,
                issue="Retention period mismatch",
                location="Section 8.1",
                evidence="Compliance doc specifies 365 days",
                required_correction="Change retention to 365 days",
            )
        ],
    )
    result = await rewriter.rewrite_async(ctx)
    assert len(result.edits) == 1
    assert result.edits[0].finding_id == "FV-401"
    assert result.edits[0].corrected_fragment == "Retain for 365 days"
