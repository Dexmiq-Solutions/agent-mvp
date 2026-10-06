"""BRD Final Validation Sub-Agent implementation.

Provides the dedicated, reusable document-level quality gate capability for the
BRD Agent architecture. Evaluates whether the complete assembled BRD is coherent,
consistent, well-grounded, and acceptable as a unified document.

Strict Architectural Boundaries:
- Final Validator evaluates the complete document; it NEVER rewrites, edits, or auto-repairs.
- Validator has NO tools (cannot call RAG, access databases, or invoke external tools).
- Validator does not interact with the user or own Agent State / BRD workflow.
- Validator uses categorical evaluation (VALID vs NEEDS_REWORK), not arbitrary numeric scores.
- The BRD Lead Agent remains the sole workflow owner and decides all subsequent workflow actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agents.brd.section_validation.agent import ValidationOutcome
from agents.brd.template import classify_requirement_item, is_metadata_or_role_item, is_tbd_value
from deepagents import create_deep_agent
from agents.brd.config import AgentConfig, create_agent_model
from observability import (
    execute_with_rate_limit_retry_async,
    extract_rate_limit_info,
    get_logger,
)

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"

# Alias for semantic clarity while preserving interoperability
FinalValidationOutcome = ValidationOutcome


def get_final_validation_system_instruction_path() -> Path:
    """Resolve the absolute path to the final validation system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_final_validation_system_instruction() -> str:
    """Load the authoritative final validation system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    path = get_final_validation_system_instruction_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required final validation system instruction file not found: {path}"
        )

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error(
            "Failed to read final validation system instruction from %s: %s", path, exc
        )
        raise

    if not content:
        raise ValueError(
            f"Final validation system instruction file is empty: {path}"
        )

    return content


class FinalValidationCategory(str, Enum):
    """Core evaluation dimensions for document-level BRD final validation."""

    TEMPLATE_COMPLIANCE = "Template Compliance"
    CROSS_SECTION_CONSISTENCY = "Cross-Section Consistency"
    REQUIREMENT_CONSISTENCY = "Requirement Consistency"
    TERMINOLOGY_CONSISTENCY = "Terminology Consistency"
    GROUNDING = "Grounding"
    COMPLETENESS = "Completeness"
    DUPLICATION = "Duplication"
    OVERALL_COHERENCE = "Overall Coherence"

    @classmethod
    def from_string(cls, val: str | "FinalValidationCategory") -> "FinalValidationCategory":
        """Convert string or enum to normalized FinalValidationCategory."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or FinalValidationCategory, got {type(val)}")

        norm = val.strip().lower().replace("_", " ").replace("-", " ")
        for cat in cls:
            if cat.value.lower() == norm or cat.name.lower() == norm.replace(" ", "_"):
                return cat

        # Fuzzy matching based on key terms
        if "cross" in norm or "cross section" in norm or "contradict" in norm:
            return cls.CROSS_SECTION_CONSISTENCY
        if "require" in norm or "conflict" in norm:
            return cls.REQUIREMENT_CONSISTENCY
        if "terminolog" in norm or "naming" in norm or "term" in norm or "vocab" in norm:
            return cls.TERMINOLOGY_CONSISTENCY
        if "ground" in norm or "fact" in norm or "evidence" in norm or "fabricat" in norm or "support" in norm:
            return cls.GROUNDING
        if "complete" in norm or "gap" in norm or "miss" in norm or "depend" in norm:
            return cls.COMPLETENESS
        if "duplicat" in norm or "redund" in norm or "overlap" in norm:
            return cls.DUPLICATION
        if "template" in norm or "structur" in norm or "format" in norm:
            return cls.TEMPLATE_COMPLIANCE
        if "coheren" in norm or "narrative" in norm or "flow" in norm or "align" in norm:
            return cls.OVERALL_COHERENCE

        return cls.CROSS_SECTION_CONSISTENCY


class FinalValidationSeverity(str, Enum):
    """Severity levels for final validation findings."""

    ERROR = "ERROR"
    WARNING = "WARNING"

    @classmethod
    def from_string(cls, val: str | "FinalValidationSeverity") -> "FinalValidationSeverity":
        """Convert string or enum to normalized FinalValidationSeverity."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or FinalValidationSeverity, got {type(val)}")

        norm = val.strip().upper()
        if "WARN" in norm:
            return cls.WARNING
        return cls.ERROR


@dataclass
class FinalValidationFinding:
    """Specific observation made during document-level validation.

    Attributes:
        category: The validation dimension (Cross-Section Consistency, Grounding, etc.).
        severity: ERROR (blocking) or WARNING (notable observation).
        issue: Concise description of the defect or inconsistency.
        explanation: Detailed rationale explaining why this violates document-level coherence.
        affected_sections: List of section names involved in this issue.
        evidence: Optional direct excerpt or reference from the document or project context.
        required_change: Actionable recommendation for what must be revised.
        metadata: Extensible metadata dictionary.
    """

    category: FinalValidationCategory | str
    severity: FinalValidationSeverity | str = FinalValidationSeverity.ERROR
    issue: str = ""
    explanation: str = ""
    affected_sections: list[str] = field(default_factory=list)
    evidence: Optional[str] = None
    required_change: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize finding to dictionary."""
        cat_val = self.category.value if isinstance(self.category, FinalValidationCategory) else str(self.category)
        sev_val = self.severity.value if isinstance(self.severity, FinalValidationSeverity) else str(self.severity)
        return {
            "category": cat_val,
            "severity": sev_val,
            "issue": self.issue,
            "explanation": self.explanation,
            "affected_sections": list(self.affected_sections),
            "evidence": self.evidence,
            "required_change": self.required_change,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalValidationFinding":
        """Deserialize dictionary into a FinalValidationFinding."""
        raw_cat = data.get("category", FinalValidationCategory.CROSS_SECTION_CONSISTENCY.value)
        try:
            cat = FinalValidationCategory.from_string(raw_cat)
        except ValueError:
            cat = raw_cat

        raw_sev = data.get("severity", FinalValidationSeverity.ERROR.value)
        try:
            sev = FinalValidationSeverity.from_string(raw_sev)
        except ValueError:
            sev = raw_sev

        return cls(
            category=cat,
            severity=sev,
            issue=str(data.get("issue", "")),
            explanation=str(data.get("explanation", "")),
            affected_sections=list(data.get("affected_sections", [])),
            evidence=data.get("evidence"),
            required_change=str(data.get("required_change", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FinalValidationContext:
    """Scoped context provided as input to the Final Validation Sub-Agent.

    Attributes:
        assembled_document: Complete Markdown text of the assembled BRD to evaluate.
        template_structure: Authoritative Markdown structure/skeleton from brd_template.md.
        available_project_information: Project facts, evidence, and context for grounding checks.
        section_names: Ordered list of top-level BRD sections.
        metadata: Extensible metadata dictionary (e.g. project_id, session notes).
    """

    assembled_document: str
    template_structure: str = ""
    available_project_information: Any = field(default_factory=list)
    section_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to JSON-compatible dictionary."""
        return {
            "assembled_document": self.assembled_document,
            "template_structure": self.template_structure,
            "available_project_information": self.available_project_information,
            "section_names": list(self.section_names),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalValidationContext":
        """Deserialize dictionary into a FinalValidationContext."""
        return cls(
            assembled_document=str(data.get("assembled_document", "")),
            template_structure=str(data.get("template_structure", "")),
            available_project_information=data.get("available_project_information", []),
            section_names=list(data.get("section_names", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FinalValidationResult:
    """Structured result returned by the Final Validation Sub-Agent.

    Attributes:
        outcome: FinalValidationOutcome (VALID or NEEDS_REWORK).
        summary: Concise summary of document-level validation conclusions.
        findings: List of FinalValidationFinding items detailing issues across the 8 dimensions.
        rework_feedback: Consolidated actionable instructions for rework when outcome is NEEDS_REWORK.
        metadata: Validation runtime metadata (e.g. duration, timestamp, document length).
    """

    outcome: FinalValidationOutcome | str
    summary: str = ""
    findings: list[FinalValidationFinding] = field(default_factory=list)
    rework_feedback: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True if the validation outcome is VALID."""
        outcome_val = self.outcome.value if isinstance(self.outcome, FinalValidationOutcome) else str(self.outcome)
        return outcome_val.strip().upper() == FinalValidationOutcome.VALID.value

    @property
    def needs_rework(self) -> bool:
        """Return True if the validation outcome is NEEDS_REWORK."""
        outcome_val = self.outcome.value if isinstance(self.outcome, FinalValidationOutcome) else str(self.outcome)
        return outcome_val.strip().upper() == FinalValidationOutcome.NEEDS_REWORK.value

    @property
    def affected_sections(self) -> list[str]:
        """Aggregate unique affected section names across all findings."""
        secs: list[str] = []
        for f in self.findings:
            for s in f.affected_sections:
                if s not in secs:
                    secs.append(s)
        return secs

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        outcome_val = self.outcome.value if isinstance(self.outcome, FinalValidationOutcome) else str(self.outcome)
        return {
            "outcome": outcome_val,
            "summary": self.summary,
            "findings": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.findings],
            "rework_feedback": self.rework_feedback,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalValidationResult":
        """Deserialize dictionary into a FinalValidationResult."""
        raw_outcome = data.get("outcome", FinalValidationOutcome.NEEDS_REWORK.value)
        try:
            outcome = FinalValidationOutcome.from_string(raw_outcome)
        except ValueError:
            outcome = raw_outcome

        raw_findings = data.get("findings", [])
        findings = [
            FinalValidationFinding.from_dict(f) if isinstance(f, dict) else f
            for f in raw_findings
        ]

        return cls(
            outcome=outcome,
            summary=str(data.get("summary", "")),
            findings=findings,
            rework_feedback=data.get("rework_feedback"),
            metadata=dict(data.get("metadata", {})),
        )


def _format_available_information(info: Any) -> str:
    """Format various types of available project information/evidence into clear text."""
    if not info:
        return "*(No specific project information provided)*"

    if isinstance(info, str):
        return info.strip()

    if isinstance(info, (list, tuple)):
        formatted_items = []
        for item in info:
            if isinstance(item, str):
                formatted_items.append(f"- {item}")
            elif isinstance(item, dict):
                src = item.get("source", "project_evidence")
                c = item.get("content", str(item))
                formatted_items.append(f"### Source: {src}\n{c}")
            elif hasattr(item, "content"):
                src = getattr(item, "source", "project_evidence")
                formatted_items.append(f"### Source: {src}\n{item.content}")
            else:
                formatted_items.append(f"- {item}")
        return "\n\n".join(formatted_items)

    return str(info)


def _build_final_validation_prompt(context: FinalValidationContext) -> str:
    """Format prompt with focused context for the Final Validation Sub-Agent."""
    parts: list[str] = [
        "## Validation Task\n"
        "Perform document-level final validation on the complete assembled Business Requirements Document (BRD).\n"
        "Evaluate the BRD as a whole for consistency, grounding, completeness, and coherence.",
    ]

    if context.section_names:
        sec_list = "\n".join([f"{idx + 1}. {s}" for idx, s in enumerate(context.section_names)])
        parts.append(f"## Authoritative Section Order\n{sec_list}")

    if context.template_structure:
        parts.append(
            f"## Authoritative Template Structure & Format\n```markdown\n{context.template_structure}\n```"
        )

    info_str = _format_available_information(context.available_project_information)
    parts.append(f"## Available Project Information & Evidence (Grounding Baseline)\n{info_str}")

    parts.append(
        f"## Complete Assembled BRD Document to Validate\n```markdown\n{context.assembled_document.strip()}\n```"
    )

    parts.append(
        "Evaluate the complete BRD strictly across the 8 document-level validation dimensions:\n"
        "1. Template Compliance: Does the assembled document contain all required sections in template order?\n"
        "2. Cross-Section Consistency: Are there factual or scope contradictions across different sections?\n"
        "3. Requirement Consistency: Do any requirements conflict with each other across sections?\n"
        "4. Terminology Consistency: Are key terms used inconsistently or ambiguously across sections?\n"
        "5. Grounding / Fact Integrity: Are important business claims supported by the project evidence?\n"
        "6. Completeness: Are there critical document-level gaps, missing workflows, or unresolved dependencies?\n"
        "7. Duplication: Are there redundant or overlapping requirements?\n"
        "8. Overall Coherence: Does the document tell a unified, logical business story from context to criteria?\n\n"
        "Special Validation Policy for Administrative Metadata & Mechanics:\n"
        "- Administrative metadata fields (Prepared By, Reviewed By, Approved By, Tech Lead, Stakeholder, Approver) "
        "legitimately containing 'TBD' or 'TBD (Suggested: <Name>)' are VALID placeholders and MUST NOT be flagged as errors.\n"
        "- Document mechanics (version, question IDs, current date) are deterministic and do not require RAG evidence.\n"
        "- Substantive business and technical requirements must still be strictly grounded.\n\n"
        "Determine the categorical outcome: VALID or NEEDS_REWORK (do NOT output numeric scores).\n"
        "For each issue identified, output a finding with category, severity (ERROR or WARNING), issue, "
        "explanation, affected_sections, evidence, and required_change.\n"
        "If outcome is NEEDS_REWORK, provide actionable rework_feedback.\n"
        "If outcome is VALID, rework_feedback must be null.\n"
        "Return ONLY a valid JSON object matching the output schema."
    )

    return "\n\n".join(parts)


def _parse_final_validation_response(
    response_text: str,
    context: FinalValidationContext,
    duration: float = 0.0,
) -> FinalValidationResult:
    """Parse LLM output text into structured FinalValidationResult, with robust JSON extraction and fallback."""
    # Attempt to locate JSON object
    json_match = re.search(r"\{\s*\"outcome\".*\}\s*", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if json_match:
        raw_json = json_match.group(0)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "outcome" in parsed:
                outcome_raw = parsed.get("outcome", FinalValidationOutcome.NEEDS_REWORK.value)
                try:
                    outcome = FinalValidationOutcome.from_string(outcome_raw)
                except ValueError:
                    outcome = FinalValidationOutcome.NEEDS_REWORK

                summary = str(parsed.get("summary", "")).strip()
                raw_findings = parsed.get("findings", [])
                findings: list[FinalValidationFinding] = []
                if isinstance(raw_findings, list):
                    for item in raw_findings:
                        if isinstance(item, dict):
                            findings.append(FinalValidationFinding.from_dict(item))
                        elif isinstance(item, str):
                            findings.append(
                                FinalValidationFinding(
                                    category=FinalValidationCategory.CROSS_SECTION_CONSISTENCY,
                                    issue=item,
                                    explanation=item,
                                    required_change=item,
                                )
                            )

                # Filter out benign administrative metadata TBD findings
                substantive_findings: list[FinalValidationFinding] = []
                for f in findings:
                    issue_text = f"{f.issue or ''} {f.explanation or ''} {f.required_change or ''}"
                    if is_metadata_or_role_item(issue_text) and any(
                        kw in issue_text.lower() for kw in ["tbd", "unknown", "suggested", "placeholder", "missing tech lead", "prepared by", "reviewed by", "approved by"]
                    ):
                        logger.info("Filtered benign administrative metadata finding from final validation: %s", f.issue)
                        continue
                    substantive_findings.append(f)
                findings = substantive_findings

                # If outcome was NEEDS_REWORK purely due to filtered metadata findings, override to VALID
                has_error_findings = any(
                    (getattr(f, "severity", None) and str(f.severity).upper() in ("ERROR", "SEVERITY.ERROR"))
                    for f in findings
                )
                if outcome == FinalValidationOutcome.NEEDS_REWORK and not has_error_findings and not findings:
                    outcome = FinalValidationOutcome.VALID
                    summary = "Complete BRD evaluated as VALID after administrative metadata normalization."

                rework_feedback = parsed.get("rework_feedback")
                if isinstance(rework_feedback, str):
                    rework_feedback = rework_feedback.strip() or None

                # Synthesize actionable rework_feedback if outcome is NEEDS_REWORK and feedback was omitted
                if outcome == FinalValidationOutcome.NEEDS_REWORK and not rework_feedback and findings:
                    feedback_lines = []
                    for f in findings:
                        cat_str = f.category if isinstance(f.category, str) else f.category.value
                        secs_str = f" [Sections: {', '.join(f.affected_sections)}]" if f.affected_sections else ""
                        feedback_lines.append(f"- [{cat_str}]{secs_str} {f.issue}: {f.required_change}")
                    rework_feedback = "\n".join(feedback_lines)

                return FinalValidationResult(
                    outcome=outcome,
                    summary=summary or f"Complete BRD evaluated as {outcome.value}.",
                    findings=findings,
                    rework_feedback=rework_feedback if outcome == FinalValidationOutcome.NEEDS_REWORK else None,
                    metadata={
                        "duration_seconds": duration,
                        "document_length": len(context.assembled_document),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to parse extracted JSON in final validation response: %s", exc)

    # Fallback if direct text was returned instead of JSON
    cleaned_text = response_text.strip()
    upper_text = cleaned_text.upper()

    is_valid_detected = (
        "VALID" in upper_text
        and "NEEDS_REWORK" not in upper_text
        and "INVALID" not in upper_text
        and "REWORK" not in upper_text
        and "FAIL" not in upper_text
    )

    if is_valid_detected:
        return FinalValidationResult(
            outcome=FinalValidationOutcome.VALID,
            summary="Complete BRD validated successfully across all document-level dimensions.",
            findings=[],
            rework_feedback=None,
            metadata={
                "duration_seconds": duration,
                "document_length": len(context.assembled_document),
                "fallback_parsing": True,
            },
        )

    # Safety invariant: If not definitively valid, treat conservatively as NEEDS_REWORK
    fallback_finding = FinalValidationFinding(
        category=FinalValidationCategory.OVERALL_COHERENCE,
        severity=FinalValidationSeverity.ERROR,
        issue="Document-level quality concerns identified in model output",
        explanation=cleaned_text[:300] if len(cleaned_text) > 300 else cleaned_text,
        required_change="Address the validation observations and revise affected sections.",
    )

    return FinalValidationResult(
        outcome=FinalValidationOutcome.NEEDS_REWORK,
        summary="Complete BRD requires rework across document-level dimensions.",
        findings=[fallback_finding],
        rework_feedback=cleaned_text[:500] if cleaned_text else "Address cross-section consistency and requirements conflicts.",
        metadata={
            "duration_seconds": duration,
            "document_length": len(context.assembled_document),
            "fallback_parsing": True,
        },
    )


class BRDFinalValidationAgent:
    """Dedicated, reusable Final BRD Validation Sub-Agent capability.

    Evaluates the complete assembled BRD across document-level dimensions (cross-section
    consistency, requirements conflicts, grounding, completeness, coherence) on behalf of
    the BRD Lead Agent.

    Architectural Invariants:
    1. Operates strictly as a quality evaluator; never generates, rewrites, or auto-repairs.
    2. Equipped with NO tools (tools=[]); cannot call RAG, query databases, or fetch data.
    3. Does not interact with user, delegate tasks, or orchestrate the workflow.
    4. Evaluates categorical outcomes (VALID vs NEEDS_REWORK) with concrete findings.
    5. Returns actionable rework feedback detailing affected sections when validation fails.
    """

    agent_name: str = "BRDFinalValidationAgent"

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        config: Optional[AgentConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        """Initialize the Final Validation Sub-Agent.

        Args:
            model: Optional pre-configured BaseChatModel instance.
            config: Optional AgentConfig instance. Used if model is omitted.
            system_instruction: Optional system instruction override. If omitted,
                loaded from system_instruction.md.
        """
        self._system_instruction = system_instruction or load_final_validation_system_instruction()
        if model:
            self._model = model
        else:
            import dataclasses
            cfg = config or AgentConfig.from_settings()
            # Validation responses are small JSONs; strictly bound max_tokens to prevent TPM exhaustion
            if cfg.max_tokens is None or cfg.max_tokens > 800:
                cfg = dataclasses.replace(cfg, max_tokens=800)
            self._model = create_agent_model(cfg)

        # Strict boundary: Sub-Agent has NO tools
        self._tools: list[Any] = []

        # DeepAgents graph without tools
        self._graph = create_deep_agent(
            model=self._model,
            tools=self._tools,
            system_prompt=self._system_instruction,
        )

    @property
    def system_instruction(self) -> str:
        """Return the authoritative final validation system instruction."""
        return self._system_instruction

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools (always empty for Final Validation Sub-Agent)."""
        return self._tools

    @property
    def graph(self) -> Any:
        """Return the underlying DeepAgents execution graph."""
        return self._graph

    def validate(
        self,
        context: FinalValidationContext | dict[str, Any],
    ) -> FinalValidationResult:
        """Validate the complete assembled BRD synchronously.

        Args:
            context: Scoped FinalValidationContext or dictionary.

        Returns:
            FinalValidationResult: Structured result with outcome, findings, and rework feedback.
        """
        ctx = FinalValidationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "BRD final validation started (document_length: %d, sections_count: %d)",
            len(ctx.assembled_document),
            len(ctx.section_names),
        )

        try:
            prompt_input = _build_final_validation_prompt(ctx)
            result = self._graph.invoke({"messages": [HumanMessage(content=prompt_input)]})
            duration = time.perf_counter() - start_time

            messages = result.get("messages", [])
            output_text = ""
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", "")
                if isinstance(content, str):
                    output_text = content
                elif isinstance(content, list):
                    text_parts = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    output_text = "".join(text_parts)
                else:
                    output_text = str(content)

            val_res = _parse_final_validation_response(output_text, ctx, duration=duration)
            logger.info(
                "BRD final validation completed (outcome: %s, duration: %.2fs, findings: %d)",
                val_res.outcome,
                duration,
                len(val_res.findings),
            )
            return val_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "BRD Final Validation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            # Invariant: Validator execution failure must NEVER produce VALID
            return FinalValidationResult(
                outcome=FinalValidationOutcome.NEEDS_REWORK,
                summary=f"Final BRD validation failed due to runtime error: {err_msg}",
                findings=[
                    FinalValidationFinding(
                        category=FinalValidationCategory.OVERALL_COHERENCE,
                        severity=FinalValidationSeverity.ERROR,
                        issue="Validation execution error",
                        explanation=err_msg,
                        required_change="Resolve runtime error and re-run final validation",
                    )
                ],
                rework_feedback=f"Final validation runtime error occurred: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                },
            )

    async def validate_async(
        self,
        context: FinalValidationContext | dict[str, Any],
    ) -> FinalValidationResult:
        """Validate the complete assembled BRD asynchronously.

        Args:
            context: Scoped FinalValidationContext or dictionary.

        Returns:
            FinalValidationResult: Structured result with outcome, findings, and rework feedback.
        """
        ctx = FinalValidationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "Async BRD final validation started (document_length: %d, sections_count: %d)",
            len(ctx.assembled_document),
            len(ctx.section_names),
        )

        try:
            prompt_input = _build_final_validation_prompt(ctx)
            result = await execute_with_rate_limit_retry_async(
                lambda: self._graph.ainvoke({"messages": [HumanMessage(content=prompt_input)]}),
                operation_name="validate_final_brd",
            )
            duration = time.perf_counter() - start_time

            messages = result.get("messages", [])
            output_text = ""
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", "")
                if isinstance(content, str):
                    output_text = content
                elif isinstance(content, list):
                    text_parts = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    output_text = "".join(text_parts)
                else:
                    output_text = str(content)

            val_res = _parse_final_validation_response(output_text, ctx, duration=duration)
            logger.info(
                "Async BRD final validation completed (outcome: %s, duration: %.2fs, findings: %d)",
                val_res.outcome,
                duration,
                len(val_res.findings),
            )
            return val_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            rate_info = extract_rate_limit_info(exc)
            logger.error(
                "Async BRD Final Validation Sub-Agent execution failed (duration: %.2fs, is_rate_limit: %s): %s",
                duration,
                rate_info.is_rate_limit,
                err_msg,
                exc_info=True,
            )
            return FinalValidationResult(
                outcome=FinalValidationOutcome.NEEDS_REWORK,
                summary=f"Final BRD validation failed due to runtime error: {err_msg}",
                findings=[
                    FinalValidationFinding(
                        category=FinalValidationCategory.OVERALL_COHERENCE,
                        severity=FinalValidationSeverity.ERROR,
                        issue="Validation execution error",
                        explanation=err_msg,
                        required_change="Resolve runtime error and re-run final validation",
                    )
                ],
                rework_feedback=f"Final validation runtime error occurred: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                    "rate_limit_info": rate_info.to_dict() if rate_info.is_rate_limit else None,
                },
            )
