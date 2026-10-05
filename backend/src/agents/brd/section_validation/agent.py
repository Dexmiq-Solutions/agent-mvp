"""BRD Section Validation Sub-Agent implementation.

Provides the dedicated, reusable quality and compliance validation capability
for the BRD Agent architecture. Evaluates whether a generated or updated BRD section
satisfies its authoritative template structure, addresses section requirements, and accurately
reflects the provided evidence without inventing facts.

Strict Architectural Boundaries:
- Section Validator evaluates and produces findings; it NEVER generates, rewrites, or auto-repairs.
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

from deepagents import create_deep_agent
from agents.brd.config import AgentConfig, create_agent_model
from agents.brd.template import (
    extract_section_requirements,
    extract_section_template,
    is_benign_administrative_metadata_finding,
    is_documentation_quality_finding,
)
from observability.logging import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"


def get_validation_system_instruction_path() -> Path:
    """Resolve the absolute path to the section validation system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_validation_system_instruction() -> str:
    """Load the authoritative section validation system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    path = get_validation_system_instruction_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required section validation system instruction file not found: {path}"
        )

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error(
            "Failed to read section validation system instruction from %s: %s", path, exc
        )
        raise

    if not content:
        raise ValueError(
            f"Section validation system instruction file is empty: {path}"
        )

    return content


class ValidationOutcome(str, Enum):
    """Categorical outcome of a section validation."""

    VALID = "VALID"
    NEEDS_REWORK = "NEEDS_REWORK"

    @classmethod
    def from_string(cls, val: str | "ValidationOutcome") -> "ValidationOutcome":
        """Convert a string or enum to normalized ValidationOutcome."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or ValidationOutcome, got {type(val)}")

        normalized = val.strip().upper()
        if "NEEDS_REWORK" in normalized or "REWORK" in normalized or "FAIL" in normalized or "INVALID" in normalized:
            return cls.NEEDS_REWORK
        if "VALID" in normalized or "PASS" in normalized:
            return cls.VALID

        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(
            f"Invalid validation outcome '{val}'. Expected VALID or NEEDS_REWORK."
        )


class ValidationCategory(str, Enum):
    """Core evaluation dimensions for BRD section validation."""

    TEMPLATE_COMPLIANCE = "Template Compliance"
    REQUIREMENT_COVERAGE = "Requirement Coverage"
    COMPLETENESS = "Completeness"
    SPECIFICITY = "Specificity"
    GROUNDING = "Grounding"
    CONSISTENCY = "Consistency"
    RELEVANCE = "Relevance"

    @classmethod
    def from_string(cls, val: str | "ValidationCategory") -> "ValidationCategory":
        """Convert string or enum to normalized ValidationCategory."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or ValidationCategory, got {type(val)}")

        norm = val.strip().lower().replace("_", " ").replace("-", " ")
        for cat in cls:
            if cat.value.lower() == norm or cat.name.lower() == norm.replace(" ", "_"):
                return cat
        # Fuzzy fallback based on key terms
        if "template" in norm or "format" in norm or "structure" in norm:
            return cls.TEMPLATE_COMPLIANCE
        if "requirement" in norm or "coverage" in norm:
            return cls.REQUIREMENT_COVERAGE
        if "complete" in norm or "missing" in norm:
            return cls.COMPLETENESS
        if "specific" in norm or "concrete" in norm or "vague" in norm:
            return cls.SPECIFICITY
        if "ground" in norm or "fact" in norm or "evidence" in norm or "fabricat" in norm:
            return cls.GROUNDING
        if "consist" in norm or "contradict" in norm:
            return cls.CONSISTENCY
        if "relevan" in norm or "scope" in norm:
            return cls.RELEVANCE

        return cls.REQUIREMENT_COVERAGE


@dataclass
class ValidationFinding:
    """Specific observation made during section validation with actionable required changes.

    Attributes:
        category: The validation dimension (Template Compliance, Grounding, etc.).
        issue: Concise description of the defect or non-compliance.
        explanation: Detailed rationale of why this violates template, requirements, or evidence.
        required_change: Concrete, actionable instruction for the Section Generator during rework.
        metadata: Extensible metadata dictionary.
    """

    category: ValidationCategory | str
    issue: str
    explanation: str
    required_change: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize finding to dictionary."""
        cat_val = self.category.value if isinstance(self.category, ValidationCategory) else str(self.category)
        return {
            "category": cat_val,
            "issue": self.issue,
            "explanation": self.explanation,
            "required_change": self.required_change,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationFinding":
        """Deserialize dictionary into a ValidationFinding."""
        raw_cat = data.get("category", ValidationCategory.REQUIREMENT_COVERAGE.value)
        try:
            cat = ValidationCategory.from_string(raw_cat)
        except ValueError:
            cat = raw_cat

        return cls(
            category=cat,
            issue=str(data.get("issue", "")),
            explanation=str(data.get("explanation", "")),
            required_change=str(data.get("required_change", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SectionValidationContext:
    """Scoped context provided as input to the Section Validation Sub-Agent.

    Attributes:
        section_name: Target section name (e.g. "5. Stakeholders & Personas").
        section_content: Complete Markdown content of the generated or updated section to evaluate.
        section_requirements: Required items or criteria the section must satisfy.
        template_structure: Authoritative Markdown structure/skeleton from brd_template.md.
        available_information: Evidence, facts, and working context used when generating the section.
        prior_rework_feedback: Optional prior feedback if this is a re-validation cycle.
        metadata: Extensible metadata dictionary (e.g. project_id, section_id).
    """

    section_name: str
    section_content: str
    section_requirements: list[str] = field(default_factory=list)
    template_structure: str = ""
    available_information: Any = field(default_factory=list)
    prior_rework_feedback: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to JSON-compatible dictionary."""
        return {
            "section_name": self.section_name,
            "section_content": self.section_content,
            "section_requirements": list(self.section_requirements),
            "template_structure": self.template_structure,
            "available_information": self.available_information,
            "prior_rework_feedback": self.prior_rework_feedback,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionValidationContext":
        """Deserialize dictionary into a SectionValidationContext."""
        return cls(
            section_name=str(data.get("section_name", "")),
            section_content=str(data.get("section_content", "")),
            section_requirements=list(data.get("section_requirements", [])),
            template_structure=str(data.get("template_structure", "")),
            available_information=data.get("available_information", []),
            prior_rework_feedback=data.get("prior_rework_feedback"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ValidationResult:
    """Structured result returned by the Section Validation Sub-Agent.

    Attributes:
        outcome: ValidationOutcome (VALID or NEEDS_REWORK).
        summary: Concise summary of validation conclusions.
        findings: List of ValidationFinding items detailing issues across the 7 dimensions.
        rework_feedback: Consolidated actionable instructions for section rework when outcome is NEEDS_REWORK.
        section_name: Optional target section identifier / heading.
        metadata: Validation runtime metadata (e.g. duration, timestamp).
    """

    outcome: ValidationOutcome | str
    summary: str = ""
    findings: list[ValidationFinding] = field(default_factory=list)
    rework_feedback: Optional[str] = None
    section_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True if the validation outcome is VALID."""
        outcome_val = self.outcome.value if isinstance(self.outcome, ValidationOutcome) else str(self.outcome)
        return outcome_val.strip().upper() == ValidationOutcome.VALID.value

    @property
    def needs_rework(self) -> bool:
        """Return True if the validation outcome is NEEDS_REWORK."""
        outcome_val = self.outcome.value if isinstance(self.outcome, ValidationOutcome) else str(self.outcome)
        return outcome_val.strip().upper() == ValidationOutcome.NEEDS_REWORK.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        outcome_val = self.outcome.value if isinstance(self.outcome, ValidationOutcome) else str(self.outcome)
        return {
            "outcome": outcome_val,
            "summary": self.summary,
            "findings": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.findings],
            "rework_feedback": self.rework_feedback,
            "section_name": self.section_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Deserialize dictionary into a ValidationResult."""
        raw_outcome = data.get("outcome", ValidationOutcome.NEEDS_REWORK.value)
        try:
            outcome = ValidationOutcome.from_string(raw_outcome)
        except ValueError:
            outcome = raw_outcome

        raw_findings = data.get("findings", [])
        findings = [
            ValidationFinding.from_dict(f) if isinstance(f, dict) else f
            for f in raw_findings
        ]

        return cls(
            outcome=outcome,
            summary=str(data.get("summary", "")),
            findings=findings,
            rework_feedback=data.get("rework_feedback"),
            section_name=data.get("section_name"),
            metadata=dict(data.get("metadata", {})),
        )


def _format_available_information(info: Any) -> str:
    """Format various types of available information/evidence into clear text."""
    if not info:
        return "*(No specific information provided)*"

    if isinstance(info, str):
        return info.strip()

    if isinstance(info, (list, tuple)):
        formatted_items = []
        for item in info:
            if isinstance(item, str):
                formatted_items.append(f"- {item}")
            elif isinstance(item, dict):
                src = item.get("source", "evidence")
                c = item.get("content", str(item))
                formatted_items.append(f"### Source: {src}\n{c}")
            elif hasattr(item, "content"):
                src = getattr(item, "source", "evidence")
                formatted_items.append(f"### Source: {src}\n{item.content}")
            else:
                formatted_items.append(f"- {item}")
        return "\n\n".join(formatted_items)

    return str(info)


def _build_validation_prompt(context: SectionValidationContext) -> str:
    """Format prompt with focused context for the Section Validation Sub-Agent."""
    parts: list[str] = []

    proj_name = context.metadata.get("project_name") if context.metadata else None
    proj_desc = context.metadata.get("project_description") if context.metadata else None
    if proj_name or proj_desc:
        proj_lines = []
        if proj_name:
            proj_lines.append(f"Project Name: {proj_name}")
        if proj_desc:
            proj_lines.append(f"Project Description: {proj_desc}")
        parts.append("## Project Context\n" + "\n".join(proj_lines))

    parts.extend([
        f"## Target Section to Validate\n{context.section_name}",
    ])

    if context.template_structure:
        parts.append(
            f"## Authoritative Template Structure & Format\n```markdown\n{context.template_structure}\n```"
        )

    if context.section_requirements:
        req_lines = [f"- {r}" for r in context.section_requirements]
        parts.append("## Section Requirements & Criteria\n" + "\n".join(req_lines))

    info_str = _format_available_information(context.available_information)
    parts.append(f"## Available Information & Evidence (Grounding Source of Truth)\n{info_str}")

    if context.prior_rework_feedback and context.prior_rework_feedback.strip():
        parts.append(
            f"## Prior Rework Feedback Provided\n{context.prior_rework_feedback.strip()}"
        )

    parts.append(
        f"## Generated Section Content to Validate\n```markdown\n{context.section_content.strip()}\n```"
    )

    parts.append(
        "Evaluate the section strictly across the 7 validation dimensions:\n"
        "1. Template Compliance: Does it follow the required Markdown headings, subsections, and tables?\n"
        "2. Requirement Coverage: Does it address all required section criteria?\n"
        "3. Completeness: Are essential elements omitted?\n"
        "4. Specificity: Is the content sufficiently concrete and actionable for a BRD?\n"
        "5. Grounding / Fact Integrity: Are all claims supported by the provided evidence? Do not accept fabricated facts.\n"
        "6. Consistency: Are there internal contradictions or conflicts with the evidence?\n"
        "7. Relevance: Does the content stay strictly focused on this section?\n\n"
        "Validation Policy for Metadata & TBD:\n"
        "- Administrative & Personnel Metadata: Fields such as Prepared By, Reviewed By, Approved By, Tech Lead, "
        "Stakeholder, Approvers, Client Tier, Lifecycle Phase, etc., legitimately default to 'TBD' or 'TBD (Suggested: <Name>)' "
        "when no confirmed evidence is supplied. Marking unknown metadata as TBD is VALID and must NOT be flagged as missing or ungrounded.\n"
        "- Document Mechanics: Version numbers (e.g. 1.0), question IDs (e.g. Q-BRD-0001), module IDs, and dates "
        "are deterministic mechanics and do not require RAG evidence.\n"
        "- Substantive Requirements: Business rules, integrations, workflows, and functional requirements MUST be grounded. "
        "Unsupported business claims must be flagged.\n"
        "- Representation of Unknowns: When information is genuinely unavailable in project evidence, "
        "transparent, honest statements acknowledging that specific details were not provided "
        "(e.g., 'Conceptual workflows were not provided in the available project information', "
        "'Detailed AI functionality was not defined in the available information', 'TBD') are VALID and grounded. "
        "Do NOT flag honest representations of non-provided information as omissions, completeness failures, or ungrounded claims.\n"
        "- Documentation-Quality Observations: Missing conceptual workflows, persona priorities/frustrations, "
        "persona-to-module links, or detailed AI behaviors must NOT cause section failure when the evidence simply does not contain them.\n"
        "- Grounding Enforcement: Unsupported or fabricated project facts presented as confirmed truth MUST be flagged under Grounding.\n\n"
        "Determine the categorical outcome: VALID or NEEDS_REWORK.\n"
        "For each issue found, provide a concrete finding with category, issue, explanation, and required_change.\n"
        "If outcome is NEEDS_REWORK, provide actionable rework_feedback summarizing what the generator must change.\n"
        "If outcome is VALID, rework_feedback must be null.\n"
        "Return ONLY a valid JSON object matching the output schema."
    )

    return "\n\n".join(parts)


def _parse_validation_response(
    response_text: str,
    context: SectionValidationContext,
    duration: float = 0.0,
) -> ValidationResult:
    """Parse LLM output text into structured ValidationResult, with robust JSON extraction and fallback."""
    from agents.brd.template import is_administrative_section, is_metadata_or_role_item

    # Attempt to locate JSON object
    json_match = re.search(r"\{\s*\"outcome\".*\}\s*", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if json_match:
        raw_json = json_match.group(0)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "outcome" in parsed:
                outcome_raw = parsed.get("outcome", ValidationOutcome.NEEDS_REWORK.value)
                try:
                    outcome = ValidationOutcome.from_string(outcome_raw)
                except ValueError:
                    outcome = ValidationOutcome.NEEDS_REWORK

                summary = str(parsed.get("summary", "")).strip()
                raw_findings = parsed.get("findings", [])
                findings: list[ValidationFinding] = []
                if isinstance(raw_findings, list):
                    for item in raw_findings:
                        if isinstance(item, dict):
                            findings.append(ValidationFinding.from_dict(item))
                        elif isinstance(item, str):
                            findings.append(
                                ValidationFinding(
                                    category=ValidationCategory.REQUIREMENT_COVERAGE,
                                    issue=item,
                                    explanation=item,
                                    required_change=item,
                                )
                            )

                rework_feedback = parsed.get("rework_feedback")
                if isinstance(rework_feedback, str):
                    rework_feedback = rework_feedback.strip() or None

                # Filter out findings that falsely penalize valid administrative TBD metadata or documentation-quality observations
                substantive_findings = [
                    f for f in findings
                    if not is_benign_administrative_metadata_finding(f, context.section_name)
                    and not is_documentation_quality_finding(f, context.section_name)
                ]

                if outcome == ValidationOutcome.NEEDS_REWORK and not substantive_findings:
                    # All findings were benign administrative metadata TBDs; section is valid
                    outcome = ValidationOutcome.VALID
                    rework_feedback = None
                    findings = []
                else:
                    findings = substantive_findings

                # Ensure actionable rework_feedback is synthesized if NEEDS_REWORK and feedback was omitted
                if outcome == ValidationOutcome.NEEDS_REWORK and not rework_feedback and findings:
                    feedback_lines = [
                        f"- [{f.category if isinstance(f.category, str) else f.category.value}] {f.issue}: {f.required_change}"
                        for f in findings
                    ]
                    rework_feedback = "\n".join(feedback_lines)
                elif outcome == ValidationOutcome.VALID:
                    rework_feedback = None

                return ValidationResult(
                    outcome=outcome,
                    summary=summary or f"Section {context.section_name} evaluated as {outcome.value}.",
                    findings=findings,
                    rework_feedback=rework_feedback,
                    section_name=context.section_name,
                    metadata={
                        "duration_seconds": duration,
                    },
                )
        except Exception as exc:
            logger.warning("Failed to parse extracted JSON in validation response: %s", exc)

    # Heuristic fallback if direct text was returned instead of JSON
    cleaned_text = response_text.strip()
    upper_text = cleaned_text.upper()

    is_valid_detected = "VALID" in upper_text and "NEEDS_REWORK" not in upper_text and "INVALID" not in upper_text and "REWORK" not in upper_text

    if is_valid_detected:
        return ValidationResult(
            outcome=ValidationOutcome.VALID,
            summary=f"Section {context.section_name} validated successfully.",
            findings=[],
            rework_feedback=None,
            section_name=context.section_name,
            metadata={
                "duration_seconds": duration,
                "fallback_parsing": True,
            },
        )

    # If not definitively valid, treat conservatively as NEEDS_REWORK per safety rules
    fallback_finding = ValidationFinding(
        category=ValidationCategory.REQUIREMENT_COVERAGE,
        issue="Validation issues identified in model output",
        explanation=cleaned_text[:300] if len(cleaned_text) > 300 else cleaned_text,
        required_change="Address the validation observations and revise section content.",
    )

    return ValidationResult(
        outcome=ValidationOutcome.NEEDS_REWORK,
        summary=f"Section {context.section_name} requires rework.",
        findings=[fallback_finding],
        rework_feedback=cleaned_text[:500] if cleaned_text else "Address section requirements and template criteria.",
        section_name=context.section_name,
        metadata={
            "duration_seconds": duration,
            "fallback_parsing": True,
        },
    )


class BRDSectionValidationAgent:
    """Dedicated, reusable Section Validation Sub-Agent capability.

    Evaluates a single generated or updated BRD section against authoritative template
    requirements, structure, and supplied evidence on behalf of the BRD Lead Agent.

    Architectural Invariants:
    1. Operates strictly as a quality evaluator; never generates, rewrites, or auto-repairs.
    2. Equipped with NO tools (tools=[]); cannot call RAG, query databases, or fetch data.
    3. Does not interact with user, delegate tasks, or orchestrate the workflow.
    4. Evaluates categorical outcomes (VALID vs NEEDS_REWORK) with concrete findings.
    5. Returns actionable rework feedback when validation fails.
    """

    agent_name: str = "BRDSectionValidationAgent"

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        config: Optional[AgentConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        """Initialize the Section Validation Sub-Agent.

        Args:
            model: Optional pre-configured BaseChatModel instance.
            config: Optional AgentConfig instance. Used if model is omitted.
            system_instruction: Optional system instruction override. If omitted,
                loaded from system_instruction.md.
        """
        self._system_instruction = system_instruction or load_validation_system_instruction()
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
        """Return the authoritative section validation system instruction."""
        return self._system_instruction

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools (always empty for Section Validation Sub-Agent)."""
        return self._tools

    @property
    def graph(self) -> Any:
        """Return the underlying DeepAgents execution graph."""
        return self._graph

    def validate(
        self,
        context: SectionValidationContext | dict[str, Any],
    ) -> ValidationResult:
        """Validate a generated or updated BRD section synchronously.

        Args:
            context: Scoped SectionValidationContext or dictionary.

        Returns:
            ValidationResult: Structured result with outcome, findings, and rework feedback.
        """
        ctx = SectionValidationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "Section validation started (section: %s)",
            ctx.section_name,
        )

        try:
            prompt_input = _build_validation_prompt(ctx)
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

            val_res = _parse_validation_response(output_text, ctx, duration=duration)
            logger.info(
                "Section validation completed (section: %s, outcome: %s, duration: %.2fs, findings: %d)",
                val_res.section_name,
                val_res.outcome,
                duration,
                len(val_res.findings),
            )
            return val_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Section Validation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            return ValidationResult(
                section_name=ctx.section_name,
                outcome=ValidationOutcome.NEEDS_REWORK,
                summary=f"Section validation failed due to runtime error: {err_msg}",
                findings=[
                    ValidationFinding(
                        category=ValidationCategory.CONSISTENCY,
                        issue="Validation execution error",
                        explanation=err_msg,
                        required_change="Re-run validation after resolving execution error",
                    )
                ],
                rework_feedback=f"Validation runtime error occurred: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                },
            )

    async def validate_async(
        self,
        context: SectionValidationContext | dict[str, Any],
    ) -> ValidationResult:
        """Validate a generated or updated BRD section asynchronously.

        Args:
            context: Scoped SectionValidationContext or dictionary.

        Returns:
            ValidationResult: Structured result with outcome, findings, and rework feedback.
        """
        ctx = SectionValidationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "Async section validation started (section: %s)",
            ctx.section_name,
        )

        try:
            prompt_input = _build_validation_prompt(ctx)
            result = await self._graph.ainvoke({"messages": [HumanMessage(content=prompt_input)]})
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

            val_res = _parse_validation_response(output_text, ctx, duration=duration)
            logger.info(
                "Async section validation completed (section: %s, outcome: %s, duration: %.2fs, findings: %d)",
                val_res.section_name,
                val_res.outcome,
                duration,
                len(val_res.findings),
            )
            return val_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Async Section Validation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            return ValidationResult(
                section_name=ctx.section_name,
                outcome=ValidationOutcome.NEEDS_REWORK,
                summary=f"Section validation failed due to runtime error: {err_msg}",
                findings=[
                    ValidationFinding(
                        category=ValidationCategory.CONSISTENCY,
                        issue="Validation execution error",
                        explanation=err_msg,
                        required_change="Re-run validation after resolving execution error",
                    )
                ],
                rework_feedback=f"Validation runtime error occurred: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                },
            )
