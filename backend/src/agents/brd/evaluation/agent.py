"""BRD Evaluation Sub-Agent implementation.

Provides the dedicated, reusable evaluation capability for the BRD Agent architecture.
The Evaluation Sub-Agent evaluates whether an ActionResult (from Direct Work, RAG, or Delegation)
contains sufficient evidence, detail, and clarity for the current BRD objective and section.

Strict Architectural Boundaries:
- The Evaluation Sub-Agent evaluates. The BRD Lead Agent decides what happens next.
- Evaluator has NO tools (cannot call RAG, access databases, or invoke external tools).
- Evaluator does not interact with the user or generate/modify BRD documents.
- Evaluator uses categorical evaluation (SUFFICIENT vs INSUFFICIENT), not arbitrary numeric scores.
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

from agents.runtime.agent import create_runtime_agent
from agents.runtime.config import AgentConfig
from agents.runtime.model import create_agent_model
from agents.runtime.state import ActionResult, ActionSource, AgentContext
from observability.logging import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"


def get_evaluation_system_instruction_path() -> Path:
    """Resolve the absolute path to the evaluation system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_evaluation_system_instruction() -> str:
    """Load the authoritative evaluation system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    path = get_evaluation_system_instruction_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required evaluation system instruction file not found: {path}"
        )

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read evaluation system instruction from %s: %s", path, exc)
        raise

    if not content:
        raise ValueError(f"Evaluation system instruction file is empty: {path}")

    return content


class EvaluationOutcome(str, Enum):
    """Categorical outcome of an evidence evaluation."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"

    @classmethod
    def from_string(cls, val: str | "EvaluationOutcome") -> "EvaluationOutcome":
        """Convert a string or enum to normalized EvaluationOutcome."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or EvaluationOutcome, got {type(val)}")

        normalized = val.strip().upper()
        if "SUFFICIENT" in normalized and "INSUFFICIENT" not in normalized:
            return cls.SUFFICIENT
        if "INSUFFICIENT" in normalized:
            return cls.INSUFFICIENT

        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"Invalid evaluation outcome '{val}'. Expected SUFFICIENT or INSUFFICIENT.")


class InformationStatus(str, Enum):
    """Status classification for individual requirement items."""

    PRESENT = "Present"
    MISSING = "Missing"
    UNCLEAR = "Unclear / Unresolved"
    CONTRADICTORY = "Contradictory"
    NOT_APPLICABLE = "Not Applicable"

    @classmethod
    def from_string(cls, val: str | "InformationStatus") -> "InformationStatus":
        """Convert string or enum to normalized InformationStatus."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or InformationStatus, got {type(val)}")

        norm = val.strip().lower().replace("_", " ").replace("-", " ")
        for s in cls:
            if s.value.lower() == norm or s.name.lower() == norm.replace(" ", "_"):
                return s
        return cls.UNCLEAR


@dataclass
class EvaluationFinding:
    """Key observation made during evaluation explaining what was observed and why it matters.

    Attributes:
        observation: What was observed in the evaluated content.
        significance: Why this observation matters for the BRD objective/section.
        status: Categorical status of the requirement item (Present, Missing, etc.).
        item: Optional specific requirement or topic name.
        metadata: Extensible metadata dictionary.
    """

    observation: str
    significance: str = ""
    status: Optional[InformationStatus | str] = None
    item: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize finding to dictionary."""
        stat_val = (
            self.status.value
            if isinstance(self.status, InformationStatus)
            else (str(self.status) if self.status is not None else None)
        )
        return {
            "observation": self.observation,
            "significance": self.significance,
            "status": stat_val,
            "item": self.item,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationFinding":
        """Deserialize dictionary into an EvaluationFinding."""
        status_raw = data.get("status")
        status = None
        if status_raw is not None:
            try:
                status = InformationStatus.from_string(status_raw)
            except ValueError:
                status = str(status_raw)

        return cls(
            observation=str(data.get("observation", "")),
            significance=str(data.get("significance", "")),
            status=status,
            item=data.get("item"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_dict_or_str(cls, val: Any) -> "EvaluationFinding":
        """Convert a string or dictionary into an EvaluationFinding."""
        if isinstance(val, cls):
            return val
        if isinstance(val, dict):
            return cls.from_dict(val)
        return cls(observation=str(val), significance="")


@dataclass
class EvaluationContext:
    """Scoped context provided as input to the Evaluation Sub-Agent.

    Attributes:
        current_objective: The specific BRD objective being worked on.
        current_section: The specific BRD section currently being researched or generated.
        section_requirements: Required items or criteria the current section must contain.
        relevant_working_context: Scoped working context necessary for evaluation.
        action_result: The common ActionResult to evaluate.
        metadata: Extensible metadata dictionary (e.g. project_id, section_id).
    """

    current_objective: str
    current_section: str
    section_requirements: list[str] = field(default_factory=list)
    relevant_working_context: str = ""
    action_result: Optional[ActionResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "current_objective": self.current_objective,
            "current_section": self.current_section,
            "section_requirements": list(self.section_requirements),
            "relevant_working_context": self.relevant_working_context,
            "action_result": self.action_result.to_dict() if self.action_result else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationContext":
        """Deserialize dictionary into an EvaluationContext."""
        act_raw = data.get("action_result")
        action_result = ActionResult.from_dict(act_raw) if isinstance(act_raw, dict) else act_raw
        return cls(
            current_objective=str(data.get("current_objective", "")),
            current_section=str(data.get("current_section", "")),
            section_requirements=list(data.get("section_requirements", [])),
            relevant_working_context=str(data.get("relevant_working_context", "")),
            action_result=action_result,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvaluationResult:
    """Structured evaluation result produced by the Evaluation Sub-Agent.

    Attributes:
        outcome: SUFFICIENT or INSUFFICIENT.
        summary: Concise explanation of why the result was considered sufficient or insufficient.
        findings: Key qualitative observations made during evaluation.
        missing_information: Specific required information not present in the action result.
        unresolved_information: Information that is present but ambiguous or unresolved.
        contradictions: Conflicting statements that prevent safe acceptance.
        metadata: Evaluation runtime metadata (e.g. duration, timestamp, action_source).
    """

    outcome: EvaluationOutcome | str
    summary: str
    findings: list[EvaluationFinding] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    unresolved_information: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sufficient(self) -> bool:
        """Return True if the evaluation outcome is SUFFICIENT."""
        val = self.outcome.value if isinstance(self.outcome, EvaluationOutcome) else str(self.outcome)
        return val.strip().upper() == EvaluationOutcome.SUFFICIENT.value

    @property
    def is_insufficient(self) -> bool:
        """Return True if the evaluation outcome is INSUFFICIENT."""
        return not self.is_sufficient

    def to_dict(self) -> dict[str, Any]:
        """Serialize EvaluationResult to JSON-compatible dictionary."""
        out_val = self.outcome.value if isinstance(self.outcome, EvaluationOutcome) else str(self.outcome)
        return {
            "outcome": out_val,
            "summary": self.summary,
            "findings": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.findings],
            "missing_information": list(self.missing_information),
            "unresolved_information": list(self.unresolved_information),
            "contradictions": list(self.contradictions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Deserialize dictionary into an EvaluationResult."""
        raw_outcome = data.get("outcome", EvaluationOutcome.INSUFFICIENT.value)
        try:
            outcome = EvaluationOutcome.from_string(raw_outcome)
        except ValueError:
            outcome = raw_outcome

        raw_findings = data.get("findings", [])
        findings = [
            EvaluationFinding.from_dict_or_str(f)
            for f in raw_findings
        ]

        return cls(
            outcome=outcome,
            summary=str(data.get("summary", "")),
            findings=findings,
            missing_information=list(data.get("missing_information", [])),
            unresolved_information=list(data.get("unresolved_information", [])),
            contradictions=list(data.get("contradictions", [])),
            metadata=dict(data.get("metadata", {})),
        )


def extract_section_requirements(section_name: str, template_content: Optional[str] = None) -> list[str]:
    """Extract required items / criteria for a given section from the authoritative BRD template.

    Reuses the existing template structure as the single source of truth without
    creating a parallel document schema.

    Args:
        section_name: Target section name (e.g. "Personas", "1. Purpose & Scope of This Document").
        template_content: Optional raw Markdown content of the BRD template.

    Returns:
        list[str]: Extracted requirement points for the section.
    """
    from agents.brd.agent import load_brd_template

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
    lines = [line.strip() for line in section_body.splitlines() if line.strip() and not line.strip().startswith("#")]
    if lines:
        return lines

    return [f"Requirements for {section_name}"]


def _build_evaluation_prompt(context: EvaluationContext) -> str:
    """Format input prompt containing focused evaluation context for the Evaluation Sub-Agent."""
    parts: list[str] = [
        f"## Current Objective\n{context.current_objective}",
        f"## Current Section\n{context.current_section}",
    ]

    if context.section_requirements:
        req_lines = [f"- {r}" for r in context.section_requirements]
        parts.append(f"## Section Requirements\n" + "\n".join(req_lines))

    if context.relevant_working_context:
        parts.append(f"## Relevant Working Context\n{context.relevant_working_context}")

    if context.action_result:
        act = context.action_result
        src = act.source.value if hasattr(act.source, "value") else str(act.source)
        act_lines = [
            f"- Source: {src}",
            f"- Execution Succeeded: {act.success}",
        ]
        if act.error:
            act_lines.append(f"- Execution Error: {act.error}")
        act_lines.append(f"\n### Action Result Content:\n{act.content}")
        parts.append("## Action Result to Evaluate\n" + "\n".join(act_lines))
    else:
        parts.append("## Action Result to Evaluate\n*(No Action Result provided)*")

    parts.append(
        "Evaluate the Action Result against the Current Objective and Section Requirements.\n"
        "Categorize each requirement item status (Present, Missing, Unclear / Unresolved, Contradictory, Not Applicable).\n"
        "Return ONLY a valid JSON object matching the output schema with outcome SUFFICIENT or INSUFFICIENT."
    )

    return "\n\n".join(parts)


def _parse_evaluation_response(
    response_text: str,
    context: EvaluationContext,
    duration: float = 0.0,
) -> EvaluationResult:
    """Parse LLM output text into structured EvaluationResult, with robust JSON extraction and fallback."""
    # Attempt to locate JSON object
    json_match = re.search(r"\{\s*\"outcome\".*\}\s*", response_text, re.DOTALL)
    if not json_match:
        # Try broader JSON matching
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if json_match:
        raw_json = json_match.group(0)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "outcome" in parsed:
                eval_res = EvaluationResult.from_dict(parsed)
                eval_res.metadata.update({
                    "duration_seconds": duration,
                    "action_source": str(context.action_result.source) if context.action_result else None,
                })
                return eval_res
        except Exception as exc:
            logger.warning("Failed to parse extracted JSON in evaluation response: %s", exc)

    # Heuristic fallback if JSON format wasn't returned
    text_upper = response_text.upper()
    is_sufficient = "SUFFICIENT" in text_upper and "INSUFFICIENT" not in text_upper
    outcome = EvaluationOutcome.SUFFICIENT if is_sufficient else EvaluationOutcome.INSUFFICIENT

    # Extract missing items or bullet points
    missing_items: list[str] = []
    unresolved_items: list[str] = []
    contradictions: list[str] = []
    findings: list[EvaluationFinding] = []

    missing_match = re.search(r"missing\s*(?:information|items)?:?\s*\n((?:[*-]\s*.+\n?)+)", response_text, re.IGNORECASE)
    if missing_match:
        missing_items = [
            line.strip().lstrip("*- ").strip()
            for line in missing_match.group(1).splitlines()
            if line.strip()
        ]

    summary = response_text.splitlines()[0] if response_text else "Evaluation completed via fallback parser."

    return EvaluationResult(
        outcome=outcome,
        summary=summary,
        findings=findings or [EvaluationFinding(observation=response_text[:300], significance="Heuristic extraction")],
        missing_information=missing_items,
        unresolved_information=unresolved_items,
        contradictions=contradictions,
        metadata={
            "duration_seconds": duration,
            "fallback_parsing": True,
            "action_source": str(context.action_result.source) if context.action_result else None,
        },
    )


class BRDEvaluationAgent:
    """Dedicated, reusable Evaluation Sub-Agent capability for the BRD Lead Agent.

    Evaluates whether an ActionResult (Direct Work, RAG, or Delegation) satisfies
    the requirements of the current BRD objective and section.

    Architectural Invariants:
    1. Operates strictly as an evaluator; never decides next workflow action.
    2. Equipped with NO tools (tools=[]); cannot call RAG, access databases, or query storage.
    3. Does not interact with user, delegate tasks, or modify BRD template/documents.
    4. Categorical evaluation (SUFFICIENT / INSUFFICIENT) without arbitrary numeric scores.
    """

    agent_name: str = "BRDEvaluationAgent"

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        config: Optional[AgentConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        """Initialize the Evaluation Sub-Agent.

        Args:
            model: Optional pre-configured BaseChatModel instance.
            config: Optional AgentConfig instance. Used if model is omitted.
            system_instruction: Optional system instruction override. If omitted,
                loaded from system_instruction.md.
        """
        self._system_instruction = system_instruction or load_evaluation_system_instruction()
        self._model = model or create_agent_model(config or AgentConfig.from_settings())

        # Strict boundary: Evaluator has NO tools
        self._tools: list[Any] = []

        # DeepAgents runtime graph without tools
        self._graph = create_runtime_agent(
            model=self._model,
            tools=self._tools,
            system_prompt=self._system_instruction,
        )

    @property
    def system_instruction(self) -> str:
        """Return the authoritative evaluation system instruction."""
        return self._system_instruction

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools (always empty for Evaluation Sub-Agent)."""
        return self._tools

    @property
    def graph(self) -> Any:
        """Return the underlying DeepAgents execution graph."""
        return self._graph

    def evaluate(
        self,
        context: EvaluationContext | dict[str, Any],
        action_result: Optional[ActionResult] = None,
    ) -> EvaluationResult:
        """Evaluate an ActionResult against current objective and section requirements synchronously.

        Args:
            context: Scoped EvaluationContext or dictionary.
            action_result: Optional ActionResult override if not set in context.

        Returns:
            EvaluationResult: Structured evaluation with outcome, findings, and gaps.
        """
        ctx = EvaluationContext.from_dict(context) if isinstance(context, dict) else context
        if action_result is not None:
            ctx.action_result = action_result

        start_time = time.perf_counter()
        logger.info(
            "Evaluation started (objective: %s, section: %s, action_source: %s)",
            ctx.current_objective,
            ctx.current_section,
            ctx.action_result.source if ctx.action_result else "none",
        )

        try:
            prompt_input = _build_evaluation_prompt(ctx)
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

            eval_res = _parse_evaluation_response(output_text, ctx, duration=duration)
            logger.info(
                "Evaluation completed (outcome: %s, duration: %.2fs, findings: %d, missing: %d)",
                eval_res.outcome,
                duration,
                len(eval_res.findings),
                len(eval_res.missing_information),
            )
            return eval_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Evaluation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            # Evaluation failure must NOT appear as successful evaluation
            return EvaluationResult(
                outcome=EvaluationOutcome.INSUFFICIENT,
                summary=f"Evaluation failed due to runtime error: {err_msg}",
                findings=[EvaluationFinding(observation=f"Error executing evaluation: {err_msg}", significance="Execution failure prevents verification")],
                missing_information=[f"Evaluation execution failed: {err_msg}"],
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                    "action_source": str(ctx.action_result.source) if ctx.action_result else None,
                },
            )

    async def evaluate_async(
        self,
        context: EvaluationContext | dict[str, Any],
        action_result: Optional[ActionResult] = None,
    ) -> EvaluationResult:
        """Evaluate an ActionResult against current objective and section requirements asynchronously.

        Args:
            context: Scoped EvaluationContext or dictionary.
            action_result: Optional ActionResult override if not set in context.

        Returns:
            EvaluationResult: Structured evaluation with outcome, findings, and gaps.
        """
        ctx = EvaluationContext.from_dict(context) if isinstance(context, dict) else context
        if action_result is not None:
            ctx.action_result = action_result

        start_time = time.perf_counter()
        logger.info(
            "Async evaluation started (objective: %s, section: %s, action_source: %s)",
            ctx.current_objective,
            ctx.current_section,
            ctx.action_result.source if ctx.action_result else "none",
        )

        try:
            prompt_input = _build_evaluation_prompt(ctx)
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

            eval_res = _parse_evaluation_response(output_text, ctx, duration=duration)
            logger.info(
                "Async evaluation completed (outcome: %s, duration: %.2fs, findings: %d, missing: %d)",
                eval_res.outcome,
                duration,
                len(eval_res.findings),
                len(eval_res.missing_information),
            )
            return eval_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Async Evaluation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            return EvaluationResult(
                outcome=EvaluationOutcome.INSUFFICIENT,
                summary=f"Evaluation failed due to runtime error: {err_msg}",
                findings=[EvaluationFinding(observation=f"Error executing evaluation: {err_msg}", significance="Execution failure prevents verification")],
                missing_information=[f"Evaluation execution failed: {err_msg}"],
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                    "action_source": str(ctx.action_result.source) if ctx.action_result else None,
                },
            )
