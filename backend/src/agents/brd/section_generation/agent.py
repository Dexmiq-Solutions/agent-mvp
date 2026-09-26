"""BRD Section Generation / Update Sub-Agent implementation.

Provides the dedicated, reusable authoring capability for the BRD Agent architecture.
The Section Generation Sub-Agent generates or updates a specific BRD section based on
available information, template requirements, and optional existing content/rework feedback.

Strict Architectural Boundaries:
- Section Generator generates/updates; the BRD Lead Agent orchestrates and owns the workflow.
- Generator has NO tools (cannot call RAG, access databases, or invoke external tools).
- Generator does not interact with the user or perform evidence evaluation or quality validation.
- Generator strictly adheres to authoritative template structures and does not invent facts.
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
from agents.brd.template import extract_section_requirements, extract_section_template
from observability.logging import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"


def get_generation_system_instruction_path() -> Path:
    """Resolve the absolute path to the section generation system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_generation_system_instruction() -> str:
    """Load the authoritative section generation system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    path = get_generation_system_instruction_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required section generation system instruction file not found: {path}"
        )

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error(
            "Failed to read section generation system instruction from %s: %s", path, exc
        )
        raise

    if not content:
        raise ValueError(
            f"Section generation system instruction file is empty: {path}"
        )

    return content


class SectionOperation(str, Enum):
    """Operation mode for section authoring: initial creation vs update/rework."""

    GENERATE = "generate"
    UPDATE = "update"

    @classmethod
    def from_string(cls, val: str | "SectionOperation") -> "SectionOperation":
        """Convert a string or enum to normalized SectionOperation."""
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or SectionOperation, got {type(val)}")

        normalized = val.strip().lower()
        if "update" in normalized or "rework" in normalized or "revision" in normalized:
            return cls.UPDATE
        if "generate" in normalized or "create" in normalized or "initial" in normalized:
            return cls.GENERATE

        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(
            f"Invalid section operation '{val}'. Expected 'generate' or 'update'."
        )


@dataclass
class SectionGenerationContext:
    """Scoped context provided as input to the Section Generation Sub-Agent.

    Attributes:
        section_name: Target section name (e.g. "5. Stakeholders & Personas").
        section_requirements: Required items or criteria the section must address.
        template_structure: Authoritative Markdown structure/skeleton from brd_template.md.
        available_information: Evidence, facts, and working context ready for generation.
        existing_content: Optional existing section content for update/rework.
        rework_feedback: Optional specific feedback from section validation or review.
        operation: GENERATE or UPDATE. Auto-detected from existing_content if not explicitly set.
        metadata: Extensible metadata dictionary (e.g. project_id, section_id).
    """

    section_name: str
    section_requirements: list[str] = field(default_factory=list)
    template_structure: str = ""
    available_information: Any = field(default_factory=list)
    existing_content: Optional[str] = None
    rework_feedback: Optional[str] = None
    operation: SectionOperation = SectionOperation.GENERATE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-configure operation mode if existing content is present."""
        if isinstance(self.operation, str):
            self.operation = SectionOperation.from_string(self.operation)
        if self.existing_content and self.existing_content.strip() and self.operation == SectionOperation.GENERATE:
            self.operation = SectionOperation.UPDATE

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to JSON-compatible dictionary."""
        op_val = self.operation.value if isinstance(self.operation, SectionOperation) else str(self.operation)
        return {
            "section_name": self.section_name,
            "section_requirements": list(self.section_requirements),
            "template_structure": self.template_structure,
            "available_information": self.available_information,
            "existing_content": self.existing_content,
            "rework_feedback": self.rework_feedback,
            "operation": op_val,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionGenerationContext":
        """Deserialize dictionary into a SectionGenerationContext."""
        op_raw = data.get("operation")
        op = SectionOperation.from_string(op_raw) if op_raw else SectionOperation.GENERATE
        return cls(
            section_name=str(data.get("section_name", "")),
            section_requirements=list(data.get("section_requirements", [])),
            template_structure=str(data.get("template_structure", "")),
            available_information=data.get("available_information", []),
            existing_content=data.get("existing_content"),
            rework_feedback=data.get("rework_feedback"),
            operation=op,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SectionGenerationResult:
    """Structured result returned by the Section Generation Sub-Agent.

    Attributes:
        section_name: Target section identifier / heading.
        content: Complete Markdown content of the generated or updated section.
        operation: SectionOperation (GENERATE or UPDATE).
        summary: Brief summary of the section creation or updates performed.
        metadata: Generation runtime metadata (e.g. duration, timestamp).
    """

    section_name: str
    content: str
    operation: SectionOperation | str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_update(self) -> bool:
        """Return True if this operation was an update/rework."""
        op_val = self.operation.value if isinstance(self.operation, SectionOperation) else str(self.operation)
        return op_val.strip().lower() == SectionOperation.UPDATE.value

    @property
    def is_generation(self) -> bool:
        """Return True if this operation was an initial generation."""
        op_val = self.operation.value if isinstance(self.operation, SectionOperation) else str(self.operation)
        return op_val.strip().lower() == SectionOperation.GENERATE.value

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        op_val = self.operation.value if isinstance(self.operation, SectionOperation) else str(self.operation)
        return {
            "section_name": self.section_name,
            "content": self.content,
            "operation": op_val,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionGenerationResult":
        """Deserialize dictionary into a SectionGenerationResult."""
        raw_op = data.get("operation", SectionOperation.GENERATE.value)
        try:
            op = SectionOperation.from_string(raw_op)
        except ValueError:
            op = raw_op

        return cls(
            section_name=str(data.get("section_name", "")),
            content=str(data.get("content", "")),
            operation=op,
            summary=str(data.get("summary", "")),
            metadata=dict(data.get("metadata", {})),
        )


# extract_section_template is imported from agents.brd.template for backward compatibility
__all__ = ["extract_section_template", "extract_section_requirements"]


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
                # Format dict item (e.g. from evidence list or action results)
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


def _build_generation_prompt(context: SectionGenerationContext) -> str:
    """Format prompt with focused context for the Section Generation Sub-Agent."""
    op_label = "Section Update / Rework" if context.operation == SectionOperation.UPDATE else "Initial Section Generation"
    parts: list[str] = [
        f"## Operation Mode\n{op_label}",
        f"## Target Section\n{context.section_name}",
    ]

    if context.template_structure:
        parts.append(
            f"## Authoritative Template Structure & Format\n```markdown\n{context.template_structure}\n```"
        )

    if context.section_requirements:
        req_lines = [f"- {r}" for r in context.section_requirements]
        parts.append("## Section Requirements & Criteria\n" + "\n".join(req_lines))

    info_str = _format_available_information(context.available_information)
    parts.append(f"## Available Information & Evidence\n{info_str}")

    if context.existing_content and context.existing_content.strip():
        parts.append(
            f"## Current Section Content (Baseline to Update)\n```markdown\n{context.existing_content.strip()}\n```"
        )

    if context.rework_feedback and context.rework_feedback.strip():
        parts.append(
            f"## Rework Feedback to Address\n{context.rework_feedback.strip()}"
        )

    parts.append(
        "Generate the complete section according to the template structure and supplied information.\n"
        "If updating an existing section, incorporate updates and rework feedback while preserving valid existing content.\n"
        "Ground all statements strictly in the provided evidence. Do NOT fabricate project facts.\n"
        "Return ONLY a valid JSON object matching the output schema."
    )

    return "\n\n".join(parts)


def _parse_generation_response(
    response_text: str,
    context: SectionGenerationContext,
    duration: float = 0.0,
) -> SectionGenerationResult:
    """Parse LLM output text into structured SectionGenerationResult, with robust JSON extraction and fallback."""
    # Attempt to locate JSON object
    json_match = re.search(r"\{\s*\"section_name\".*\}\s*", response_text, re.DOTALL)
    if not json_match:
        # Try broader JSON matching
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if json_match:
        raw_json = json_match.group(0)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "content" in parsed:
                content = str(parsed.get("content", "")).strip()
                summary = str(parsed.get("summary", ""))
                sec_name = str(parsed.get("section_name", context.section_name))
                raw_op = parsed.get("operation", context.operation.value)
                try:
                    op = SectionOperation.from_string(raw_op)
                except ValueError:
                    op = context.operation

                return SectionGenerationResult(
                    section_name=sec_name,
                    content=content,
                    operation=op,
                    summary=summary or f"Section {sec_name} {'updated' if op == SectionOperation.UPDATE else 'generated'}.",
                    metadata={
                        "duration_seconds": duration,
                    },
                )
        except Exception as exc:
            logger.warning("Failed to parse extracted JSON in generation response: %s", exc)

    # Heuristic fallback if direct Markdown was returned instead of JSON
    cleaned_content = response_text.strip()
    # Strip potential outer markdown code fences if wrapped
    if cleaned_content.startswith("```markdown") and cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[len("```markdown"): -3].strip()
    elif cleaned_content.startswith("```") and cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[3:-3].strip()

    summary = (
        f"Section {context.section_name} updated."
        if context.operation == SectionOperation.UPDATE
        else f"Section {context.section_name} generated."
    )

    return SectionGenerationResult(
        section_name=context.section_name,
        content=cleaned_content,
        operation=context.operation,
        summary=summary,
        metadata={
            "duration_seconds": duration,
            "fallback_parsing": True,
        },
    )


class BRDSectionGenerationAgent:
    """Dedicated, reusable Section Generation / Update Sub-Agent capability.

    Authors or updates a specific BRD section on behalf of the BRD Lead Agent.
    Adheres strictly to the authoritative BRD template structure and grounds
    all content in supplied evidence without inventing project facts.

    Architectural Invariants:
    1. Operates strictly as a section author; never decides workflow progression.
    2. Equipped with NO tools (tools=[]); cannot call RAG, query databases, or fetch data.
    3. Does not interact with user, delegate tasks, or validate its own output.
    4. Supports both initial generation and updates/rework seamlessly.
    """

    agent_name: str = "BRDSectionGenerationAgent"

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        config: Optional[AgentConfig] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        """Initialize the Section Generation Sub-Agent.

        Args:
            model: Optional pre-configured BaseChatModel instance.
            config: Optional AgentConfig instance. Used if model is omitted.
            system_instruction: Optional system instruction override. If omitted,
                loaded from system_instruction.md.
        """
        self._system_instruction = system_instruction or load_generation_system_instruction()
        self._model = model or create_agent_model(config or AgentConfig.from_settings())

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
        """Return the authoritative section generation system instruction."""
        return self._system_instruction

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools (always empty for Section Generation Sub-Agent)."""
        return self._tools

    @property
    def graph(self) -> Any:
        """Return the underlying DeepAgents execution graph."""
        return self._graph

    def generate(
        self,
        context: SectionGenerationContext | dict[str, Any],
    ) -> SectionGenerationResult:
        """Generate or update a BRD section synchronously.

        Args:
            context: Scoped SectionGenerationContext or dictionary.

        Returns:
            SectionGenerationResult: Structured result with generated/updated section content.
        """
        ctx = SectionGenerationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "Section generation started (section: %s, operation: %s)",
            ctx.section_name,
            ctx.operation.value,
        )

        try:
            prompt_input = _build_generation_prompt(ctx)
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

            gen_res = _parse_generation_response(output_text, ctx, duration=duration)
            logger.info(
                "Section generation completed (section: %s, operation: %s, duration: %.2fs, content_len: %d)",
                gen_res.section_name,
                gen_res.operation,
                duration,
                len(gen_res.content),
            )
            return gen_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Section Generation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            return SectionGenerationResult(
                section_name=ctx.section_name,
                content=ctx.existing_content or "",
                operation=ctx.operation,
                summary=f"Section generation failed due to runtime error: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                },
            )

    async def generate_async(
        self,
        context: SectionGenerationContext | dict[str, Any],
    ) -> SectionGenerationResult:
        """Generate or update a BRD section asynchronously.

        Args:
            context: Scoped SectionGenerationContext or dictionary.

        Returns:
            SectionGenerationResult: Structured result with generated/updated section content.
        """
        ctx = SectionGenerationContext.from_dict(context) if isinstance(context, dict) else context

        start_time = time.perf_counter()
        logger.info(
            "Async section generation started (section: %s, operation: %s)",
            ctx.section_name,
            ctx.operation.value,
        )

        try:
            prompt_input = _build_generation_prompt(ctx)
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

            gen_res = _parse_generation_response(output_text, ctx, duration=duration)
            logger.info(
                "Async section generation completed (section: %s, operation: %s, duration: %.2fs, content_len: %d)",
                gen_res.section_name,
                gen_res.operation,
                duration,
                len(gen_res.content),
            )
            return gen_res

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc).strip() or exc.__class__.__name__
            logger.error(
                "Async Section Generation Sub-Agent execution failed (duration: %.2fs): %s",
                duration,
                err_msg,
                exc_info=True,
            )
            return SectionGenerationResult(
                section_name=ctx.section_name,
                content=ctx.existing_content or "",
                operation=ctx.operation,
                summary=f"Section generation failed due to runtime error: {err_msg}",
                metadata={
                    "error": err_msg,
                    "duration_seconds": duration,
                },
            )
