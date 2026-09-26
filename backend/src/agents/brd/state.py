"""BRD Agent State models and lifecycle tracking.

Defines the working context and operational state for the BRD Lead Agent:
- Overall BRD objective vs immediate working task
- Authoritative template sections and progress tracking
- Evidence and working information
- Unresolved information and gaps
- Integration with DeepAgents/Agent runtime
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Optional, Sequence, Union

from deepagents import DeepAgentState

from agents.brd.delegation import DelegatedTask, DelegationResult, TaskResult
from agents.brd.evaluation.agent import EvaluationResult
from agents.brd.section_validation.agent import ValidationResult
from agents.runtime.state import ActionResult


class BRDSectionStatus(str, Enum):
    """Lifecycle status of a BRD section as defined in the context foundation."""

    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    NEEDS_REVISION = "Needs Revision"

    @classmethod
    def from_string(cls, val: str | "BRDSectionStatus") -> "BRDSectionStatus":
        """Convert string or enum to normalized BRDSectionStatus.

        Supports case-insensitive, space-separated, or snake_case inputs.
        """
        if isinstance(val, cls):
            return val
        if not isinstance(val, str):
            raise ValueError(f"Expected str or BRDSectionStatus, got {type(val)}")

        normalized = val.strip().lower().replace("_", " ").replace("-", " ")
        for status in cls:
            if status.value.lower() == normalized or status.name.lower() == normalized.replace(" ", "_"):
                return status

        try:
            return cls(val)
        except ValueError:
            valid = [s.value for s in cls]
            raise ValueError(
                f"Invalid section status '{val}'. Expected one of: {valid}"
            )


# Backward-compatible alias
SectionStatus = BRDSectionStatus


def _match_section_name(name: str, available_sections: Sequence[str]) -> str:
    """Match a section name against available sections, supporting clean titles.

    Allows matching "Introduction" to "1. Introduction" or exact matches.
    """
    cleaned_input = name.strip()
    # 1. Exact match
    for sec in available_sections:
        if sec.lower() == cleaned_input.lower():
            return sec

    # 2. Number-stripped match (e.g. "1. Introduction" -> "Introduction")
    input_stripped = re.sub(r"^\d+[\.\)]\s*", "", cleaned_input).strip().lower()
    for sec in available_sections:
        sec_stripped = re.sub(r"^\d+[\.\)]\s*", "", sec).strip().lower()
        if sec_stripped == input_stripped:
            return sec

    return cleaned_input


from agents.brd.progression import SectionProgressionResult
from agents.brd.assembly import BRDAssemblyResult
from agents.brd.final_validation.agent import FinalValidationResult


@dataclass
class BRDAgentState:
    """Working context and operational state for the BRD Lead Agent.

    Represents what the Agent currently knows, is working on, and needs to remember
    while progressing toward the BRD generation objective.

    Attributes:
        objective: Overall BRD goal (e.g. "Produce an evidence-grounded Business Requirements Document").
        current_task: Immediate task or objective context currently being executed.
        template_sections: Ordered list of required BRD sections derived from the template.
        current_section: The specific section the Agent is currently focused on.
        section_progress: Map of section names to their current BRDSectionStatus.
        evidence: Working information and evidence collected during execution.
        unresolved_information: Gaps, ambiguities, or missing information requiring resolution.
        metadata: Extensible metadata dictionary (e.g. project_id, session notes).
    """

    objective: str = "Produce an evidence-grounded Business Requirements Document"
    current_task: Optional[str] = None
    template_sections: list[str] = field(default_factory=list)
    current_section: Optional[str] = None
    section_progress: dict[str, BRDSectionStatus] = field(default_factory=dict)
    evidence: list[Any] = field(default_factory=list)
    unresolved_information: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    delegated_tasks: list[DelegatedTask] = field(default_factory=list)
    task_results: list[TaskResult] = field(default_factory=list)
    delegation_result: Optional[DelegationResult] = None
    latest_action_result: Optional[ActionResult] = None
    latest_evaluation_result: Optional[EvaluationResult] = None
    evaluation_history: list[EvaluationResult] = field(default_factory=list)
    section_content: dict[str, str] = field(default_factory=dict)
    rework_feedback: dict[str, str] = field(default_factory=dict)
    latest_section_result: Optional[Any] = None
    latest_validation_result: Optional[ValidationResult] = None
    validation_history: list[ValidationResult] = field(default_factory=list)
    latest_progression_result: Optional[SectionProgressionResult] = None
    progression_history: list[SectionProgressionResult] = field(default_factory=list)
    assembled_brd: Optional[str] = None
    latest_assembly_result: Optional[BRDAssemblyResult] = None
    latest_final_validation_result: Optional[FinalValidationResult] = None
    final_validation_history: list[FinalValidationResult] = field(default_factory=list)
    final_validation_recovery_cycles: int = 0
    final_validation_recovery_exhausted: bool = False

    @classmethod
    def initialize_from_template(
        cls,
        sections: Sequence[str],
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BRDAgentState":
        """Initialize a fresh BRDAgentState using the sections extracted from the BRD template.

        All sections default to BRDSectionStatus.NOT_STARTED.
        """
        sec_list = list(sections)
        progress = {s: BRDSectionStatus.NOT_STARTED for s in sec_list}
        return cls(
            objective=objective or "Produce an evidence-grounded Business Requirements Document",
            current_task=current_task,
            template_sections=sec_list,
            current_section=None,
            section_progress=progress,
            evidence=[],
            unresolved_information=[],
            metadata=dict(metadata or {}),
        )

    def set_current_task(self, task: Optional[str]) -> None:
        """Set the immediate task or objective context currently being executed."""
        self.current_task = task

    def set_delegated_tasks(self, tasks: Sequence[DelegatedTask]) -> None:
        """Store the list of delegated tasks decomposed from the Lead Agent objective."""
        self.delegated_tasks = list(tasks)

    def add_task_result(self, result: TaskResult) -> None:
        """Record an attributable result from a temporary sub-agent execution."""
        self.task_results.append(result)

    def set_delegation_result(self, result: Optional[DelegationResult]) -> None:
        """Store the consolidated delegation result."""
        self.delegation_result = result

    def set_action_result(self, action_result: Optional[ActionResult]) -> None:
        """Store the latest unified action result (converging Direct Work, RAG, or Delegation)."""
        self.latest_action_result = action_result

    def set_evaluation_result(self, result: Optional[EvaluationResult]) -> None:
        """Store the latest evaluation result and record it in history."""
        self.latest_evaluation_result = result
        if result is not None:
            self.evaluation_history.append(result)

    def get_latest_evaluation_result(self) -> Optional[EvaluationResult]:
        """Retrieve the latest evaluation result."""
        return self.latest_evaluation_result

    def _available_section_keys(self) -> list[str]:
        """Aggregate all available section identifiers across state structures."""
        keys = (
            list(self.section_content.keys())
            + list(self.section_progress.keys())
            + list(self.template_sections)
        )
        return list(dict.fromkeys(keys))

    def get_section_content(self, section: str) -> Optional[str]:
        """Get the stored Markdown content for a given section."""
        canonical = _match_section_name(section, self._available_section_keys())
        return self.section_content.get(canonical)

    def set_section_content(self, section: str, content: str) -> None:
        """Store or update the Markdown content for a given section."""
        canonical = _match_section_name(section, self._available_section_keys())
        self.section_content[canonical] = content

    def get_rework_feedback(self, section: str) -> Optional[str]:
        """Get the validation / rework feedback for a given section."""
        canonical = _match_section_name(section, self._available_section_keys())
        return self.rework_feedback.get(canonical)

    def set_rework_feedback(self, section: str, feedback: str) -> None:
        """Record validation / rework feedback for a given section."""
        canonical = _match_section_name(section, self._available_section_keys())
        self.rework_feedback[canonical] = feedback

    def clear_rework_feedback(self, section: str) -> None:
        """Clear validation / rework feedback once addressed."""
        canonical = _match_section_name(section, self._available_section_keys())
        self.rework_feedback.pop(canonical, None)

    def set_section_result(self, result: Optional[Any]) -> None:
        """Store the latest section generation / update result."""
        self.latest_section_result = result

    def set_validation_result(self, result: Optional[ValidationResult]) -> None:
        """Store the latest section validation result and record it in validation history."""
        self.latest_validation_result = result
        if result is not None:
            self.validation_history.append(result)

    def get_latest_validation_result(self) -> Optional[ValidationResult]:
        """Retrieve the latest section validation result."""
        return self.latest_validation_result

    def set_progression_result(self, result: Optional[SectionProgressionResult]) -> None:
        """Store the latest section progression result and record it in progression history."""
        self.latest_progression_result = result
        if result is not None:
            self.progression_history.append(result)

    def get_latest_progression_result(self) -> Optional[SectionProgressionResult]:
        """Retrieve the latest section progression result."""
        return self.latest_progression_result

    def set_assembled_brd(self, document: Optional[str]) -> None:
        """Store the complete assembled BRD document."""
        self.assembled_brd = document

    def get_assembled_brd(self) -> Optional[str]:
        """Retrieve the complete assembled BRD document if available."""
        return self.assembled_brd

    def set_assembly_result(self, result: Optional[BRDAssemblyResult]) -> None:
        """Store the latest BRD assembly result and update assembled_brd."""
        self.latest_assembly_result = result
        if result is not None and result.assembled_document:
            self.assembled_brd = result.assembled_document

    def get_latest_assembly_result(self) -> Optional[BRDAssemblyResult]:
        """Retrieve the latest BRD assembly result."""
        return self.latest_assembly_result

    @property
    def is_assembled(self) -> bool:
        """True if a non-empty assembled BRD document is stored in state."""
        return bool(self.assembled_brd and self.assembled_brd.strip())

    def set_final_validation_result(self, result: Optional[FinalValidationResult]) -> None:
        """Store the latest final BRD validation result and append to history."""
        self.latest_final_validation_result = result
        if result is not None:
            self.final_validation_history.append(result)

    def get_latest_final_validation_result(self) -> Optional[FinalValidationResult]:
        """Retrieve the latest final BRD validation result."""
        return self.latest_final_validation_result

    def increment_final_validation_recovery_cycle(self) -> int:
        """Increment the count of executed final validation recovery cycles and return the new count."""
        self.final_validation_recovery_cycles += 1
        return self.final_validation_recovery_cycles

    def reset_final_validation_recovery_cycles(self) -> None:
        """Reset final validation recovery cycle counters."""
        self.final_validation_recovery_cycles = 0
        self.final_validation_recovery_exhausted = False

    def set_final_validation_recovery_exhausted(self, exhausted: bool = True) -> None:
        """Set whether final validation recovery limit was exhausted."""
        self.final_validation_recovery_exhausted = exhausted

    def clear_final_validation_history(self) -> None:
        """Clear final validation history and latest result."""
        self.final_validation_history.clear()
        self.latest_final_validation_result = None
        self.final_validation_recovery_cycles = 0
        self.final_validation_recovery_exhausted = False

    def get_remaining_sections(self, template_sections: Optional[Sequence[str]] = None) -> list[str]:
        """Return dynamically derived list of template sections not yet COMPLETED."""
        sections = (
            list(template_sections)
            if template_sections is not None
            else list(self.template_sections)
        )
        return [
            s for s in sections
            if self.get_section_status(s) != BRDSectionStatus.COMPLETED
        ]

    def clear_delegation(self) -> None:
        """Clear delegated execution state for the next workflow cycle."""
        self.delegated_tasks.clear()
        self.task_results.clear()
        self.delegation_result = None

    def get_section_status(self, section: str) -> Optional[BRDSectionStatus]:
        """Get the current progress status for a given section."""
        canonical = _match_section_name(section, self._available_section_keys())
        return self.section_progress.get(canonical)

    def update_section_status(
        self,
        section: str,
        status: BRDSectionStatus | str,
    ) -> None:
        """Update the progress status of a section."""
        resolved_status = BRDSectionStatus.from_string(status)
        canonical = _match_section_name(section, self._available_section_keys())
        self.section_progress[canonical] = resolved_status

    def set_current_section(
        self,
        section: Optional[str],
        auto_in_progress: bool = True,
    ) -> None:
        """Set the section the Agent is currently working on.

        If auto_in_progress is True and the section is currently NOT_STARTED,
        transitions its status to IN_PROGRESS.
        """
        if section is None:
            self.current_section = None
            return

        canonical = _match_section_name(section, self._available_section_keys())
        self.current_section = canonical

        if auto_in_progress:
            current_stat = self.section_progress.get(canonical)
            if current_stat is None or current_stat == BRDSectionStatus.NOT_STARTED:
                self.section_progress[canonical] = BRDSectionStatus.IN_PROGRESS

    def add_evidence(self, item: Any) -> None:
        """Add an evidence or working information item to working context."""
        self.evidence.append(item)

    def add_unresolved(self, item: str) -> None:
        """Record a missing or unresolved information item / gap."""
        if item not in self.unresolved_information:
            self.unresolved_information.append(item)

    def resolve_unresolved(self, item: str) -> bool:
        """Mark an unresolved item as resolved by removing it.

        Returns:
            bool: True if the item was found and removed, False otherwise.
        """
        if item in self.unresolved_information:
            self.unresolved_information.remove(item)
            return True
        return False

    def get_completed_sections(self) -> list[str]:
        """Return list of sections that are Completed."""
        return [
            s for s, stat in self.section_progress.items()
            if stat == BRDSectionStatus.COMPLETED
        ]

    def get_in_progress_sections(self) -> list[str]:
        """Return list of sections that are In Progress."""
        return [
            s for s, stat in self.section_progress.items()
            if stat == BRDSectionStatus.IN_PROGRESS
        ]

    def get_unstarted_sections(self) -> list[str]:
        """Return list of sections that are Not Started."""
        return [
            s for s, stat in self.section_progress.items()
            if stat == BRDSectionStatus.NOT_STARTED
        ]

    def get_needs_revision_sections(self) -> list[str]:
        """Return list of sections that Need Revision."""
        return [
            s for s, stat in self.section_progress.items()
            if stat == BRDSectionStatus.NEEDS_REVISION
        ]

    @property
    def is_complete(self) -> bool:
        """Check whether all template sections have reached Completed status."""
        if not self.template_sections:
            return False
        return all(
            self.section_progress.get(s) == BRDSectionStatus.COMPLETED
            for s in self.template_sections
        )

    @property
    def section_processing_complete(self) -> bool:
        """True when every required top-level template section has reached COMPLETED status."""
        return self.is_complete

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-serializable dictionary."""
        data = asdict(self)
        # Convert enums to string values
        data["section_progress"] = {
            k: v.value if isinstance(v, BRDSectionStatus) else str(v)
            for k, v in self.section_progress.items()
        }
        if self.delegation_result is not None:
            data["delegation_result"] = self.delegation_result.to_dict()
        if self.latest_action_result is not None:
            data["latest_action_result"] = self.latest_action_result.to_dict()
        if self.latest_evaluation_result is not None:
            data["latest_evaluation_result"] = self.latest_evaluation_result.to_dict()
        data["evaluation_history"] = [
            e.to_dict() if hasattr(e, "to_dict") else e for e in self.evaluation_history
        ]
        if self.latest_section_result is not None:
            data["latest_section_result"] = (
                self.latest_section_result.to_dict()
                if hasattr(self.latest_section_result, "to_dict")
                else self.latest_section_result
            )
        if self.latest_validation_result is not None:
            data["latest_validation_result"] = (
                self.latest_validation_result.to_dict()
                if hasattr(self.latest_validation_result, "to_dict")
                else self.latest_validation_result
            )
        data["validation_history"] = [
            v.to_dict() if hasattr(v, "to_dict") else v for v in self.validation_history
        ]
        if self.latest_progression_result is not None:
            data["latest_progression_result"] = (
                self.latest_progression_result.to_dict()
                if hasattr(self.latest_progression_result, "to_dict")
                else self.latest_progression_result
            )
        data["progression_history"] = [
            p.to_dict() if hasattr(p, "to_dict") else p for p in self.progression_history
        ]
        data["assembled_brd"] = self.assembled_brd
        if self.latest_assembly_result is not None:
            data["latest_assembly_result"] = (
                self.latest_assembly_result.to_dict()
                if hasattr(self.latest_assembly_result, "to_dict")
                else self.latest_assembly_result
            )
        if self.latest_final_validation_result is not None:
            data["latest_final_validation_result"] = (
                self.latest_final_validation_result.to_dict()
                if hasattr(self.latest_final_validation_result, "to_dict")
                else self.latest_final_validation_result
            )
        data["final_validation_history"] = [
            v.to_dict() if hasattr(v, "to_dict") else v for v in self.final_validation_history
        ]
        data["final_validation_recovery_cycles"] = self.final_validation_recovery_cycles
        data["final_validation_recovery_exhausted"] = self.final_validation_recovery_exhausted
        data["section_content"] = dict(self.section_content)
        data["rework_feedback"] = dict(self.rework_feedback)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BRDAgentState":
        """Deserialize a dictionary into a BRDAgentState instance."""
        raw_progress = data.get("section_progress", {})
        parsed_progress = {
            k: BRDSectionStatus.from_string(v)
            for k, v in raw_progress.items()
        }
        delegated_tasks = [
            DelegatedTask.from_dict(t) if isinstance(t, dict) else t
            for t in data.get("delegated_tasks", [])
        ]
        task_results = [
            TaskResult.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("task_results", [])
        ]
        del_res_raw = data.get("delegation_result")
        delegation_result = (
            DelegationResult.from_dict(del_res_raw)
            if isinstance(del_res_raw, dict)
            else del_res_raw
        )
        act_res_raw = data.get("latest_action_result")
        latest_action_result = (
            ActionResult.from_dict(act_res_raw)
            if isinstance(act_res_raw, dict)
            else act_res_raw
        )
        eval_res_raw = data.get("latest_evaluation_result")
        latest_evaluation_result = (
            EvaluationResult.from_dict(eval_res_raw)
            if isinstance(eval_res_raw, dict)
            else eval_res_raw
        )
        eval_hist_raw = data.get("evaluation_history", [])
        evaluation_history = [
            EvaluationResult.from_dict(e) if isinstance(e, dict) else e
            for e in eval_hist_raw
        ]
        sec_res_raw = data.get("latest_section_result")
        latest_section_result = sec_res_raw
        if isinstance(sec_res_raw, dict):
            try:
                from agents.brd.section_generation.agent import SectionGenerationResult
                latest_section_result = SectionGenerationResult.from_dict(sec_res_raw)
            except Exception:
                latest_section_result = sec_res_raw

        val_res_raw = data.get("latest_validation_result")
        latest_validation_result = (
            ValidationResult.from_dict(val_res_raw)
            if isinstance(val_res_raw, dict)
            else val_res_raw
        )
        val_hist_raw = data.get("validation_history", [])
        validation_history = [
            ValidationResult.from_dict(v) if isinstance(v, dict) else v
            for v in val_hist_raw
        ]
        prog_res_raw = data.get("latest_progression_result")
        latest_progression_result = (
            SectionProgressionResult.from_dict(prog_res_raw)
            if isinstance(prog_res_raw, dict)
            else prog_res_raw
        )
        prog_hist_raw = data.get("progression_history", [])
        progression_history = [
            SectionProgressionResult.from_dict(p) if isinstance(p, dict) else p
            for p in prog_hist_raw
        ]
        asmb_res_raw = data.get("latest_assembly_result")
        latest_assembly_result = (
            BRDAssemblyResult.from_dict(asmb_res_raw)
            if isinstance(asmb_res_raw, dict)
            else asmb_res_raw
        )
        assembled_brd = data.get("assembled_brd")
        fval_res_raw = data.get("latest_final_validation_result")
        latest_final_validation_result = (
            FinalValidationResult.from_dict(fval_res_raw)
            if isinstance(fval_res_raw, dict)
            else fval_res_raw
        )
        fval_hist_raw = data.get("final_validation_history", [])
        final_validation_history = [
            FinalValidationResult.from_dict(v) if isinstance(v, dict) else v
            for v in fval_hist_raw
        ]
        final_validation_recovery_cycles = int(data.get("final_validation_recovery_cycles", 0))
        final_validation_recovery_exhausted = bool(data.get("final_validation_recovery_exhausted", False))

        return cls(
            objective=data.get("objective", "Produce an evidence-grounded Business Requirements Document"),
            current_task=data.get("current_task"),
            template_sections=list(data.get("template_sections", [])),
            current_section=data.get("current_section"),
            section_progress=parsed_progress,
            evidence=list(data.get("evidence", [])),
            unresolved_information=list(data.get("unresolved_information", [])),
            metadata=dict(data.get("metadata", {})),
            delegated_tasks=delegated_tasks,
            task_results=task_results,
            delegation_result=delegation_result,
            latest_action_result=latest_action_result,
            latest_evaluation_result=latest_evaluation_result,
            evaluation_history=evaluation_history,
            section_content=dict(data.get("section_content", {})),
            rework_feedback=dict(data.get("rework_feedback", {})),
            latest_section_result=latest_section_result,
            latest_validation_result=latest_validation_result,
            validation_history=validation_history,
            latest_progression_result=latest_progression_result,
            progression_history=progression_history,
            assembled_brd=assembled_brd,
            latest_assembly_result=latest_assembly_result,
            latest_final_validation_result=latest_final_validation_result,
            final_validation_history=final_validation_history,
            final_validation_recovery_cycles=final_validation_recovery_cycles,
            final_validation_recovery_exhausted=final_validation_recovery_exhausted,
        )


class BRDDeepAgentState(DeepAgentState, total=False):
    """DeepAgents / LangGraph-compatible state representation for the BRD Lead Agent.

    Extends DeepAgentState with BRD domain working context.
    """

    objective: Optional[str]
    current_task: Optional[str]
    current_section: Optional[str]
    template_sections: Optional[list[str]]
    section_progress: Optional[dict[str, str]]
    evidence: Optional[list[Any]]
    unresolved_information: Optional[list[str]]
    delegated_tasks: Optional[list[dict[str, Any]]]
    task_results: Optional[list[dict[str, Any]]]
    delegation_result: Optional[dict[str, Any]]
    latest_action_result: Optional[dict[str, Any]]
    latest_evaluation_result: Optional[dict[str, Any]]
    evaluation_history: Optional[list[dict[str, Any]]]
    section_content: Optional[dict[str, str]]
    rework_feedback: Optional[dict[str, str]]
    latest_section_result: Optional[dict[str, Any]]
    latest_validation_result: Optional[dict[str, Any]]
    validation_history: Optional[list[dict[str, Any]]]
    latest_progression_result: Optional[dict[str, Any]]
    progression_history: Optional[list[dict[str, Any]]]
    assembled_brd: Optional[str]
    latest_assembly_result: Optional[dict[str, Any]]
    latest_final_validation_result: Optional[dict[str, Any]]
    final_validation_history: Optional[list[dict[str, Any]]]
    final_validation_recovery_cycles: Optional[int]
    final_validation_recovery_exhausted: Optional[bool]
