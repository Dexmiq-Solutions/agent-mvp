"""BRD Lead Agent implementation.

Establishes the BRD Lead Agent as the domain-specific Agent responsible for the
Business Requirements Document (BRD) generation objective.

Loads its foundational behavioral system instruction from the authoritative
external Markdown artifact (system_instruction.md).
Loads its required BRD document structure from the authoritative
external Markdown artifact (brd_template.md).
Maintains its working context and progress via BRDAgentState.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agents.brd.delegation import (
    DelegatedTask,
    DelegationResult,
    TaskResult,
    collect_task_results,
    decompose_objective,
    execute_subagent_task,
    execute_subagent_task_async,
)
from agents.brd.evaluation import (
    BRDEvaluationAgent,
    EvaluationContext,
    EvaluationOutcome,
    EvaluationResult,
)
from agents.brd.section_generation import (
    BRDSectionGenerationAgent,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
)
from agents.brd.section_validation import (
    BRDSectionValidationAgent,
    SectionValidationContext,
    ValidationCategory,
    ValidationFinding,
    ValidationOutcome,
    ValidationResult,
)
from agents.brd.template import (
    classify_requirement_item,
    extract_brd_sections,
    extract_section_requirements,
    extract_section_template,
    get_brd_template_path,
    is_administrative_section,
    is_benign_administrative_metadata_finding,
    is_documentation_quality_finding,
    is_metadata_or_role_item,
    is_substantive_business_clarification_item,
    is_substantive_requirement,
    is_tbd_value,
    load_brd_template,
)
from agents.brd.progression import (
    SectionProgressionResult,
    determine_next_section,
    get_remaining_sections,
    initialize_progression,
    is_section_processing_complete,
    progress_to_next_section,
)
from agents.brd.assembly import (
    BRDAssemblyResult,
    assemble_brd_document,
)
from agents.brd.final_validation import (
    BRDFinalValidationAgent,
    FinalValidationCategory,
    FinalValidationContext,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
    FinalValidationSeverity,
)
from agents.brd.recovery import (
    FinalValidationRecoveryResult,
    MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    format_section_rework_guidance,
    resolve_affected_sections,
)
from agents.brd.state import BRDAgentState, BRDSectionStatus
from deepagents import create_deep_agent
from agents.brd.config import AgentConfig, create_agent_model
from agents.brd.context import (
    ActionResult,
    ActionSource,
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
    reset_current_agent_context,
    set_current_agent_context,
)
from observability import (
    LLMExecutionSummary,
    LLMTelemetryTracker,
    execute_with_rate_limit_retry_async,
    extract_rate_limit_info,
    get_current_telemetry_tracker,
    reset_current_telemetry_tracker,
    scoped_telemetry_context,
    set_current_telemetry_tracker,
)
from observability.logging import (
    TraceActor,
    format_trace_event,
    get_logger,
    log_trace_event,
)
from services.rag_service import RAGService
from tools import get_default_tools
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import create_search_project_knowledge_tool, search_project_knowledge
from deepagents.backends.protocol import BackendProtocol
from langgraph.store.base import BaseStore
from agents.brd.memory import (
    DEFAULT_PROJECT_MEMORY_FILE,
    ProjectMemoryStoreBackend,
    create_project_namespace_factory,
    create_readonly_memory_middleware,
    create_readonly_memory_permissions,
    get_default_memory_store,
    normalize_memory_path,
)

logger = get_logger(__name__)

SYSTEM_INSTRUCTION_FILE = "system_instruction.md"
BRD_TEMPLATE_FILE = "brd_template.md"


def get_system_instruction_path() -> Path:
    """Resolve the absolute path to the BRD system instruction Markdown file."""
    return Path(__file__).resolve().parent / SYSTEM_INSTRUCTION_FILE


def load_system_instruction() -> str:
    """Load the authoritative BRD Lead Agent system instruction from Markdown.

    Returns:
        str: Content of the system instruction.

    Raises:
        FileNotFoundError: If the system instruction Markdown file does not exist.
        ValueError: If the system instruction Markdown file is empty.
    """
    instruction_path = get_system_instruction_path()
    if not instruction_path.is_file():
        raise FileNotFoundError(
            f"Required BRD Lead Agent system instruction file not found: {instruction_path}"
        )

    try:
        content = instruction_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read BRD system instruction from %s: %s", instruction_path, exc)
        raise

    if not content:
        raise ValueError(
            f"BRD Lead Agent system instruction file is empty: {instruction_path}"
        )

    return content




class _SystemInstructionDescriptor:
    """Descriptor providing access to system instruction on both instances and the class."""

    def __get__(self, instance: Optional["BRDLeadAgent"], owner: Optional[type] = None) -> str:
        if instance is not None and hasattr(instance, "_system_instruction"):
            return instance._system_instruction
        return load_system_instruction()


class _BRDTemplateDescriptor:
    """Descriptor providing access to BRD template on both instances and the class."""

    def __get__(self, instance: Optional["BRDLeadAgent"], owner: Optional[type] = None) -> str:
        if instance is not None and hasattr(instance, "_template"):
            return instance._template
        return load_brd_template()


class ActionType(str, Enum):
    """Action type decided by the BRD Lead Agent for objective execution."""

    RAG = "rag"
    DIRECT_WORK = "direct_work"
    DELEGATION = "delegation"


@dataclass
class ActionDecision:
    """Action decision formulated by the BRD Lead Agent.

    Attributes:
        action_type: ActionType (RAG, DIRECT_WORK, DELEGATION)
        query: Specific search query if action_type is RAG
        direct_work_objective: Objective to focus on if action_type is DIRECT_WORK
        direct_work_instructions: Specific analytical instructions for DIRECT_WORK
        delegated_tasks: List of subtasks if action_type is DELEGATION
        reasoning: Rationale behind selecting this action
        metadata: Extensible metadata dictionary
    """

    action_type: ActionType
    query: Optional[str] = None
    direct_work_objective: Optional[str] = None
    direct_work_instructions: Optional[str] = None
    delegated_tasks: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "query": self.query,
            "direct_work_objective": self.direct_work_objective,
            "direct_work_instructions": self.direct_work_instructions,
            "delegated_tasks": list(self.delegated_tasks),
            "reasoning": self.reasoning,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionDecision:
        action_type_raw = data.get("action_type", ActionType.DIRECT_WORK.value)
        action_type = ActionType(action_type_raw) if isinstance(action_type_raw, str) else action_type_raw
        return cls(
            action_type=action_type,
            query=data.get("query"),
            direct_work_objective=data.get("direct_work_objective"),
            direct_work_instructions=data.get("direct_work_instructions"),
            delegated_tasks=data.get("delegated_tasks", []),
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {}),
        )


class WorkflowDecision(str, Enum):
    """Workflow decision determined by the BRD Lead Agent following evaluation."""

    PROCEED_TO_SECTION_GENERATION = "proceed_to_section_generation"
    RAG = "rag"
    ASK_USER = "ask_user"


class GapResolutionAction(str, Enum):
    """Gap resolution path decided by Lead Agent after interpreting evaluation findings."""

    PROCEED_TO_SECTION_GENERATION = "proceed_to_section_generation"
    RAG = "rag"
    ASK_USER = "ask_user"


@dataclass
class GapResolutionDecision:
    """Decision formulated by the Lead Agent upon interpreting an EvaluationResult.

    Attributes:
        action: GapResolutionAction (PROCEED_TO_SECTION_GENERATION, RAG, ASK_USER)
        query: Focused knowledge search query if action is RAG
        clarification_question: Specific targeted question if action is ASK_USER
        reasoning: Rationale for gap resolution strategy
        identified_gaps: Summary of identified gaps
        metadata: Extensible metadata dictionary
    """

    action: GapResolutionAction
    query: Optional[str] = None
    clarification_question: Optional[str] = None
    reasoning: str = ""
    identified_gaps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decision(self) -> WorkflowDecision:
        """Backward compatibility with WorkflowDecision."""
        if self.action == GapResolutionAction.PROCEED_TO_SECTION_GENERATION:
            return WorkflowDecision.PROCEED_TO_SECTION_GENERATION
        elif self.action == GapResolutionAction.RAG:
            return WorkflowDecision.RAG
        else:
            return WorkflowDecision.ASK_USER

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "query": self.query,
            "clarification_question": self.clarification_question,
            "reasoning": self.reasoning,
            "identified_gaps": list(self.identified_gaps),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GapResolutionDecision:
        action_raw = data.get("action", GapResolutionAction.PROCEED_TO_SECTION_GENERATION.value)
        action = GapResolutionAction(action_raw) if isinstance(action_raw, str) else action_raw
        return cls(
            action=action,
            query=data.get("query"),
            clarification_question=data.get("clarification_question"),
            reasoning=data.get("reasoning", ""),
            identified_gaps=data.get("identified_gaps", []),
            metadata=data.get("metadata", {}),
        )


class SectionFailureCategory(str, Enum):
    """Categorical classification of unresolved section validation failure."""

    EVIDENCE_GAP = "evidence_gap"
    GENERATION_QUALITY = "generation_quality"
    TEMPLATE_CONFIGURATION = "template_configuration"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass
class SectionReworkStrategy:
    """Strategy formulated by the Lead Agent upon interpreting section validation findings."""

    section_name: str
    requires_rework: bool
    rework_guidance: str = ""
    specific_adjustments: list[str] = field(default_factory=list)
    reasoning: str = ""
    requires_retrieval: bool = False
    retrieval_query: Optional[str] = None
    missing_evidence_items: list[str] = field(default_factory=list)
    failure_category: Optional[SectionFailureCategory | str] = None
    clarification_question: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cat_val = (
            self.failure_category.value
            if isinstance(self.failure_category, SectionFailureCategory)
            else self.failure_category
        )
        return {
            "section_name": self.section_name,
            "requires_rework": self.requires_rework,
            "rework_guidance": self.rework_guidance,
            "specific_adjustments": list(self.specific_adjustments),
            "reasoning": self.reasoning,
            "requires_retrieval": self.requires_retrieval,
            "retrieval_query": self.retrieval_query,
            "missing_evidence_items": list(self.missing_evidence_items),
            "failure_category": cat_val,
            "clarification_question": self.clarification_question,
            "metadata": dict(self.metadata),
        }


@dataclass
class FinalValidationStrategy:
    """Recovery strategy formulated by the Lead Agent upon interpreting complete BRD validation findings."""

    requires_recovery: bool
    affected_sections: list[str] = field(default_factory=list)
    section_guidance: dict[str, str] = field(default_factory=dict)
    overall_strategy: str = ""
    cycle: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_recovery": self.requires_recovery,
            "affected_sections": list(self.affected_sections),
            "section_guidance": dict(self.section_guidance),
            "overall_strategy": self.overall_strategy,
            "cycle": self.cycle,
            "metadata": dict(self.metadata),
        }


def _build_project_context_prompt_block(
    context: Optional[AgentContext] = None,
    state_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Format non-intrusive explicit project context for prompt grounding (DEF-004)."""
    meta = state_metadata or {}
    proj_name = getattr(context, "project_name", None) or meta.get("project_name")
    proj_desc = getattr(context, "project_description", None) or meta.get("project_description")
    avail_docs = getattr(context, "available_documents", None) or meta.get("available_documents", [])

    lines = []
    if proj_name:
        lines.append(f"Project Name: {proj_name}")
    if proj_desc:
        lines.append(f"Project Description: {proj_desc}")
    if avail_docs:
        docs_str = ", ".join(avail_docs) if isinstance(avail_docs, (list, tuple)) else str(avail_docs)
        lines.append(f"Available Project Sources (indexed in RAG): {docs_str}")

    if not lines:
        return ""
    return "## Project Context\n" + "\n".join(lines) + "\n\n"


class BRDLeadAgent:
    """Domain-specific Agent responsible for accomplishing the BRD generation objective.

    The BRD Lead Agent is the primary agent for the BRD generation use case.
    It operates across three foundational pillars:
    1. System Instruction (system_instruction.md): Defines how the Agent behaves.
    2. BRD Template (brd_template.md): Defines what the Agent must produce.
    3. Agent State (BRDAgentState): Defines what the Agent currently knows and remembers.

    It executes directly within the DeepAgents execution harness,
    operating within application-provided project boundaries and utilizing existing
    agent tools (including RAG knowledge retrieval).
    """

    agent_name: str = "BRDLeadAgent"
    system_instruction: str = _SystemInstructionDescriptor()  # type: ignore[assignment]
    default_system_prompt: str = _SystemInstructionDescriptor()  # type: ignore[assignment]
    brd_template: str = _BRDTemplateDescriptor()  # type: ignore[assignment]
    template: str = _BRDTemplateDescriptor()  # type: ignore[assignment]

    def __init__(
        self,
        runtime: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        model: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        state: Optional[BRDAgentState] = None,
        rag_service: Optional[RAGService] = None,
        enable_rag: bool = True,
        evaluator: Optional[BRDEvaluationAgent] = None,
        section_generator: Optional[BRDSectionGenerationAgent] = None,
        section_validator: Optional[BRDSectionValidationAgent] = None,
        final_validator: Optional[BRDFinalValidationAgent] = None,
        store: Optional[BaseStore] = None,
        project_id: Optional[str] = None,
        memory_sources: Optional[Sequence[str]] = None,
        enable_memory: bool = True,
        checkpointer: Optional[Any] = None,
    ) -> None:
        """Initialize the BRD Lead Agent.

        Args:
            runtime: Deprecated parameter retained for backward compatibility.
            config: Optional AgentConfig instance.
            model: Optional pre-configured BaseChatModel.
            tools: Optional sequence of tools. Defaults to get_default_tools().
            system_instruction: Optional system instruction override. If omitted, loaded from
                system_instruction.md.
            system_prompt: Deprecated alias for system_instruction for backward compatibility.
            template: Optional BRD template override. If omitted, loaded from brd_template.md.
            state: Optional BRDAgentState working state instance. If omitted, initialized
                from the template structure.
            rag_service: Optional pre-configured RAGService instance to inject for knowledge retrieval.
            enable_rag: Whether to equip the search_project_knowledge tool when constructing runtime.
                Defaults to True.
            evaluator: Optional pre-configured BRDEvaluationAgent instance. If omitted, instantiated
                on demand using the active model.
            section_generator: Optional pre-configured BRDSectionGenerationAgent instance. If omitted,
                instantiated on demand using the active model.
            section_validator: Optional pre-configured BRDSectionValidationAgent instance. If omitted,
                instantiated on demand using the active model.
            final_validator: Optional pre-configured BRDFinalValidationAgent instance. If omitted,
                instantiated on demand using the active model.
            store: Optional LangGraph BaseStore instance for long-term memory.
            project_id: Optional application-supplied project identifier for memory scoping.
            memory_sources: Optional sequence of memory file paths to preload (defaults to ['/memory/project_context.md']).
            enable_memory: Whether to equip long-term project memory on the Lead Agent (defaults to True).
            checkpointer: Optional checkpointer for state persistence.
        """
        if system_instruction is not None:
            resolved_instruction = system_instruction
        elif system_prompt is not None:
            resolved_instruction = system_prompt
        else:
            resolved_instruction = load_system_instruction()
        self._system_instruction = resolved_instruction
        self._evaluator = evaluator
        self._section_generator = section_generator
        self._section_validator = section_validator
        self._final_validator = final_validator

        resolved_template = template or load_brd_template()
        self._template = resolved_template

        # Initialize working state from template structure if not injected
        if state is not None:
            self._state = state
        else:
            self._state = BRDAgentState.initialize_from_template(sections=self.sections)

        # Configure project identity and long-term memory
        self._enable_memory = enable_memory
        self._project_id = project_id
        if project_id and "project_id" not in self._state.metadata:
            self._state.metadata["project_id"] = project_id

        if enable_memory:
            self._store = store if store is not None else get_default_memory_store()
            self._memory_sources = [
                normalize_memory_path(p) for p in (memory_sources or [DEFAULT_PROJECT_MEMORY_FILE])
            ]
            ns_factory = create_project_namespace_factory(
                bound_project_id=self._project_id,
                agent=self,
            )
            self._memory_backend = ProjectMemoryStoreBackend(
                namespace=ns_factory,
                store=self._store,
            )
            self._memory_permissions = create_readonly_memory_permissions()
            self._memory_middleware = create_readonly_memory_middleware(
                backend=self._memory_backend,
                sources=self._memory_sources,
            )
        else:
            self._store = None
            self._memory_sources = []
            self._memory_backend = None
            self._memory_permissions = None
            self._memory_middleware = None

        self._config = config or AgentConfig.from_settings()
        self._model = model or create_agent_model(self._config)

        if tools is not None:
            self._tools = list(tools)
        elif not enable_rag:
            self._tools = [echo_diagnostic_tool]
        elif rag_service is not None:
            self._tools = [
                echo_diagnostic_tool,
                create_search_project_knowledge_tool(rag_service=rag_service),
            ]
        else:
            self._tools = get_default_tools()

        self._checkpointer = checkpointer

        deepagent_kwargs: dict[str, Any] = {}
        if self._enable_memory:
            deepagent_kwargs["store"] = self._store
            deepagent_kwargs["backend"] = self._memory_backend
            deepagent_kwargs["memory"] = self._memory_sources
            deepagent_kwargs["permissions"] = self._memory_permissions
            deepagent_kwargs["middleware"] = [self._memory_middleware]
        if self._checkpointer is not None:
            deepagent_kwargs["checkpointer"] = self._checkpointer

        self._graph = create_deep_agent(
            model=self._model,
            tools=self._tools,
            system_prompt=resolved_instruction,
            **deepagent_kwargs,
        )
        self._telemetry_tracker: Optional[LLMTelemetryTracker] = None

    @property
    def telemetry_tracker(self) -> Optional[LLMTelemetryTracker]:
        """Return the current LLMTelemetryTracker for the active run."""
        return getattr(self, "_telemetry_tracker", None)

    @property
    def has_rag_capability(self) -> bool:
        """Return True if the search_project_knowledge tool is equipped on this agent."""
        return any(
            getattr(tool, "name", "") == "search_project_knowledge"
            for tool in self.tools
        )

    @property
    def has_memory_capability(self) -> bool:
        """Return True if long-term memory is enabled and equipped on this agent."""
        return self._enable_memory and self._store is not None

    @property
    def store(self) -> Optional[BaseStore]:
        """Return the attached LangGraph BaseStore instance for long-term memory."""
        return self._store

    @property
    def memory_backend(self) -> Optional[BackendProtocol]:
        """Return the attached DeepAgents StoreBackend adapter."""
        return self._memory_backend

    @property
    def project_id(self) -> Optional[str]:
        """Return the active project identifier for this agent."""
        return self._project_id or self._state.metadata.get("project_id")

    @property
    def memory_sources(self) -> list[str]:
        """Return the list of configured memory source paths."""
        return list(self._memory_sources)

    @property
    def config(self) -> AgentConfig:
        """Return the active Agent configuration."""
        return self._config

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools."""
        return self._tools

    @property
    def graph(self) -> Any:
        """Return the compiled DeepAgents graph."""
        return self._graph

    @property
    def sections(self) -> list[str]:
        """Return ordered required BRD sections dynamically extracted from the template."""
        if hasattr(self, "_state") and self._state and self._state.template_sections:
            return list(self._state.template_sections)
        return extract_brd_sections(self._template)

    @property
    def state(self) -> BRDAgentState:
        """Return the active BRDAgentState working context."""
        return self._state

    @property
    def evaluator(self) -> BRDEvaluationAgent:
        """Return the attached Evaluation Sub-Agent capability."""
        if self._evaluator is None:
            self._evaluator = BRDEvaluationAgent(model=self.model)
        return self._evaluator

    @property
    def section_generator(self) -> BRDSectionGenerationAgent:
        """Return the attached Section Generation Sub-Agent capability."""
        if self._section_generator is None:
            self._section_generator = BRDSectionGenerationAgent(model=self.model)
        return self._section_generator

    @property
    def section_validator(self) -> BRDSectionValidationAgent:
        """Return the attached Section Validation Sub-Agent capability."""
        if self._section_validator is None:
            self._section_validator = BRDSectionValidationAgent(model=self.model)
        return self._section_validator

    @property
    def final_validator(self) -> BRDFinalValidationAgent:
        """Return the attached Final Validation Sub-Agent capability."""
        if self._final_validator is None:
            self._final_validator = BRDFinalValidationAgent(model=self.model)
        return self._final_validator

    def reset_state(self) -> None:
        """Reset the Agent working state to initial unstarted state based on template sections."""
        self._state = BRDAgentState.initialize_from_template(sections=self.sections)

    @property
    def is_section_processing_complete(self) -> bool:
        """Check whether all template sections have completed validation."""
        return is_section_processing_complete(self._state, self.sections)

    @property
    def assembled_brd(self) -> Optional[str]:
        """Complete assembled BRD Markdown document if assembly has been performed."""
        return self._state.get_assembled_brd()

    @property
    def is_assembled(self) -> bool:
        """True if a complete assembled BRD document is available in state."""
        return self._state.is_assembled

    @property
    def latest_assembly_result(self) -> Optional[BRDAssemblyResult]:
        """Retrieve the latest BRD assembly result."""
        return self._state.get_latest_assembly_result()

    @property
    def latest_final_validation_result(self) -> Optional[FinalValidationResult]:
        """Retrieve the latest final BRD validation result."""
        return self._state.get_latest_final_validation_result()

    @property
    def final_validation_recovery_cycles(self) -> int:
        """Current number of final-validation recovery cycles executed."""
        return self._state.final_validation_recovery_cycles

    @property
    def is_final_validation_recovery_exhausted(self) -> bool:
        """True if final-validation recovery was attempted and exhausted the maximum limit without success."""
        return self._state.final_validation_recovery_exhausted

    def get_remaining_sections(self) -> list[str]:
        """Return dynamically derived list of uncompleted template sections."""
        return get_remaining_sections(self._state, self.sections)

    def assemble_brd(
        self,
        context: Optional[AgentContext] = None,
    ) -> BRDAssemblyResult:
        """Deterministically assemble all completed BRD sections into the complete document.

        Workflow:
        1. Verifies that all required top-level template sections are COMPLETED.
        2. Retrieves validated content for each section from state in exact template order.
        3. Preserves section boundaries and headings without modifying semantic content.
        4. Combines sections into the final BRD document and stores it in state.assembled_brd.
        5. Returns BRDAssemblyResult.

        Args:
            context: Optional tenant AgentContext.

        Returns:
            BRDAssemblyResult: The complete assembled BRD document and execution metadata.

        Raises:
            ValueError: If section processing is incomplete, sections are missing content,
                or the template contains no valid sections.
        """
        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id")

        return assemble_brd_document(
            state=self._state,
            template_sections=self.sections,
            template_content=self._template,
            project_id=project_id,
        )

    async def assemble_brd_async(
        self,
        context: Optional[AgentContext] = None,
    ) -> BRDAssemblyResult:
        """Asynchronously assemble all completed BRD sections into the complete document."""
        return self.assemble_brd(context=context)

    def validate_final_brd(
        self,
        assembled_document: Optional[str] = None,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
        auto_recover: bool = False,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    ) -> FinalValidationResult:
        """Perform document-level final validation on the assembled BRD.

        Evaluates the complete assembled document across cross-section consistency,
        requirement conflicts, grounding, completeness, terminology, and overall coherence.

        Args:
            assembled_document: Optional assembled document override. Defaults to state.assembled_brd.
            available_project_information: Optional project evidence override. Defaults to state evidence.
            template_structure: Optional template override. Defaults to self._template.
            context: Optional tenant AgentContext.
            auto_recover: If True and validation requires rework, automatically executes the recovery loop.
            max_recovery_cycles: Maximum recovery cycles if auto_recover is enabled (default 3).

        Returns:
            FinalValidationResult: Structured result with outcome, findings, and rework feedback.

        Raises:
            ValueError: If no assembled BRD document is available or template structure is missing.
        """
        doc = assembled_document if assembled_document is not None else self._state.get_assembled_brd()
        if not doc or not doc.strip():
            raise ValueError(
                "Cannot perform final validation: no assembled BRD document is available in Agent State."
            )

        tmpl = template_structure if template_structure is not None else self._template
        if not tmpl or not tmpl.strip():
            raise ValueError(
                "Cannot perform final validation: template structure is missing."
            )

        if available_project_information is not None:
            info_payload = available_project_information
        else:
            info_parts: list[Any] = []
            if self._state.latest_action_result and self._state.latest_action_result.content:
                info_parts.append({
                    "source": str(self._state.latest_action_result.source),
                    "content": self._state.latest_action_result.content,
                })
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.task_results:
                for tr in self._state.task_results:
                    info_parts.append({
                        "source": f"delegated_task:{tr.task_id}",
                        "content": tr.output,
                    })
            info_payload = info_parts if info_parts else "*(No prior project evidence collected)*"

        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id", "unknown")

        val_ctx = FinalValidationContext(
            assembled_document=doc,
            template_structure=tmpl,
            available_project_information=info_payload,
            section_names=self.sections,
            metadata={
                "project_id": project_id,
            },
        )

        logger.info(
            "BRD final validation started (project_id: %s, document_length: %d, sections: %d)",
            project_id,
            len(doc),
            len(self.sections),
        )

        result = self.final_validator.validate(val_ctx)
        self._state.set_final_validation_result(result)

        if result.is_valid:
            logger.info(
                "BRD final validation completed (project_id: %s, outcome: VALID)",
                project_id,
            )
        else:
            logger.warning(
                "BRD final validation requires rework (project_id: %s, findings: %d, affected_sections: %s)",
                project_id,
                len(result.findings),
                result.affected_sections,
            )

        if auto_recover and result.needs_rework:
            return self.recover_final_validation(
                initial_result=result,
                max_cycles=max_recovery_cycles,
                available_project_information=available_project_information,
                template_structure=template_structure,
                context=context,
            )

        return result

    async def validate_final_brd_async(
        self,
        assembled_document: Optional[str] = None,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
        auto_recover: bool = False,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    ) -> FinalValidationResult:
        """Asynchronously perform document-level final validation on the assembled BRD."""
        doc = assembled_document if assembled_document is not None else self._state.get_assembled_brd()
        if not doc or not doc.strip():
            raise ValueError(
                "Cannot perform final validation: no assembled BRD document is available in Agent State."
            )

        tmpl = template_structure if template_structure is not None else self._template
        if not tmpl or not tmpl.strip():
            raise ValueError(
                "Cannot perform final validation: template structure is missing."
            )

        if available_project_information is not None:
            info_payload = available_project_information
        else:
            info_parts: list[Any] = []
            if self._state.latest_action_result and self._state.latest_action_result.content:
                info_parts.append({
                    "source": str(self._state.latest_action_result.source),
                    "content": self._state.latest_action_result.content,
                })
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.task_results:
                for tr in self._state.task_results:
                    info_parts.append({
                        "source": f"delegated_task:{tr.task_id}",
                        "content": tr.output,
                    })
            info_payload = info_parts if info_parts else "*(No prior project evidence collected)*"

        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id", "unknown")

        val_ctx = FinalValidationContext(
            assembled_document=doc,
            template_structure=tmpl,
            available_project_information=info_payload,
            section_names=self.sections,
            metadata={
                "project_id": project_id,
            },
        )

        logger.info(
            "Async BRD final validation started (project_id: %s, document_length: %d, sections: %d)",
            project_id,
            len(doc),
            len(self.sections),
        )

        phase = "8_FINAL_RECOVERY" if self._state.final_validation_recovery_cycles > 0 else "7_FINAL_VALIDATION"
        with scoped_telemetry_context(
            component="BRDFinalValidationAgent",
            phase=phase,
            operation="validate_final_brd",
        ):
            result = await self.final_validator.validate_async(val_ctx)
        self._state.set_final_validation_result(result)

        if result.is_valid:
            logger.info(
                "Async BRD final validation completed (project_id: %s, outcome: VALID)",
                project_id,
            )
        else:
            logger.warning(
                "Async BRD final validation requires rework (project_id: %s, findings: %d, affected_sections: %s)",
                project_id,
                len(result.findings),
                result.affected_sections,
            )

        if auto_recover and result.needs_rework:
            return await self.recover_final_validation_async(
                initial_result=result,
                max_cycles=max_recovery_cycles,
                available_project_information=available_project_information,
                template_structure=template_structure,
                context=context,
            )

        return result

    def interpret_final_validation(
        self,
        final_validation_result: Optional[FinalValidationResult] = None,
        context: Optional[AgentContext] = None,
        assembled_content: Optional[str] = None,
    ) -> FinalValidationStrategy:
        """Intelligently interpret complete BRD final validation findings and formulate recovery strategy.

        Responsibilities:
        - Sub-agent evaluates complete BRD document and returns FinalValidationResult.
        - Lead Agent interprets the findings and determines the recovery strategy:
          * Resolves and prioritizes affected sections.
          * Generates section-specific rework guidance.
          * Formulates the overall recovery plan.
        - Application enforces the deterministic recovery limits (max 3 cycles) and executes the rework.
        """
        current_res = final_validation_result or self._state.get_latest_final_validation_result()
        if current_res is None:
            raise ValueError("No FinalValidationResult available to interpret.")

        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        if current_res.is_valid:
            strategy = FinalValidationStrategy(
                requires_recovery=False,
                overall_strategy="Complete BRD satisfies all cross-section consistency and completeness gates.",
                cycle=self._state.final_validation_recovery_cycles,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="FINAL_VALIDATION_INTERPRETED",
                requires_recovery=False,
            )
            return strategy

        candidate_affected = current_res.affected_sections
        resolved_affected = resolve_affected_sections(candidate_affected, self.sections)

        section_guidance: dict[str, str] = {}
        for sec in resolved_affected:
            section_guidance[sec] = format_section_rework_guidance(sec, current_res)

        strategy = FinalValidationStrategy(
            requires_recovery=True,
            affected_sections=resolved_affected,
            section_guidance=section_guidance,
            overall_strategy=f"Rework required across {len(resolved_affected)} sections ({', '.join(resolved_affected)}) to resolve cross-section findings.",
            cycle=self._state.final_validation_recovery_cycles,
        )

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="FINAL_VALIDATION_INTERPRETED",
            requires_recovery=True,
            affected_sections=resolved_affected,
        )
        return strategy

    async def interpret_final_validation_async(
        self,
        final_validation_result: Optional[FinalValidationResult] = None,
        context: Optional[AgentContext] = None,
        assembled_content: Optional[str] = None,
    ) -> FinalValidationStrategy:
        """Asynchronously interpret complete BRD final validation findings and formulate recovery strategy."""
        return self.interpret_final_validation(
            final_validation_result=final_validation_result,
            context=context,
            assembled_content=assembled_content,
        )

    def recover_final_validation(
        self,
        initial_result: Optional[FinalValidationResult] = None,
        max_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
        max_section_rework_attempts: int = 2,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> FinalValidationRecoveryResult:
        """Execute the bounded document-level final validation recovery loop.

        When Final Validation determines that the assembled BRD needs rework:
        1. Reads the final-validation findings and diagnoses.
        2. Identifies all affected BRD template sections.
        3. Sends each affected section through the existing Section Generation/Update
           capability with actionable validation feedback and findings.
        4. Runs existing Section Validation on the updated section (with existing retry mechanism).
        5. Once all affected sections are valid, deterministically reassembles the BRD.
        6. Runs Final Validation again on the newly assembled document.
        7. Repeats this recovery cycle up to max_cycles (bounded limit of 3).
        8. Stops safely if any section cannot be validated or max cycles are exhausted.

        Args:
            initial_result: Optional initial FinalValidationResult. If omitted, retrieved from state
                or computed via validate_final_brd().
            max_cycles: Maximum final validation recovery cycles (default 3).
            max_section_rework_attempts: Maximum retry attempts during section-level validation (default 2).
            available_project_information: Optional project evidence override.
            template_structure: Optional template structure override.
            context: Optional tenant AgentContext.

        Returns:
            FinalValidationRecoveryResult: The outcome of recovery with cycle count and status.
        """
        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id", "unknown")

        current_result = initial_result or self._state.get_latest_final_validation_result()
        if current_result is None:
            current_result = self.validate_final_brd(
                available_project_information=available_project_information,
                template_structure=template_structure,
                context=effective_ctx,
                auto_recover=False,
            )

        if current_result.is_valid:
            logger.info(
                "BRD final validation already VALID; no recovery needed (project_id: %s)",
                project_id,
            )
            self._state.final_validation_recovery_exhausted = False
            return FinalValidationRecoveryResult.from_validation_result(
                result=current_result,
                recovery_cycles=self._state.final_validation_recovery_cycles,
                exhausted=False,
                reworked_sections=[],
            )

        logger.info(
            "BRD final validation recovery started (project_id: %s, max_cycles: %d)",
            project_id,
            max_cycles,
        )

        all_reworked_sections: list[str] = []
        cycle = 0

        try:
            while cycle < max_cycles:
                cycle += 1
                self._state.increment_final_validation_recovery_cycle()

                # 1. Lead Agent interprets final validation findings and determines recovery strategy
                recovery_strategy = self.interpret_final_validation(
                    final_validation_result=current_result,
                    context=effective_ctx,
                )
                resolved_affected = recovery_strategy.affected_sections

                if not recovery_strategy.requires_recovery or not resolved_affected:
                    logger.warning(
                        "Final validation recovery identified no resolvable affected sections (project_id: %s, cycle: %d); stopping recovery safely",
                        project_id,
                        cycle,
                    )
                    break

                logger.info(
                    "Final validation recovery identified affected sections (project_id: %s, cycle: %d, affected_sections: %s)",
                    project_id,
                    cycle,
                    resolved_affected,
                )

                # 2. Rework each affected section sequentially
                cycle_sections_passed = True
                for sec in resolved_affected:
                    if sec not in all_reworked_sections:
                        all_reworked_sections.append(sec)

                    logger.info(
                        "Final validation section rework started (project_id: %s, section: %s, cycle: %d)",
                        project_id,
                        sec,
                        cycle,
                    )

                    sec_feedback = recovery_strategy.section_guidance.get(sec) or format_section_rework_guidance(sec, current_result)

                    try:
                        gen_res, sec_val_res = self.generate_and_validate_section(
                            section=sec,
                            rework_feedback=sec_feedback,
                            existing_content=self._state.get_section_content(sec),
                            context=effective_ctx,
                            max_rework_attempts=max_section_rework_attempts,
                            auto_progress=False,
                        )
                    except Exception as exc:
                        logger.error(
                            "Error during final validation recovery section rework for '%s' (project_id: %s, cycle: %d): %s",
                            sec,
                            project_id,
                            cycle,
                            exc,
                            exc_info=True,
                        )
                        cycle_sections_passed = False
                        break

                    logger.info(
                        "Final validation section rework completed (project_id: %s, section: %s, cycle: %d)",
                        project_id,
                        sec,
                        cycle,
                    )
                    logger.info(
                        "Final validation section validation completed (project_id: %s, section: %s, outcome: %s)",
                        project_id,
                        sec,
                        sec_val_res.outcome,
                    )

                    if not sec_val_res.is_valid:
                        logger.warning(
                            "Final validation recovery section validation failed for '%s' (project_id: %s, cycle: %d); stopping recovery safely",
                            sec,
                            project_id,
                            cycle,
                        )
                        cycle_sections_passed = False
                        break

                if not cycle_sections_passed:
                    # Affected section cannot safely be updated -> stop recovery attempt safely
                    break

                # 3. Deterministically reassemble the BRD
                try:
                    asmb_res = self.assemble_brd(context=effective_ctx)
                except Exception as exc:
                    logger.error(
                        "Error during BRD reassembly in final validation recovery (project_id: %s, cycle: %d): %s",
                        project_id,
                        cycle,
                        exc,
                        exc_info=True,
                    )
                    break

                # 4. Run Final Validation again on the newly assembled document
                try:
                    current_result = self.validate_final_brd(
                        assembled_document=asmb_res.assembled_document,
                        available_project_information=available_project_information,
                        template_structure=template_structure,
                        context=effective_ctx,
                        auto_recover=False,
                    )
                except Exception as exc:
                    logger.error(
                        "Error during BRD final re-validation (project_id: %s, cycle: %d): %s",
                        project_id,
                        cycle,
                        exc,
                        exc_info=True,
                    )
                    break

                logger.info(
                    "Final validation recovery cycle %d completed (project_id: %s, outcome: %s)",
                    cycle,
                    project_id,
                    current_result.outcome,
                )

                if current_result.is_valid:
                    logger.info(
                        "BRD final validation passed after recovery (project_id: %s, total_cycles: %d)",
                        project_id,
                        cycle,
                    )
                    self._state.final_validation_recovery_exhausted = False
                    return FinalValidationRecoveryResult.from_validation_result(
                        result=current_result,
                        recovery_cycles=self._state.final_validation_recovery_cycles,
                        exhausted=False,
                        reworked_sections=all_reworked_sections,
                    )
        except Exception as exc:
            logger.error(
                "Error during final validation recovery (project_id: %s, cycle: %d): %s",
                project_id,
                cycle,
                exc,
                exc_info=True,
            )

        # Loop completed without VALID outcome
        is_exhausted = (cycle >= max_cycles) and (not current_result.is_valid)
        if is_exhausted:
            self._state.final_validation_recovery_exhausted = True
            logger.warning(
                "BRD final validation recovery exhausted (project_id: %s, max_cycles: %d reached without VALID outcome)",
                project_id,
                max_cycles,
            )

        return FinalValidationRecoveryResult.from_validation_result(
            result=current_result,
            recovery_cycles=self._state.final_validation_recovery_cycles,
            exhausted=is_exhausted,
            reworked_sections=all_reworked_sections,
        )

    async def recover_final_validation_async(
        self,
        initial_result: Optional[FinalValidationResult] = None,
        max_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
        max_section_rework_attempts: int = 2,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> FinalValidationRecoveryResult:
        """Asynchronously execute the bounded document-level final validation recovery loop."""
        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id", "unknown")

        current_result = initial_result or self._state.get_latest_final_validation_result()
        if current_result is None:
            current_result = await self.validate_final_brd_async(
                available_project_information=available_project_information,
                template_structure=template_structure,
                context=effective_ctx,
                auto_recover=False,
            )

        if current_result.is_valid:
            logger.info(
                "BRD final validation already VALID; no recovery needed (project_id: %s)",
                project_id,
            )
            self._state.final_validation_recovery_exhausted = False
            return FinalValidationRecoveryResult.from_validation_result(
                result=current_result,
                recovery_cycles=self._state.final_validation_recovery_cycles,
                exhausted=False,
                reworked_sections=[],
            )

        logger.info(
            "Async BRD final validation recovery started (project_id: %s, max_cycles: %d)",
            project_id,
            max_cycles,
        )

        all_reworked_sections: list[str] = []
        cycle = 0

        try:
            while cycle < max_cycles:
                cycle += 1
                self._state.increment_final_validation_recovery_cycle()

                # 1. Lead Agent interprets final validation findings and determines recovery strategy
                recovery_strategy = await self.interpret_final_validation_async(
                    final_validation_result=current_result,
                    context=effective_ctx,
                )
                resolved_affected = recovery_strategy.affected_sections

                if not recovery_strategy.requires_recovery or not resolved_affected:
                    logger.warning(
                        "Async final validation recovery identified no resolvable affected sections (project_id: %s, cycle: %d); stopping recovery safely",
                        project_id,
                        cycle,
                    )
                    break

                logger.info(
                    "Final validation recovery identified affected sections (project_id: %s, cycle: %d, affected_sections: %s)",
                    project_id,
                    cycle,
                    resolved_affected,
                )

                cycle_sections_passed = True
                for sec in resolved_affected:
                    if sec not in all_reworked_sections:
                        all_reworked_sections.append(sec)

                    logger.info(
                        "Final validation section rework started (project_id: %s, section: %s, cycle: %d)",
                        project_id,
                        sec,
                        cycle,
                    )

                    sec_feedback = recovery_strategy.section_guidance.get(sec) or format_section_rework_guidance(sec, current_result)

                    try:
                        gen_res, sec_val_res = await self.generate_and_validate_section_async(
                            section=sec,
                            rework_feedback=sec_feedback,
                            existing_content=self._state.get_section_content(sec),
                            context=effective_ctx,
                            max_rework_attempts=max_section_rework_attempts,
                            auto_progress=False,
                        )
                    except Exception as exc:
                        logger.error(
                            "Error during async final validation recovery section rework for '%s' (project_id: %s, cycle: %d): %s",
                            sec,
                            project_id,
                            cycle,
                            exc,
                            exc_info=True,
                        )
                        cycle_sections_passed = False
                        break

                    logger.info(
                        "Final validation section rework completed (project_id: %s, section: %s, cycle: %d)",
                        project_id,
                        sec,
                        cycle,
                    )
                    logger.info(
                        "Final validation section validation completed (project_id: %s, section: %s, outcome: %s)",
                        project_id,
                        sec,
                        sec_val_res.outcome,
                    )

                    if not sec_val_res.is_valid:
                        logger.warning(
                            "Async final validation recovery section validation failed for '%s' (project_id: %s, cycle: %d); stopping recovery safely",
                            sec,
                            project_id,
                            cycle,
                        )
                        cycle_sections_passed = False
                        break

                if not cycle_sections_passed:
                    break

                try:
                    asmb_res = await self.assemble_brd_async(context=effective_ctx)
                except Exception as exc:
                    logger.error(
                        "Error during BRD reassembly in async final validation recovery (project_id: %s, cycle: %d): %s",
                        project_id,
                        cycle,
                        exc,
                        exc_info=True,
                    )
                    break

                try:
                    current_result = await self.validate_final_brd_async(
                        assembled_document=asmb_res.assembled_document,
                        available_project_information=available_project_information,
                        template_structure=template_structure,
                        context=effective_ctx,
                        auto_recover=False,
                    )
                except Exception as exc:
                    logger.error(
                        "Error during async BRD final re-validation (project_id: %s, cycle: %d): %s",
                        project_id,
                        cycle,
                        exc,
                        exc_info=True,
                    )
                    break

                logger.info(
                    "Final validation recovery cycle %d completed (project_id: %s, outcome: %s)",
                    cycle,
                    project_id,
                    current_result.outcome,
                )

                if current_result.is_valid:
                    logger.info(
                        "BRD final validation passed after recovery (project_id: %s, total_cycles: %d)",
                        project_id,
                        cycle,
                    )
                    self._state.final_validation_recovery_exhausted = False
                    return FinalValidationRecoveryResult.from_validation_result(
                        result=current_result,
                        recovery_cycles=self._state.final_validation_recovery_cycles,
                        exhausted=False,
                        reworked_sections=all_reworked_sections,
                    )
        except Exception as exc:
            logger.error(
                "Error during async final validation recovery (project_id: %s, cycle: %d): %s",
                project_id,
                cycle,
                exc,
                exc_info=True,
            )

        is_exhausted = (cycle >= max_cycles) and (not current_result.is_valid)
        if is_exhausted:
            self._state.final_validation_recovery_exhausted = True
            logger.warning(
                "BRD final validation recovery exhausted (project_id: %s, max_cycles: %d reached without VALID outcome)",
                project_id,
                max_cycles,
            )

        return FinalValidationRecoveryResult.from_validation_result(
            result=current_result,
            recovery_cycles=self._state.final_validation_recovery_cycles,
            exhausted=is_exhausted,
            reworked_sections=all_reworked_sections,
        )

    def validate_and_recover_final_brd(
        self,
        assembled_document: Optional[str] = None,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    ) -> FinalValidationRecoveryResult:
        """Validate the assembled BRD and automatically execute recovery if rework is needed."""
        val_res = self.validate_final_brd(
            assembled_document=assembled_document,
            available_project_information=available_project_information,
            template_structure=template_structure,
            context=context,
            auto_recover=False,
        )
        if val_res.is_valid:
            return FinalValidationRecoveryResult.from_validation_result(
                result=val_res,
                recovery_cycles=self._state.final_validation_recovery_cycles,
                exhausted=False,
            )
        return self.recover_final_validation(
            initial_result=val_res,
            max_cycles=max_recovery_cycles,
            available_project_information=available_project_information,
            template_structure=template_structure,
            context=context,
        )

    async def validate_and_recover_final_brd_async(
        self,
        assembled_document: Optional[str] = None,
        available_project_information: Optional[Sequence[Any] | str] = None,
        template_structure: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    ) -> FinalValidationRecoveryResult:
        """Asynchronously validate the assembled BRD and automatically execute recovery if rework is needed."""
        val_res = await self.validate_final_brd_async(
            assembled_document=assembled_document,
            available_project_information=available_project_information,
            template_structure=template_structure,
            context=context,
            auto_recover=False,
        )
        if val_res.is_valid:
            return FinalValidationRecoveryResult.from_validation_result(
                result=val_res,
                recovery_cycles=self._state.final_validation_recovery_cycles,
                exhausted=False,
            )
        return await self.recover_final_validation_async(
            initial_result=val_res,
            max_cycles=max_recovery_cycles,
            available_project_information=available_project_information,
            template_structure=template_structure,
            context=context,
        )


    def progress_section(
        self,
        section: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> SectionProgressionResult:
        """Deterministically progress from current completed section to the next template section.

        Workflow:
        1. Verifies current section is COMPLETED.
        2. Advances to the next section in authoritative template order.
        3. Marks next section IN_PROGRESS.
        4. If no more sections remain, marks section processing as complete.
        5. Records result in state.latest_progression_result and progression_history.

        Args:
            section: Optional section override. Defaults to state.current_section.
            context: Optional tenant AgentContext.

        Returns:
            SectionProgressionResult: Progression outcome detailing transition and completion state.

        Raises:
            ValueError: If current_section is missing, not in template, or not completed.
        """
        if section:
            self._state.set_current_section(section, auto_in_progress=False)

        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id")

        result = progress_to_next_section(
            state=self._state,
            template_sections=self.sections,
            project_id=project_id,
        )
        self._state.set_progression_result(result)
        return result

    def initialize_section_progression(
        self,
        context: Optional[AgentContext] = None,
    ) -> SectionProgressionResult:
        """Initialize or resume section progression by activating the first uncompleted section."""
        effective_ctx = context or AgentContext()
        project_id = effective_ctx.project_id or self._state.metadata.get("project_id")

        result = initialize_progression(
            state=self._state,
            template_sections=self.sections,
            project_id=project_id,
        )
        self._state.set_progression_result(result)
        return result

    def decompose_objective(
        self,
        objective: str,
        context: Optional[str] = None,
        tasks_hint: Optional[int] = None,
    ) -> list[DelegatedTask]:
        """Decompose an objective dynamically into an appropriate number of bounded tasks.

        Supports dynamic task decomposition (e.g. 2, 3, 5, etc.) based on objective requirements.

        Args:
            objective: Overarching objective or composite task to decompose.
            context: Optional scoped context or specifications.
            tasks_hint: Optional suggested number of tasks.

        Returns:
            list[DelegatedTask]: Dynamic list of bounded tasks.
        """
        return decompose_objective(
            objective=objective,
            context=context,
            model=self.model,
            tasks_hint=tasks_hint,
        )

    def delegate(
        self,
        tasks: Sequence[DelegatedTask | dict[str, Any]] | str,
        context: Optional[AgentContext] = None,
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        tools: Optional[Sequence[Any]] = None,
    ) -> AgentRunResponse:
        """Decompose an objective and delegate bounded tasks to temporary task-scoped sub-agents.

        Executes each delegated task within a temporary sub-agent execution, preserves task-level
        attribution, collects results into a unified DelegationResult, converts it to a common
        ActionResult, updates the Lead Agent working state, and returns an AgentRunResponse.

        Args:
            tasks: Either a sequence of DelegatedTask (or dicts) or a raw string objective to dynamically decompose.
            context: Optional AgentContext for tenant project isolation.
            objective: Optional overarching objective description.
            current_task: Optional immediate working task description to set on state.
            tools: Optional sequence of tools to make available to sub-agents (recursive delegation disallowed).

        Returns:
            AgentRunResponse: Normalized response containing the unified delegation output and action result.
        """
        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        if current_task is not None:
            self._state.current_task = current_task

        if isinstance(tasks, str):
            overall_obj = objective or tasks
            task_objs = self.decompose_objective(objective=tasks, context=self._state.current_task)
        else:
            overall_obj = objective or self._state.current_task or self._state.objective
            task_objs = [DelegatedTask.from_dict(t) if isinstance(t, dict) else t for t in tasks]

        self._state.set_delegated_tasks(task_objs)

        # Execute each delegated task sequentially via temporary task-scoped sub-agent
        subagent_results: list[TaskResult] = []
        for t in task_objs:
            res = execute_subagent_task(
                task=t,
                model=self.model,
                parent_context=effective_ctx,
                tools=tools,
            )
            subagent_results.append(res)
            self._state.add_task_result(res)

        # Collect task results into unified delegation result
        delegation_result = collect_task_results(
            task_results=subagent_results,
            overall_objective=overall_obj,
        )
        self._state.set_delegation_result(delegation_result)

        # Produce unified Action Result converging to common result boundary
        action_result = delegation_result.to_action_result(context=effective_ctx)
        self._state.set_action_result(action_result)

        return AgentRunResponse(
            output_text=delegation_result.content,
            context=effective_ctx,
            model=getattr(self.model, "model_name", str(self.model)),
            success=delegation_result.success,
            error=action_result.error,
            state=self._state,
            action_result=action_result,
        )

    async def delegate_async(
        self,
        tasks: Sequence[DelegatedTask | dict[str, Any]] | str,
        context: Optional[AgentContext] = None,
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        tools: Optional[Sequence[Any]] = None,
    ) -> AgentRunResponse:
        """Asynchronously delegate bounded tasks to temporary task-scoped sub-agents."""
        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        if current_task is not None:
            self._state.current_task = current_task

        if isinstance(tasks, str):
            overall_obj = objective or tasks
            task_objs = self.decompose_objective(objective=tasks, context=self._state.current_task)
        else:
            overall_obj = objective or self._state.current_task or self._state.objective
            task_objs = [DelegatedTask.from_dict(t) if isinstance(t, dict) else t for t in tasks]

        self._state.set_delegated_tasks(task_objs)

        subagent_results: list[TaskResult] = []
        for t in task_objs:
            res = await execute_subagent_task_async(
                task=t,
                model=self.model,
                parent_context=effective_ctx,
                tools=tools,
            )
            subagent_results.append(res)
            self._state.add_task_result(res)

        delegation_result = collect_task_results(
            task_results=subagent_results,
            overall_objective=overall_obj,
        )
        self._state.set_delegation_result(delegation_result)

        action_result = delegation_result.to_action_result(context=effective_ctx)
        self._state.set_action_result(action_result)

        return AgentRunResponse(
            output_text=delegation_result.content,
            context=effective_ctx,
            model=getattr(self.model, "model_name", str(self.model)),
            success=delegation_result.success,
            error=action_result.error,
            state=self._state,
            action_result=action_result,
        )

    def decide_action(
        self,
        objective: Optional[str] = None,
        context: Optional[AgentContext] = None,
        input_text: Optional[str] = None,
        section_name: Optional[str] = None,
    ) -> ActionDecision:
        """Intelligently analyze the objective and state to decide the next action.

        Responsibilities:
        - Lead Agent decides which action type is appropriate (RAG, DIRECT_WORK, DELEGATION).
        - Lead Agent formulates action parameters:
          * RAG: targeted search query
          * DIRECT_WORK: analytical reasoning objective and instructions
          * DELEGATION: decomposed subtasks
        """
        effective_obj = objective or input_text or (f"Section: {section_name}" if section_name else None) or self._state.current_task or self._state.objective or "Analyze project requirements and generate BRD"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        obj_lower = effective_obj.lower()

        # If LLM model is available, attempt LLM decision
        try:
            project_block = _build_project_context_prompt_block(effective_ctx, self._state.metadata)
            prompt = (
                f"You are the BRD Lead Agent. Analyze the current objective and decide the best action.\n"
                f"{project_block}"
                f"Objective: {effective_obj}\n"
                f"Section: {section_name or self._state.current_section or 'General'}\n"
                f"Available tools: RAG equipped={self.has_rag_capability}\n\n"
                f"Choose exactly one action_type:\n"
                f"- 'rag': Search project knowledge base for existing documents, specs, or background.\n"
                f"- 'direct_work': Perform direct analytical synthesis and requirements drafting.\n"
                f"- 'delegation': Decompose a complex objective into multiple delegated subtasks.\n\n"
                f"Respond in valid JSON with fields: 'action_type', 'query', 'direct_work_objective', 'direct_work_instructions', 'delegated_tasks', 'reasoning'."
            )
            msg = self._model.invoke([HumanMessage(content=prompt)])
            raw_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                act_type_str = data.get("action_type", "").lower().strip()
                if act_type_str == "rag" and self.has_rag_capability:
                    decision = ActionDecision(
                        action_type=ActionType.RAG,
                        query=data.get("query") or (data.get("rag_queries", [effective_obj])[0] if data.get("rag_queries") else effective_obj),
                        reasoning=data.get("reasoning", data.get("rationale", "RAG selected to retrieve relevant domain information.")),
                    )
                elif act_type_str == "delegation":
                    tasks = data.get("delegated_tasks") or [effective_obj]
                    decision = ActionDecision(
                        action_type=ActionType.DELEGATION,
                        delegated_tasks=tasks if isinstance(tasks, list) else [str(tasks)],
                        reasoning=data.get("reasoning", data.get("rationale", "Delegation selected for multi-task objective decomposition.")),
                    )
                else:
                    decision = ActionDecision(
                        action_type=ActionType.DIRECT_WORK,
                        direct_work_objective=data.get("direct_work_objective") or effective_obj,
                        direct_work_instructions=data.get("direct_work_instructions") or "Analyze and synthesize requirements.",
                        reasoning=data.get("reasoning", data.get("rationale", "Direct work selected for analytical drafting.")),
                    )
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.LEAD_AGENT,
                    event_name="ACTION_DECIDED",
                    action_type=decision.action_type.value,
                    reasoning=decision.reasoning,
                )
                return decision
        except Exception as exc:
            logger.debug("Lead Agent LLM action decision fell back to analytical reasoning: %s", exc)

        # Fallback heuristic when model does not return structured JSON
        if self.has_rag_capability and any(k in obj_lower for k in ["search", "find", "retrieve", "lookup", "knowledge", "rag", "docs", "security"]):
            decision = ActionDecision(
                action_type=ActionType.RAG,
                query=f"Retrieve specifications and requirements for: {effective_obj}",
                reasoning="RAG selected because objective requests information retrieval.",
            )
        elif any(k in obj_lower for k in ["decompose", "subtasks", "delegate", "parallel"]):
            decision = ActionDecision(
                action_type=ActionType.DELEGATION,
                delegated_tasks=[f"Analyze requirements for: {effective_obj}", f"Draft scope for: {effective_obj}"],
                reasoning="Delegation selected because objective requests decomposition into subtasks.",
            )
        else:
            decision = ActionDecision(
                action_type=ActionType.DIRECT_WORK,
                direct_work_objective=effective_obj,
                direct_work_instructions=f"Synthesize and analyze requirements directly for: {effective_obj}",
                reasoning="Direct work selected as default analytical path for requirement formulation.",
            )

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="ACTION_DECIDED",
            action_type=decision.action_type.value,
            reasoning=decision.reasoning,
        )
        return decision

    async def decide_action_async(
        self,
        objective: Optional[str] = None,
        context: Optional[AgentContext] = None,
        input_text: Optional[str] = None,
        section_name: Optional[str] = None,
    ) -> ActionDecision:
        """Asynchronously analyze the objective and state to decide the next action."""
        effective_obj = objective or input_text or (f"Section: {section_name}" if section_name else None) or self._state.current_task or self._state.objective or "Analyze project requirements and generate BRD"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        obj_lower = effective_obj.lower()

        try:
            project_block = _build_project_context_prompt_block(effective_ctx, self._state.metadata)
            prompt = (
                f"You are the BRD Lead Agent. Analyze the current objective and decide the best action.\n"
                f"{project_block}"
                f"Objective: {effective_obj}\n"
                f"Section: {section_name or self._state.current_section or 'General'}\n"
                f"Available tools: RAG equipped={self.has_rag_capability}\n\n"
                f"Choose exactly one action_type:\n"
                f"- 'rag': Search project knowledge base for existing documents, specs, or background.\n"
                f"- 'direct_work': Perform direct analytical synthesis and requirements drafting.\n"
                f"- 'delegation': Decompose a complex objective into multiple delegated subtasks.\n\n"
                f"Respond in valid JSON with fields: 'action_type', 'query', 'direct_work_objective', 'direct_work_instructions', 'delegated_tasks', 'reasoning'."
            )
            with scoped_telemetry_context(
                component="BRDLeadAgent",
                phase="ORCHESTRATION",
                operation="decide_action",
                section=section_name or self._state.current_section or "General",
            ):
                msg = await execute_with_rate_limit_retry_async(
                    lambda: self._model.ainvoke([HumanMessage(content=prompt)]),
                    operation_name="decide_action",
                )
            raw_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                act_type_str = data.get("action_type", "").lower().strip()
                if act_type_str == "rag" and self.has_rag_capability:
                    decision = ActionDecision(
                        action_type=ActionType.RAG,
                        query=data.get("query") or (data.get("rag_queries", [effective_obj])[0] if data.get("rag_queries") else effective_obj),
                        reasoning=data.get("reasoning", data.get("rationale", "RAG selected to retrieve relevant domain information.")),
                    )
                elif act_type_str == "delegation":
                    tasks = data.get("delegated_tasks") or [effective_obj]
                    decision = ActionDecision(
                        action_type=ActionType.DELEGATION,
                        delegated_tasks=tasks if isinstance(tasks, list) else [str(tasks)],
                        reasoning=data.get("reasoning", data.get("rationale", "Delegation selected for multi-task objective decomposition.")),
                    )
                else:
                    decision = ActionDecision(
                        action_type=ActionType.DIRECT_WORK,
                        direct_work_objective=data.get("direct_work_objective") or effective_obj,
                        direct_work_instructions=data.get("direct_work_instructions") or "Analyze and synthesize requirements.",
                        reasoning=data.get("reasoning", data.get("rationale", "Direct work selected for analytical drafting.")),
                    )
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.LEAD_AGENT,
                    event_name="ACTION_DECIDED",
                    action_type=decision.action_type.value,
                    reasoning=decision.reasoning,
                )
                return decision
        except Exception as exc:
            logger.debug("Lead Agent async LLM action decision fell back to analytical reasoning: %s", exc)

        return self.decide_action(objective=objective, context=context, input_text=input_text, section_name=section_name)

    def perform_direct_work(
        self,
        objective: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> ActionResult:
        """Perform direct analytical reasoning and synthesis without external tools or delegation.

        Lead Agent directly analyzes the requirements and records the outcome as an ActionResult.
        """
        effective_obj = objective or self._state.current_task or self._state.objective or "Direct analytical work"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="DIRECT_WORK_STARTED",
            objective=effective_obj,
        )

        prompt = f"Objective: {effective_obj}\nInstructions: {instructions or 'Analyze the objective thoroughly and provide structured requirements.'}"
        response = self._execute_graph(input_text=prompt, ctx=effective_ctx)
        action_res = ActionResult(
            source=ActionSource.DIRECT_WORK,
            content=response.output_text,
            success=response.success,
            context=effective_ctx,
            metadata={"agent_run_id": agent_run_id, "objective": effective_obj, "instructions": instructions},
        )
        self._state.set_action_result(action_res)
        self._state.add_agent_work({
            "source": ActionSource.DIRECT_WORK.value,
            "content": action_res.content,
            "objective": effective_obj,
        })

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="DIRECT_WORK_COMPLETED",
            success=action_res.success,
        )
        return action_res

    async def perform_direct_work_async(
        self,
        objective: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> ActionResult:
        """Asynchronously perform direct analytical reasoning and synthesis."""
        effective_obj = objective or self._state.current_task or self._state.objective or "Direct analytical work"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="DIRECT_WORK_STARTED",
            objective=effective_obj,
        )

        prompt = f"Objective: {effective_obj}\nInstructions: {instructions or 'Analyze the objective thoroughly and provide structured requirements.'}"
        response = await self._execute_graph_async(input_text=prompt, ctx=effective_ctx)
        action_res = ActionResult(
            source=ActionSource.DIRECT_WORK,
            content=response.output_text,
            success=response.success,
            context=effective_ctx,
            metadata={"agent_run_id": agent_run_id, "objective": effective_obj, "instructions": instructions},
        )
        self._state.set_action_result(action_res)
        self._state.add_agent_work({
            "source": ActionSource.DIRECT_WORK.value,
            "content": action_res.content,
            "objective": effective_obj,
        })

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="DIRECT_WORK_COMPLETED",
            success=action_res.success,
        )
        return action_res

    def execute(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
        current_task: Optional[str] = None,
        delegate: bool = False,
    ) -> AgentRunResponse:
        """Execute a synchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            current_task: Optional immediate task context override to associate with this execution.
            delegate: If True, execute via delegation and sub-agents.

        Returns:
            AgentRunResponse: Normalized response containing output text, artifacts, and state.
        """
        # Adopt or normalize incoming state if passed in request
        if isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

        # Update current task on working state if provided
        if current_task is not None:
            self._state.current_task = current_task

        # Preserve project context in state metadata
        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx is None:
            effective_ctx = AgentContext(project_id=self.project_id)
        elif not effective_ctx.project_id and self.project_id:
            effective_ctx = AgentContext(
                project_id=self.project_id,
                conversation_id=effective_ctx.conversation_id,
                user_id=effective_ctx.user_id,
                metadata=effective_ctx.metadata,
            )

        if effective_ctx and effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id
        if effective_ctx and effective_ctx.conversation_id:
            self._state.metadata["conversation_id"] = effective_ctx.conversation_id

        # Route to application-owned full BRD workflow if state is waiting for user
        if self._state.is_waiting_for_user:
            prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
            return self.run_workflow(
                request=prompt_text,
                context=effective_ctx,
                current_task=current_task,
            )

        # Route to delegation if requested
        should_delegate = delegate or (
            isinstance(request, AgentRunRequest)
            and bool(request.context.metadata.get("delegate", False))
        )
        if should_delegate:
            prompt_text = request.input_text if isinstance(request, AgentRunRequest) else request
            return self.delegate(
                tasks=prompt_text,
                context=effective_ctx,
                current_task=current_task,
            )

        prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
        response = self._execute_graph(input_text=prompt_text, ctx=effective_ctx)
        # Converge onto common Action Result boundary
        action_res = response.to_action_result()
        response.action_result = action_res
        self._state.set_action_result(action_res)
        response.state = self._state
        return response

    def _execute_graph(
        self,
        input_text: str,
        ctx: AgentContext,
    ) -> AgentRunResponse:
        start_time = time.perf_counter()
        token = set_current_agent_context(ctx)
        try:
            inputs = {"messages": [HumanMessage(content=input_text)]}
            exec_config: dict[str, Any] = {}
            if ctx.conversation_id:
                exec_config["configurable"] = {"thread_id": ctx.conversation_id}
            result = self._graph.invoke(inputs, config=exec_config if exec_config else None)
            return self._process_graph_result(result, ctx, start_time)
        finally:
            reset_current_agent_context(token)

    async def _execute_graph_async(
        self,
        input_text: str,
        ctx: AgentContext,
    ) -> AgentRunResponse:
        start_time = time.perf_counter()
        token = set_current_agent_context(ctx)
        try:
            inputs = {"messages": [HumanMessage(content=input_text)]}
            exec_config: dict[str, Any] = {}
            if ctx.conversation_id:
                exec_config["configurable"] = {"thread_id": ctx.conversation_id}
            result = await self._graph.ainvoke(inputs, config=exec_config if exec_config else None)
            return self._process_graph_result(result, ctx, start_time)
        finally:
            reset_current_agent_context(token)

    def _process_graph_result(
        self,
        result: dict[str, Any],
        context: AgentContext,
        start_time: float,
    ) -> AgentRunResponse:
        messages = result.get("messages", [])
        output_text = ""
        tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc)

        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", "")
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                output_text = "".join(text_parts)
            else:
                output_text = str(content)

        return AgentRunResponse(
            output_text=output_text,
            messages=messages,
            tool_calls=tool_calls,
            context=context,
            model=getattr(self._config, "model", ""),
            success=True,
        )

    async def execute_async(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
        current_task: Optional[str] = None,
        delegate: bool = False,
    ) -> AgentRunResponse:
        """Execute an asynchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            current_task: Optional immediate task context override to associate with this execution.
            delegate: If True, execute via delegation and sub-agents.

        Returns:
            AgentRunResponse: Normalized response containing output text, artifacts, and state.
        """
        if isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

        if current_task is not None:
            self._state.current_task = current_task

        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx is None:
            effective_ctx = AgentContext(project_id=self.project_id)
        elif not effective_ctx.project_id and self.project_id:
            effective_ctx = AgentContext(
                project_id=self.project_id,
                conversation_id=effective_ctx.conversation_id,
                user_id=effective_ctx.user_id,
                metadata=effective_ctx.metadata,
            )

        if effective_ctx and effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id
        if effective_ctx and effective_ctx.conversation_id:
            self._state.metadata["conversation_id"] = effective_ctx.conversation_id

        # Route to application-owned full BRD workflow if state is waiting for user
        if self._state.is_waiting_for_user:
            prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
            return await self.run_workflow_async(
                request=prompt_text,
                context=effective_ctx,
                current_task=current_task,
            )

        should_delegate = delegate or (
            isinstance(request, AgentRunRequest)
            and bool(request.context.metadata.get("delegate", False))
        )
        if should_delegate:
            prompt_text = request.input_text if isinstance(request, AgentRunRequest) else request
            return await self.delegate_async(
                tasks=prompt_text,
                context=effective_ctx,
                current_task=current_task,
            )

        prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
        response = await self._execute_graph_async(input_text=prompt_text, ctx=effective_ctx)
        action_res = response.to_action_result()
        response.action_result = action_res
        self._state.set_action_result(action_res)
        response.state = self._state
        return response

    async def stream_async(
        self,
        request: AgentRunRequest | str,
        context: Optional[AgentContext] = None,
        prior_messages: Optional[Sequence[Any]] = None,
        current_task: Optional[str] = None,
        initial_state: Optional[BRDAgentState] = None,
        consolidate_clarification: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the controlled BRD workflow asynchronously via streaming.

        The controlled BRD workflow is the sole user-facing execution path for BRD
        conversations. Every user message (initial request, follow-up, clarification
        answer, or resume) directly enters the controlled BRD workflow.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            prior_messages: Optional sequence of prior BaseMessage instances.
            current_task: Optional immediate task context override.
            initial_state: Optional pre-existing/restored BRDAgentState for conversation resumption.
            consolidate_clarification: Whether to consolidate unresolved business gaps until end of drafting pass.

        Yields:
            dict[str, Any]: Incremental content chunks and progress events from the workflow.
        """
        if initial_state is not None:
            self._state = initial_state
        elif isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

        if current_task is not None:
            self._state.current_task = current_task

        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx is None:
            effective_ctx = AgentContext(project_id=self.project_id)
        elif not effective_ctx.project_id and self.project_id:
            effective_ctx = AgentContext(
                project_id=self.project_id,
                conversation_id=effective_ctx.conversation_id,
                user_id=effective_ctx.user_id,
                project_name=effective_ctx.project_name,
                project_description=effective_ctx.project_description,
                available_documents=list(effective_ctx.available_documents),
                metadata=effective_ctx.metadata,
            )

        if effective_ctx and effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id
        if effective_ctx and effective_ctx.conversation_id:
            self._state.metadata["conversation_id"] = effective_ctx.conversation_id
        if effective_ctx and effective_ctx.project_name:
            self._state.metadata["project_name"] = effective_ctx.project_name
        if effective_ctx and effective_ctx.project_description:
            self._state.metadata["project_description"] = effective_ctx.project_description
        if effective_ctx and effective_ctx.available_documents:
            self._state.metadata["available_documents"] = list(effective_ctx.available_documents)

        prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
        async for event in self.stream_workflow_async(
            request=prompt_text,
            context=effective_ctx,
            current_task=current_task,
            initial_state=self._state,
            prior_messages=prior_messages,
            consolidate_clarification=consolidate_clarification,
        ):
            yield event

    def set_project_memory(
        self,
        key: str,
        content: str | bytes,
        project_id: Optional[str] = None,
    ) -> bool:
        """Application-controlled write of authoritative project memory.

        Persists content into the framework BaseStore under the project's namespace.
        The Lead Agent has read-only access to this memory during execution.

        Args:
            key: Memory path or identifier (e.g., 'project_context.md' or '/memory/project_context.md').
            content: Text or binary content to persist.
            project_id: Optional project identifier override. If omitted, uses agent project_id.

        Returns:
            bool: True if write succeeded, False otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            logger.warning("Cannot write project memory: memory capability is disabled on this agent")
            return False

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to write project-scoped memory")

        normalized_path = normalize_memory_path(key)
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content

        logger.info(
            "Writing authoritative project memory (project_id: %s, key: %s, size: %d bytes)",
            resolved_pid,
            normalized_path,
            len(content_bytes),
        )

        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            responses = self._memory_backend.upload_files([(normalized_path, content_bytes)])
            for resp in responses:
                if resp.error is not None:
                    logger.error(
                        "Failed to write project memory (project_id: %s, key: %s): %s",
                        resolved_pid,
                        resp.path,
                        resp.error,
                    )
                    return False
            logger.info(
                "Authoritative project memory write succeeded (project_id: %s, key: %s)",
                resolved_pid,
                normalized_path,
            )
            return True
        except Exception as exc:
            logger.error(
                "Error writing authoritative project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return False
        finally:
            reset_current_agent_context(tok)

    async def set_project_memory_async(
        self,
        key: str,
        content: str | bytes,
        project_id: Optional[str] = None,
    ) -> bool:
        """Asynchronous application-controlled write of authoritative project memory.

        Args:
            key: Memory path or identifier.
            content: Text or binary content to persist.
            project_id: Optional project identifier override.

        Returns:
            bool: True if write succeeded, False otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            logger.warning("Cannot write project memory: memory capability is disabled on this agent")
            return False

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to write project-scoped memory")

        normalized_path = normalize_memory_path(key)
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content

        logger.info(
            "Async writing authoritative project memory (project_id: %s, key: %s, size: %d bytes)",
            resolved_pid,
            normalized_path,
            len(content_bytes),
        )

        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            responses = await self._memory_backend.aupload_files([(normalized_path, content_bytes)])
            for resp in responses:
                if resp.error is not None:
                    logger.error(
                        "Async failed to write project memory (project_id: %s, key: %s): %s",
                        resolved_pid,
                        resp.path,
                        resp.error,
                    )
                    return False
            logger.info(
                "Async authoritative project memory write succeeded (project_id: %s, key: %s)",
                resolved_pid,
                normalized_path,
            )
            return True
        except Exception as exc:
            logger.error(
                "Async error writing authoritative project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return False
        finally:
            reset_current_agent_context(tok)

    def get_project_memory(
        self,
        key: str,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        """Application-controlled read of authoritative project memory.

        Args:
            key: Memory path or identifier.
            project_id: Optional project identifier override.

        Returns:
            Optional[str]: Memory content as string if found, None otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return None

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to read project-scoped memory")

        normalized_path = normalize_memory_path(key)
        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            responses = self._memory_backend.download_files([normalized_path])
            if responses and responses[0].content is not None:
                logger.debug(
                    "Read project memory (project_id: %s, key: %s, found: True)",
                    resolved_pid,
                    normalized_path,
                )
                return responses[0].content.decode("utf-8")
            logger.debug(
                "Read project memory (project_id: %s, key: %s, found: False)",
                resolved_pid,
                normalized_path,
            )
            return None
        except Exception as exc:
            logger.error(
                "Error reading project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return None
        finally:
            reset_current_agent_context(tok)

    async def get_project_memory_async(
        self,
        key: str,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        """Asynchronous application-controlled read of authoritative project memory.

        Args:
            key: Memory path or identifier.
            project_id: Optional project identifier override.

        Returns:
            Optional[str]: Memory content as string if found, None otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return None

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to read project-scoped memory")

        normalized_path = normalize_memory_path(key)
        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            responses = await self._memory_backend.adownload_files([normalized_path])
            if responses and responses[0].content is not None:
                logger.debug(
                    "Async read project memory (project_id: %s, key: %s, found: True)",
                    resolved_pid,
                    normalized_path,
                )
                return responses[0].content.decode("utf-8")
            logger.debug(
                "Async read project memory (project_id: %s, key: %s, found: False)",
                resolved_pid,
                normalized_path,
            )
            return None
        except Exception as exc:
            logger.error(
                "Async error reading project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return None
        finally:
            reset_current_agent_context(tok)

    def list_project_memory(
        self,
        pattern: str = "*",
        project_id: Optional[str] = None,
    ) -> list[str]:
        """List stored memory file paths for the specified project.

        Args:
            pattern: Glob matching pattern (defaults to '*').
            project_id: Optional project identifier override.

        Returns:
            list[str]: Matching memory file paths.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return []

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to list project-scoped memory")

        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            res = self._memory_backend.glob(pattern)
            return [m["path"] for m in res.matches]
        except Exception as exc:
            logger.error(
                "Error listing project memory (project_id: %s): %s",
                resolved_pid,
                exc,
                exc_info=True,
            )
            return []
        finally:
            reset_current_agent_context(tok)

    async def list_project_memory_async(
        self,
        pattern: str = "*",
        project_id: Optional[str] = None,
    ) -> list[str]:
        """Asynchronously list stored memory file paths for the specified project.

        Args:
            pattern: Glob matching pattern (defaults to '*').
            project_id: Optional project identifier override.

        Returns:
            list[str]: Matching memory file paths.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return []

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to list project-scoped memory")

        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            res = await self._memory_backend.aglob(pattern)
            return [m["path"] for m in res.matches]
        except Exception as exc:
            logger.error(
                "Async error listing project memory (project_id: %s): %s",
                resolved_pid,
                exc,
                exc_info=True,
            )
            return []
        finally:
            reset_current_agent_context(tok)

    def delete_project_memory(
        self,
        key: str,
        project_id: Optional[str] = None,
    ) -> bool:
        """Application-controlled deletion of project memory.

        Args:
            key: Memory path or identifier.
            project_id: Optional project identifier override.

        Returns:
            bool: True if deletion succeeded, False otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return False

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to delete project-scoped memory")

        normalized_path = normalize_memory_path(key)
        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            res = self._memory_backend.delete(normalized_path)
            return res.error is None
        except Exception as exc:
            logger.error(
                "Error deleting project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return False
        finally:
            reset_current_agent_context(tok)

    async def delete_project_memory_async(
        self,
        key: str,
        project_id: Optional[str] = None,
    ) -> bool:
        """Asynchronous application-controlled deletion of project memory.

        Args:
            key: Memory path or identifier.
            project_id: Optional project identifier override.

        Returns:
            bool: True if deletion succeeded, False otherwise.

        Raises:
            ValueError: If project_id is not specified and cannot be resolved.
        """
        if not self._enable_memory or self._memory_backend is None:
            return False

        resolved_pid = project_id or self.project_id
        if not resolved_pid:
            raise ValueError("project_id must be provided to delete project-scoped memory")

        normalized_path = normalize_memory_path(key)
        tok = set_current_agent_context(AgentContext(project_id=resolved_pid))
        try:
            res = await self._memory_backend.adelete(normalized_path)
            return res.error is None
        except Exception as exc:
            logger.error(
                "Async error deleting project memory (project_id: %s, key: %s): %s",
                resolved_pid,
                normalized_path,
                exc,
                exc_info=True,
            )
            return False
        finally:
            reset_current_agent_context(tok)

    def evaluate(
        self,
        action_result: Optional[ActionResult] = None,
        objective: Optional[str] = None,
        section: Optional[str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        context: Optional[AgentContext] = None,
        relevant_working_context: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate an ActionResult against current objective and section requirements.

        Workflow:
        ActionResult -> Evaluation Sub-Agent -> EvaluationResult -> BRD Lead Agent State

        Args:
            action_result: Optional specific ActionResult to evaluate. Defaults to
                self.state.latest_action_result.
            objective: Optional objective. Defaults to self.state.current_task or self.state.objective.
            section: Optional section. Defaults to self.state.current_section.
            section_requirements: Optional section requirements. If omitted, extracted
                from the authoritative template for current_section.
            context: Optional AgentContext for tenant boundaries.
            relevant_working_context: Optional additional working context.

        Returns:
            EvaluationResult: Structured evaluation with outcome, findings, and gaps.

        Raises:
            ValueError: If no ActionResult is provided or present in state.
        """
        target_result = action_result or self._state.latest_action_result
        if target_result is None:
            raise ValueError(
                "No ActionResult provided or available in state to evaluate."
            )

        cur_obj = objective or self._state.current_task or self._state.objective
        cur_sec = section or self._state.current_section or "Unspecified Section"

        if section_requirements is not None:
            if isinstance(section_requirements, str):
                sec_reqs = [section_requirements]
            else:
                sec_reqs = list(section_requirements)
        else:
            sec_reqs = extract_section_requirements(cur_sec, self._template)

        working_ctx = relevant_working_context or ""
        if not working_ctx and self._state.evidence:
            evidence_texts = [
                str(e.get("content", e) if isinstance(e, dict) else e)
                for e in self._state.evidence[-3:]
            ]
            working_ctx = "Prior Evidence Summary:\n" + "\n".join(evidence_texts)

        effective_ctx = context or target_result.context or AgentContext()
        eval_ctx = EvaluationContext(
            current_objective=cur_obj,
            current_section=cur_sec,
            section_requirements=sec_reqs,
            relevant_working_context=working_ctx,
            action_result=target_result,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
                "available_documents": getattr(effective_ctx, "available_documents", None) or self._state.metadata.get("available_documents"),
            },
        )

        eval_result = self.evaluator.evaluate(context=eval_ctx)
        self._state.set_evaluation_result(eval_result)
        return eval_result

    async def evaluate_async(
        self,
        action_result: Optional[ActionResult] = None,
        objective: Optional[str] = None,
        section: Optional[str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        context: Optional[AgentContext] = None,
        relevant_working_context: Optional[str] = None,
    ) -> EvaluationResult:
        """Asynchronously evaluate an ActionResult against current objective and section requirements."""
        target_result = action_result or self._state.latest_action_result
        if target_result is None:
            raise ValueError(
                "No ActionResult provided or available in state to evaluate."
            )

        cur_obj = objective or self._state.current_task or self._state.objective
        cur_sec = section or self._state.current_section or "Unspecified Section"

        if section_requirements is not None:
            if isinstance(section_requirements, str):
                sec_reqs = [section_requirements]
            else:
                sec_reqs = list(section_requirements)
        else:
            sec_reqs = extract_section_requirements(cur_sec, self._template)

        working_ctx = relevant_working_context or ""
        if not working_ctx and self._state.evidence:
            evidence_texts = [
                str(e.get("content", e) if isinstance(e, dict) else e)
                for e in self._state.evidence[-3:]
            ]
            working_ctx = "Prior Evidence Summary:\n" + "\n".join(evidence_texts)

        effective_ctx = context or target_result.context or AgentContext()
        eval_ctx = EvaluationContext(
            current_objective=cur_obj,
            current_section=cur_sec,
            section_requirements=sec_reqs,
            relevant_working_context=working_ctx,
            action_result=target_result,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
                "available_documents": getattr(effective_ctx, "available_documents", None) or self._state.metadata.get("available_documents"),
            },
        )

        eval_result = await self.evaluator.evaluate_async(context=eval_ctx)
        self._state.set_evaluation_result(eval_result)
        return eval_result

    def interpret_evaluation(
        self,
        evaluation_result: Optional[EvaluationResult] = None,
        objective: Optional[str] = None,
        context: Optional[AgentContext] = None,
        can_rag_resolve: Optional[bool] = None,
        eval_result: Optional[EvaluationResult] = None,
    ) -> GapResolutionDecision:
        """Intelligently interpret an EvaluationResult and decide the gap resolution strategy.

        Responsibilities:
        - Sub-agent evaluates and returns findings.
        - Lead Agent interprets the findings:
          * If SUFFICIENT: proceeds toward BRD section generation.
          * If INSUFFICIENT: decides whether gaps warrant further RAG retrieval or User Clarification,
            synthesizing the targeted search query or clarification question.
        """
        eval_res = evaluation_result or eval_result or self._state.latest_evaluation_result
        if eval_res is None:
            raise ValueError("No EvaluationResult provided or available in state to interpret.")

        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        if eval_res.is_sufficient:
            decision = GapResolutionDecision(
                action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
                reasoning="Evaluation outcome is SUFFICIENT. Proceeding toward BRD section generation.",
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="EVALUATION_INTERPRETED",
                outcome="SUFFICIENT",
                action=decision.action.value,
            )
            return decision

        all_gaps = list(eval_res.missing_information) + list(eval_res.unresolved_information)
        if not self.has_rag_capability:
            q_text = f"Clarification needed on: {', '.join(all_gaps[:3])}" if all_gaps else "Please provide additional project specifications."
            decision = GapResolutionDecision(
                action=GapResolutionAction.ASK_USER,
                clarification_question=q_text,
                reasoning="RAG capability is not available. Missing information requires direct user clarification.",
                identified_gaps=all_gaps,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="EVALUATION_INTERPRETED",
                outcome="INSUFFICIENT",
                action=decision.action.value,
                reasoning=decision.reasoning,
            )
            return decision

        if can_rag_resolve is not None:
            if can_rag_resolve:
                query = f"Retrieve specifications for: {', '.join(all_gaps[:3])}" if all_gaps else "Retrieve project requirements"
                decision = GapResolutionDecision(
                    action=GapResolutionAction.RAG,
                    query=query,
                    reasoning="Explicit RAG feasibility specified: RAG will retrieve missing evidence.",
                    identified_gaps=all_gaps,
                )
            else:
                q_text = f"Clarification needed on: {', '.join(all_gaps[:3])}" if all_gaps else "Please clarify missing project specifications."
                decision = GapResolutionDecision(
                    action=GapResolutionAction.ASK_USER,
                    clarification_question=q_text,
                    reasoning="Explicit RAG feasibility specified as False: User clarification required.",
                    identified_gaps=all_gaps,
                )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="EVALUATION_INTERPRETED",
                outcome="INSUFFICIENT",
                action=decision.action.value,
            )
            return decision

        # Lead Agent analytical evaluation of gaps
        user_centric_indicators = [
            "user preference", "user confirmation", "stakeholder sign-off",
            "budget approval", "confirm with user", "ask user",
            "client preference", "pricing decision", "timeline agreement",
            "sign-off", "approval", "budget", "pricing",
        ]
        has_user_gap = any(any(ind in g.lower() for ind in user_centric_indicators) for g in all_gaps)

        if has_user_gap:
            q_text = f"Clarification needed regarding: {', '.join(all_gaps[:3])}"
            decision = GapResolutionDecision(
                action=GapResolutionAction.ASK_USER,
                clarification_question=q_text,
                reasoning="Lead Agent determined that identified gaps require business sign-off or user decision.",
                identified_gaps=all_gaps,
            )
        else:
            rag_query = f"Retrieve requirements and specifications for: {', '.join(all_gaps[:3])}" if all_gaps else f"Retrieve project details for section {self._state.current_section or 'BRD'}"
            decision = GapResolutionDecision(
                action=GapResolutionAction.RAG,
                query=rag_query,
                reasoning="Lead Agent determined that identified gaps pertain to specifications discoverable via project documentation.",
                identified_gaps=all_gaps,
            )

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="EVALUATION_INTERPRETED",
            outcome="INSUFFICIENT",
            action=decision.action.value,
            reasoning=decision.reasoning,
        )
        return decision

    async def interpret_evaluation_async(
        self,
        evaluation_result: Optional[EvaluationResult] = None,
        objective: Optional[str] = None,
        context: Optional[AgentContext] = None,
        can_rag_resolve: Optional[bool] = None,
        eval_result: Optional[EvaluationResult] = None,
    ) -> GapResolutionDecision:
        """Asynchronously interpret an EvaluationResult and decide the gap resolution strategy."""
        return self.interpret_evaluation(
            evaluation_result=evaluation_result,
            objective=objective,
            context=context,
            can_rag_resolve=can_rag_resolve,
            eval_result=eval_result,
        )

    def decide_next_step(
        self,
        evaluation_result: Optional[EvaluationResult] = None,
        can_rag_resolve: Optional[bool] = None,
    ) -> WorkflowDecision:
        """Decide the next workflow action based on an EvaluationResult.

        Delegates to interpret_evaluation to preserve single source of intelligence.
        """
        gap_dec = self.interpret_evaluation(
            evaluation_result=evaluation_result,
            can_rag_resolve=can_rag_resolve,
        )
        return gap_dec.decision

    def _can_rag_reasonably_provide(
        self,
        missing_items: Sequence[str],
        unresolved_items: Sequence[str],
    ) -> bool:
        """Determine whether missing/unresolved information can reasonably exist in project knowledge base.

        Heuristic:
        - Items regarding external user decisions, business approvals, personal preferences, or budget
          agreements cannot be found in static project docs -> Ask User.
        - Items regarding requirements, specs, workflows, architecture, personas, or integrations
          typically exist in repository knowledge -> RAG.
        """
        all_gaps = list(missing_items) + list(unresolved_items)
        if not all_gaps:
            return False

        user_decision_indicators = [
            "user preference",
            "user confirmation",
            "stakeholder sign-off",
            "budget approval",
            "confirm with user",
            "ask user",
            "client preference",
            "pricing decision",
            "timeline agreement",
        ]

        for gap in all_gaps:
            gap_lower = gap.lower()
            for indicator in user_decision_indicators:
                if indicator in gap_lower:
                    return False

        return True

    def retry_with_rag(
        self,
        query: Optional[str] = None,
        context: Optional[AgentContext] = None,
        evaluate_after: bool = True,
    ) -> tuple[AgentRunResponse, Optional[EvaluationResult]]:
        """Execute RAG retry path when Lead Agent determines RAG can resolve missing information.

        Workflow:
        Lead Agent -> search_project_knowledge -> ActionResult -> (optional re-evaluation) -> EvaluationResult

        Args:
            query: Focused knowledge search query. If omitted, synthesized from missing information.
            context: Optional tenant AgentContext.
            evaluate_after: If True, immediately re-evaluates the resulting ActionResult with Evaluation Sub-Agent.

        Returns:
            tuple[AgentRunResponse, Optional[EvaluationResult]]: Response and optional re-evaluation result.
        """
        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        if not query:
            latest_eval = self._state.latest_evaluation_result
            if latest_eval and latest_eval.missing_information:
                query = f"Retrieve project details for: {', '.join(latest_eval.missing_information[:3])}"
            else:
                query = f"Retrieve relevant project knowledge for section {self._state.current_section or 'BRD'}"

        logger.info("Executing RAG retry path with query: %s", query)
        request = AgentRunRequest(input_text=query, context=effective_ctx)
        response = self.execute(request=request, context=effective_ctx)

        if response.action_result and response.action_result.is_rag and response.action_result.content:
            self._state.add_evidence({
                "source": ActionSource.RAG.value,
                "query": query,
                "content": response.action_result.content,
                "is_authoritative": True,
            })

        eval_result = None
        if evaluate_after and response.action_result:
            eval_result = self.evaluate(
                action_result=response.action_result,
                context=effective_ctx,
            )

        return response, eval_result

    def construct_section_retrieval_query(
        self,
        section: str,
        requirements: Optional[Sequence[str]] = None,
        context: Optional[AgentContext] = None,
    ) -> str:
        """Construct a section-aware RAG search query driven by the section's specific requirements.

        Enforces DEF-008:
        Different BRD sections have different information needs. The query incorporates
        the current section name and its authoritative template requirements.
        """
        sec_reqs = (
            list(requirements)
            if requirements is not None
            else extract_section_requirements(section, self._template)
        )
        if sec_reqs:
            clean_reqs = [re.sub(r"^[0-9\.\-\*\s]+", "", r).strip() for r in sec_reqs if r.strip()]
            req_summary = "; ".join(clean_reqs[:3])
            return f"Section: {section}. Requirements: {req_summary}"
        return f"Project requirements, specifications, and architecture for {section}"

    def construct_section_rework_query(
        self,
        section: str,
        missing_items: Sequence[str],
        context: Optional[AgentContext] = None,
    ) -> str:
        """Construct a targeted RAG retrieval query to recover missing information for section rework.

        Enforces DEF-010:
        Recovers missing information by querying the project knowledge base for the
        specific gaps identified during validation.
        """
        clean_items = [re.sub(r"^[0-9\.\-\*\s]+", "", it).strip() for it in missing_items if it.strip()]
        topic_summary = "; ".join(clean_items[:3])
        return f"Section: {section}. Missing information: {topic_summary}"

    async def retrieve_section_evidence_async(
        self,
        section: Optional[str] = None,
        query: Optional[str | Sequence[str]] = None,
        requirements: Optional[Sequence[str]] = None,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Execute project-scoped RAG retrieval for a specific BRD section and store authoritative evidence.

        Preserves project boundary isolation by passing project_id via runtime AgentContext.
        Adds retrieved knowledge to state.evidence with source='rag' and is_authoritative=True.
        """
        if not self.has_rag_capability:
            logger.info("Skipping section retrieval for '%s': RAG capability not equipped.", section)
            return None

        target_sec = section or kwargs.get("section_name") or self._state.current_section or "BRD Section"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        # Skip RAG for administrative sections (mechanics and administrative metadata do not require RAG)
        if is_administrative_section(target_sec):
            logger.info("Skipping RAG retrieval for administrative section '%s' (mechanics and administrative metadata do not require RAG).", target_sec)
            return None

        if isinstance(query, (list, tuple)):
            search_query = self.construct_section_retrieval_query(
                section=target_sec,
                requirements=query,
                context=effective_ctx,
            )
        elif query:
            search_query = str(query)
        elif requirements:
            search_query = self.construct_section_retrieval_query(
                section=target_sec,
                requirements=requirements,
                context=effective_ctx,
            )
        else:
            search_query = self.construct_section_retrieval_query(
                section=target_sec,
                context=effective_ctx,
            )

        # Avoid executing identical retrieval query repeatedly during rework
        executed_queries = self._state.metadata.setdefault("executed_rag_queries", {})
        if search_query in executed_queries:
            logger.info("Skipping duplicate RAG query for '%s': already executed (%s)", target_sec, search_query)
            return executed_queries[search_query]

        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.TOOL,
            event_name="SECTION_RAG_RETRIEVAL_STARTED",
            section=target_sec,
            query=search_query,
        )

        rag_tool = next((t for t in self._tools if getattr(t, "name", "") == "search_project_knowledge"), None)
        if rag_tool is None:
            return None

        token = set_current_agent_context(effective_ctx)
        try:
            if hasattr(rag_tool, "ainvoke"):
                result_str = await rag_tool.ainvoke({"query": search_query})
            else:
                result_str = rag_tool.invoke({"query": search_query})
        finally:
            reset_current_agent_context(token)

        if not result_str or not isinstance(result_str, str):
            return None

        executed_queries[search_query] = result_str

        if "[RETRIEVAL_SUCCESS]" in result_str:
            evidence_item = {
                "source": ActionSource.RAG.value,
                "section": target_sec,
                "query": search_query,
                "content": result_str,
                "is_authoritative": True,
            }
            self._state.add_evidence(evidence_item)
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.TOOL,
                event_name="SECTION_RAG_RETRIEVAL_COMPLETED",
                section=target_sec,
                success=True,
            )
            return result_str
        elif "[NO_EVIDENCE]" in result_str:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.TOOL,
                event_name="SECTION_RAG_RETRIEVAL_NO_EVIDENCE",
                section=target_sec,
            )
            return None
        else:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.TOOL,
                event_name="SECTION_RAG_RETRIEVAL_FAILED",
                section=target_sec,
                result=result_str,
            )
            return None

    def retrieve_section_evidence(
        self,
        section: Optional[str] = None,
        query: Optional[str | Sequence[str]] = None,
        requirements: Optional[Sequence[str]] = None,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Synchronous execution of section-specific RAG retrieval."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.retrieve_section_evidence_async(
                        section=section,
                        query=query,
                        requirements=requirements,
                        context=context,
                        **kwargs,
                    ),
                ).result()
        else:
            return loop.run_until_complete(
                self.retrieve_section_evidence_async(
                    section=section,
                    query=query,
                    requirements=requirements,
                    context=context,
                    **kwargs,
                )
            )

    def receive_user_clarification(
        self,
        answer: str,
        resolved_item: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> None:
        """Incorporate user clarification into Agent State and context.

        Args:
            answer: User's clarifying answer or decision.
            resolved_item: Optional specific unresolved or missing item that this answer resolves.
            context: Optional tenant AgentContext.
        """
        logger.info("Received user clarification (len: %d)", len(answer))
        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        clarification_res = ActionResult(
            source="user_clarification",
            content=answer,
            success=True,
            metadata={"resolved_item": resolved_item},
        )
        self._state.set_action_result(clarification_res)

        self._state.add_evidence({
            "source": "user_clarification",
            "content": answer,
            "resolved_item": resolved_item,
        })

        if resolved_item:
            self._state.resolve_unresolved(resolved_item)

        self._state.clear_waiting_for_user()

    def generate_section(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> SectionGenerationResult:
        """Generate or update a BRD section via the Section Generation Sub-Agent.

        Workflow:
        BRD Lead Agent (prepares context/evidence) -> Section Generation Sub-Agent ->
        SectionGenerationResult -> Lead Agent stores result in BRDAgentState.

        Args:
            section: Target section name (e.g. "5. Stakeholders & Personas"). Defaults to
                self.state.current_section.
            available_information: Optional specific information or evidence to base the section on.
                If omitted, gathered from latest action result, working evidence, and task results.
            section_requirements: Optional specific requirements/criteria. If omitted,
                extracted dynamically from the authoritative BRD template.
            template_structure: Optional template snippet. If omitted, extracted dynamically
                from the authoritative BRD template.
            existing_content: Optional baseline section content for update/rework. If omitted,
                retrieved from state.get_section_content().
            rework_feedback: Optional validation or review feedback. If omitted, retrieved
                from state.get_rework_feedback().
            context: Optional tenant AgentContext.

        Returns:
            SectionGenerationResult: Generated or updated section content and metadata.

        Raises:
            ValueError: If no target section can be identified.
        """
        target_sec = section or self._state.current_section
        if not target_sec:
            raise ValueError(
                "No target section specified or set as current_section in Agent State."
            )
        self._state.set_current_section(target_sec)
        canonical_sec = self._state.current_section or target_sec

        if section_requirements is not None:
            sec_reqs = [section_requirements] if isinstance(section_requirements, str) else list(section_requirements)
        else:
            sec_reqs = extract_section_requirements(canonical_sec, self._template)

        sec_template = template_structure or extract_section_template(canonical_sec, self._template)

        if available_information is not None:
            info_payload = available_information
        else:
            info_parts: list[Any] = []
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.latest_action_result and self._state.latest_action_result.content:
                if self._state.latest_action_result.is_rag:
                    info_parts.append({
                        "source": ActionSource.RAG.value,
                        "content": self._state.latest_action_result.content,
                    })
                elif self._state.latest_action_result.is_direct_work:
                    info_parts.append({
                        "source": "agent_analysis",
                        "content": self._state.latest_action_result.content,
                    })
            if self._state.agent_work:
                for w in self._state.agent_work[-2:]:
                    info_parts.append({
                        "source": "agent_analysis",
                        "content": w.get("content") if isinstance(w, dict) else getattr(w, "content", str(w)),
                    })
            if self._state.task_results:
                for tr in self._state.task_results:
                    info_parts.append({
                        "source": f"delegated_task:{tr.task_id}",
                        "content": tr.output,
                    })
            info_payload = info_parts if info_parts else "*(No prior evidence collected)*"

        resolved_existing = (
            existing_content
            if existing_content is not None
            else self._state.get_section_content(canonical_sec)
        )
        resolved_rework = (
            rework_feedback
            if rework_feedback is not None
            else self._state.get_rework_feedback(canonical_sec)
        )

        op = (
            SectionOperation.UPDATE
            if (resolved_existing and resolved_existing.strip())
            else SectionOperation.GENERATE
        )

        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        gen_ctx = SectionGenerationContext(
            section_name=canonical_sec,
            section_requirements=sec_reqs,
            template_structure=sec_template,
            available_information=info_payload,
            existing_content=resolved_existing,
            rework_feedback=resolved_rework,
            operation=op,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
                "available_documents": getattr(effective_ctx, "available_documents", None) or self._state.metadata.get("available_documents"),
                "document_version": self.resolve_document_version(),
                "document_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "question_id_prefix": "Q-BRD-",
            },
        )

        result = self.section_generator.generate(gen_ctx)

        if result.content:
            self._state.set_section_content(canonical_sec, result.content)
        self._state.set_section_result(result)
        if resolved_rework:
            self._state.clear_rework_feedback(canonical_sec)

        return result

    async def generate_section_async(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> SectionGenerationResult:
        """Asynchronously generate or update a BRD section via the Section Generation Sub-Agent."""
        target_sec = section or self._state.current_section
        if not target_sec:
            raise ValueError(
                "No target section specified or set as current_section in Agent State."
            )
        self._state.set_current_section(target_sec)
        canonical_sec = self._state.current_section or target_sec

        if section_requirements is not None:
            sec_reqs = [section_requirements] if isinstance(section_requirements, str) else list(section_requirements)
        else:
            sec_reqs = extract_section_requirements(canonical_sec, self._template)

        sec_template = template_structure or extract_section_template(canonical_sec, self._template)

        if available_information is not None:
            info_payload = available_information
        else:
            info_parts: list[Any] = []
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.latest_action_result and self._state.latest_action_result.content:
                if self._state.latest_action_result.is_rag:
                    info_parts.append({
                        "source": ActionSource.RAG.value,
                        "content": self._state.latest_action_result.content,
                    })
                elif self._state.latest_action_result.is_direct_work:
                    info_parts.append({
                        "source": "agent_analysis",
                        "content": self._state.latest_action_result.content,
                    })
            if self._state.agent_work:
                for w in self._state.agent_work[-2:]:
                    info_parts.append({
                        "source": "agent_analysis",
                        "content": w.get("content") if isinstance(w, dict) else getattr(w, "content", str(w)),
                    })
            if self._state.task_results:
                for tr in self._state.task_results:
                    info_parts.append({
                        "source": f"delegated_task:{tr.task_id}",
                        "content": tr.output,
                    })
            info_payload = info_parts if info_parts else "*(No prior evidence collected)*"

        resolved_existing = (
            existing_content
            if existing_content is not None
            else self._state.get_section_content(canonical_sec)
        )
        resolved_rework = (
            rework_feedback
            if rework_feedback is not None
            else self._state.get_rework_feedback(canonical_sec)
        )

        op = (
            SectionOperation.UPDATE
            if (resolved_existing and resolved_existing.strip())
            else SectionOperation.GENERATE
        )

        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        gen_ctx = SectionGenerationContext(
            section_name=canonical_sec,
            section_requirements=sec_reqs,
            template_structure=sec_template,
            available_information=info_payload,
            existing_content=resolved_existing,
            rework_feedback=resolved_rework,
            operation=op,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
                "available_documents": getattr(effective_ctx, "available_documents", None) or self._state.metadata.get("available_documents"),
                "document_version": self.resolve_document_version(),
                "document_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "question_id_prefix": "Q-BRD-",
            },
        )

        phase = "8_FINAL_RECOVERY" if self._state.final_validation_recovery_cycles > 0 else "5_SECTION_ITERATION"
        op_name = "update_section" if op == SectionOperation.UPDATE else "generate_section"
        with scoped_telemetry_context(
            component="BRDSectionGenerationAgent",
            phase=phase,
            operation=op_name,
            section=canonical_sec,
        ):
            result = await self.section_generator.generate_async(gen_ctx)

        if result.content:
            self._state.set_section_content(canonical_sec, result.content)
        self._state.set_section_result(result)
        if resolved_rework:
            self._state.clear_rework_feedback(canonical_sec)

        return result

    def update_section(
        self,
        section: Optional[str] = None,
        new_information: Optional[Sequence[Any] | str] = None,
        rework_feedback: Optional[str] = None,
        existing_content: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> SectionGenerationResult:
        """Update an existing BRD section incorporating new information and rework feedback."""
        return self.generate_section(
            section=section,
            available_information=new_information,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
        )

    async def update_section_async(
        self,
        section: Optional[str] = None,
        new_information: Optional[Sequence[Any] | str] = None,
        rework_feedback: Optional[str] = None,
        existing_content: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> SectionGenerationResult:
        """Asynchronously update an existing BRD section incorporating new information and rework feedback."""
        return await self.generate_section_async(
            section=section,
            available_information=new_information,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
        )

    def validate_section(
        self,
        section: Optional[str] = None,
        section_content: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        prior_rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> ValidationResult:
        """Validate a generated or updated BRD section via the Section Validation Sub-Agent.

        Workflow:
        BRD Lead Agent (prepares validation context/evidence) -> Section Validation Sub-Agent ->
        ValidationResult -> Lead Agent inspects outcome, updates state and section status,
        and records rework feedback if rework is required.

        Args:
            section: Target section name (e.g. "5. Stakeholders & Personas"). Defaults to
                self.state.current_section.
            section_content: Complete section Markdown to validate. If omitted, retrieved
                from state.get_section_content().
            available_information: Optional specific information or evidence to validate against.
                If omitted, gathered from state (latest action result, working evidence, task results).
            section_requirements: Optional specific requirements/criteria. If omitted,
                extracted dynamically from the authoritative BRD template.
            template_structure: Optional template snippet. If omitted, extracted dynamically
                from the authoritative BRD template.
            prior_rework_feedback: Optional prior rework feedback if re-validating. If omitted,
                retrieved from state.get_rework_feedback().
            context: Optional tenant AgentContext.

        Returns:
            ValidationResult: Structured validation result with outcome, findings, and rework feedback.

        Raises:
            ValueError: If no target section can be identified or no content is available to validate.
        """
        target_sec = section or self._state.current_section
        if not target_sec:
            raise ValueError(
                "No target section specified or set as current_section in Agent State."
            )
        self._state.set_current_section(target_sec)
        canonical_sec = self._state.current_section or target_sec

        content_to_validate = (
            section_content
            if section_content is not None
            else self._state.get_section_content(canonical_sec)
        )
        if not content_to_validate or not content_to_validate.strip():
            raise ValueError(
                f"No section content available to validate for '{canonical_sec}'."
            )

        if section_requirements is not None:
            sec_reqs = (
                [section_requirements]
                if isinstance(section_requirements, str)
                else list(section_requirements)
            )
        else:
            sec_reqs = extract_section_requirements(canonical_sec, self._template)

        sec_template = (
            template_structure
            or extract_section_template(canonical_sec, self._template)
        )

        if available_information is not None:
            info_payload = available_information
        else:
            info_parts: list[Any] = []
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.latest_action_result and self._state.latest_action_result.is_rag and self._state.latest_action_result.content:
                info_parts.append({
                    "source": ActionSource.RAG.value,
                    "content": self._state.latest_action_result.content,
                })
            info_payload = info_parts if info_parts else "*(No prior project evidence collected)*"

        resolved_prior_rework = (
            prior_rework_feedback
            if prior_rework_feedback is not None
            else self._state.get_rework_feedback(canonical_sec)
        )

        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        val_ctx = SectionValidationContext(
            section_name=canonical_sec,
            section_content=content_to_validate,
            section_requirements=sec_reqs,
            template_structure=sec_template,
            available_information=info_payload,
            prior_rework_feedback=resolved_prior_rework,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
            },
        )

        result = self.section_validator.validate(val_ctx)
        self._state.set_validation_result(result)

        if result.is_valid:
            self._state.update_section_status(canonical_sec, BRDSectionStatus.COMPLETED)
            self._state.clear_rework_feedback(canonical_sec)
            logger.info("Section accepted (section: %s)", canonical_sec)
        else:
            self._state.update_section_status(canonical_sec, BRDSectionStatus.NEEDS_REVISION)
            if result.rework_feedback:
                self._state.set_rework_feedback(canonical_sec, result.rework_feedback)
            logger.warning(
                "Rework required for section: %s (findings: %d)",
                canonical_sec,
                len(result.findings),
            )

        return result

    async def validate_section_async(
        self,
        section: Optional[str] = None,
        section_content: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        prior_rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> ValidationResult:
        """Asynchronously validate a generated or updated BRD section via Section Validation Sub-Agent."""
        target_sec = section or self._state.current_section
        if not target_sec:
            raise ValueError(
                "No target section specified or set as current_section in Agent State."
            )
        self._state.set_current_section(target_sec)
        canonical_sec = self._state.current_section or target_sec

        content_to_validate = (
            section_content
            if section_content is not None
            else self._state.get_section_content(canonical_sec)
        )
        if not content_to_validate or not content_to_validate.strip():
            raise ValueError(
                f"No section content available to validate for '{canonical_sec}'."
            )

        if section_requirements is not None:
            sec_reqs = (
                [section_requirements]
                if isinstance(section_requirements, str)
                else list(section_requirements)
            )
        else:
            sec_reqs = extract_section_requirements(canonical_sec, self._template)

        sec_template = (
            template_structure
            or extract_section_template(canonical_sec, self._template)
        )

        if available_information is not None:
            info_payload = available_information
        else:
            info_parts: list[Any] = []
            if self._state.evidence:
                info_parts.extend(self._state.evidence)
            if self._state.latest_action_result and self._state.latest_action_result.is_rag and self._state.latest_action_result.content:
                info_parts.append({
                    "source": ActionSource.RAG.value,
                    "content": self._state.latest_action_result.content,
                })
            info_payload = info_parts if info_parts else "*(No prior project evidence collected)*"

        resolved_prior_rework = (
            prior_rework_feedback
            if prior_rework_feedback is not None
            else self._state.get_rework_feedback(canonical_sec)
        )

        effective_ctx = context or AgentContext()
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        val_ctx = SectionValidationContext(
            section_name=canonical_sec,
            section_content=content_to_validate,
            section_requirements=sec_reqs,
            template_structure=sec_template,
            available_information=info_payload,
            prior_rework_feedback=resolved_prior_rework,
            metadata={
                "project_id": effective_ctx.project_id or self._state.metadata.get("project_id"),
                "project_name": getattr(effective_ctx, "project_name", None) or self._state.metadata.get("project_name"),
                "project_description": getattr(effective_ctx, "project_description", None) or self._state.metadata.get("project_description"),
            },
        )

        phase = "8_FINAL_RECOVERY" if self._state.final_validation_recovery_cycles > 0 else "5_SECTION_ITERATION"
        with scoped_telemetry_context(
            component="BRDSectionValidationAgent",
            phase=phase,
            operation="validate_section",
            section=canonical_sec,
        ):
            result = await self.section_validator.validate_async(val_ctx)
        self._state.set_validation_result(result)

        if result.is_valid:
            self._state.update_section_status(canonical_sec, BRDSectionStatus.COMPLETED)
            self._state.clear_rework_feedback(canonical_sec)
            logger.info("Section accepted (section: %s)", canonical_sec)
        else:
            self._state.update_section_status(canonical_sec, BRDSectionStatus.NEEDS_REVISION)
            if result.rework_feedback:
                self._state.set_rework_feedback(canonical_sec, result.rework_feedback)
            logger.warning(
                "Rework required for section: %s (findings: %d)",
                canonical_sec,
                len(result.findings),
            )

        return result

    def construct_section_clarification_question(
        self,
        section_name: str,
        validation_result: Optional[ValidationResult] = None,
        missing_items: Optional[Sequence[str]] = None,
    ) -> str:
        """Construct a targeted, domain-specific clarification question for an evidence gap.

        Follows DEF-011 policy: communicates clearly what project/business information
        is missing from available evidence without exposing internal prompt or implementation details.
        """
        items = list(missing_items or [])
        findings = validation_result.findings if validation_result else []

        details: list[str] = []
        for f in findings:
            cat_val = f.category.value if hasattr(f.category, "value") else str(f.category)
            if cat_val in {
                ValidationCategory.GROUNDING.value,
                ValidationCategory.COMPLETENESS.value,
                ValidationCategory.REQUIREMENT_COVERAGE.value,
            }:
                desc = getattr(f, "issue", None) or getattr(f, "explanation", None) or getattr(f, "required_change", None)
                if desc and desc not in details:
                    details.append(desc)

        if not details and items:
            details = items

        if details:
            specifics = "; ".join(details[:3])
            return (
                f"The \"{section_name}\" section could not be completed because required project/business information "
                f"is missing from the available documentation: {specifics}.\n\n"
                f"Please provide clarification or specific details for {section_name} so we can complete this section."
            )
        return (
            f"The \"{section_name}\" section could not be completed because required project specifications "
            f"are missing from the available evidence.\n\n"
            f"Please provide the necessary business rules and requirements for {section_name}."
        )

    def resolve_document_version(self) -> str:
        """Determine document version deterministically:
        - New BRD -> 1.0
        - Existing BRD update -> increment minor version (e.g. 1.0 -> 1.1)
        """
        existing_version = self._state.metadata.get("document_version")
        if not existing_version:
            assembled_doc = getattr(self._state, "assembled_brd", None) or getattr(self._state, "assembled_document", None) or self._state.metadata.get("existing_brd_content")
            if assembled_doc:
                match = re.search(r"Version\s*[:\s|]\s*v?(\d+\.\d+)", assembled_doc, re.IGNORECASE)
                if match:
                    existing_version = match.group(1)
        if existing_version:
            try:
                parts = existing_version.split(".")
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                new_version = f"{major}.{minor + 1}"
                self._state.metadata["document_version"] = new_version
                return new_version
            except Exception:
                pass
        self._state.metadata["document_version"] = "1.0"
        return "1.0"

    def _ingest_user_conversation_evidence(
        self,
        input_prompt: Optional[str],
        prior_messages: Optional[Sequence[Any]] = None,
    ) -> None:
        """Extract authoritative user conversation statements into working evidence.

        Strict Provenance Rules:
        - User messages become evidence with source='user_conversation', role='user', is_authoritative=True.
        - Assistant messages are NEVER added as authoritative evidence (prevents model-echo loops).
        - Evidence items are deduplicated by (content, source) via state.add_evidence.
        """
        if prior_messages:
            for msg in prior_messages:
                role = None
                content = None
                if isinstance(msg, HumanMessage):
                    role = "user"
                    content = msg.content
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role") or msg.get("sender") or msg.get("type")
                    content = msg.get("content") or msg.get("text")
                elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                    role = str(msg[0])
                    content = str(msg[1])
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    role = getattr(msg, "role")
                    content = getattr(msg, "content")

                # Authoritative user evidence ONLY
                if role in ("user", "human") and content:
                    content_str = str(content).strip()
                    if content_str:
                        self._state.add_evidence({
                            "source": "user_conversation",
                            "role": "user",
                            "content": content_str,
                            "is_authoritative": True,
                        })

        if input_prompt and isinstance(input_prompt, str) and input_prompt.strip():
            self._state.add_evidence({
                "source": "user_conversation",
                "role": "user",
                "content": input_prompt.strip(),
                "is_authoritative": True,
            })

    def construct_consolidated_clarification_question(
        self,
        unresolved_items: Sequence[str],
    ) -> str:
        """Construct a consolidated clarification question grouping genuine unresolved business/technical gaps."""
        deduped: list[str] = []
        seen: set[str] = set()
        for item in unresolved_items:
            clean = str(item).strip().rstrip(".")
            if not clean:
                continue
            if is_metadata_or_role_item(clean):
                continue
            if clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)

        if not deduped:
            return "Please provide any additional business or technical requirements to complete the BRD."

        lines = [
            "Before I finalize the BRD, I need clarification on the following items:\n",
        ]
        for i, item in enumerate(deduped, 1):
            if not item.endswith("?"):
                lines.append(f"{i}. Please clarify the requirement or specification for: {item}")
            else:
                lines.append(f"{i}. {item}")

        return "\n".join(lines)

    def _format_section_failure_diagnostic(
        self,
        section_name: str,
        failure_category: SectionFailureCategory | str,
        validation_result: Optional[ValidationResult] = None,
        completed_sections: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], str]:
        """Format structured diagnostic dictionary and user-facing diagnostic text for section rework failure."""
        cat_str = (
            failure_category.value
            if isinstance(failure_category, SectionFailureCategory)
            else str(failure_category)
        )
        completed = completed_sections if completed_sections is not None else [
            sec for sec in self.sections if self._state.get_section_status(sec) == BRDSectionStatus.COMPLETED
        ]
        findings_data = []
        findings_bullets = []
        if validation_result and validation_result.findings:
            for f in validation_result.findings:
                c_val = f.category.value if hasattr(f.category, "value") else str(f.category)
                sev = getattr(f, "severity", None)
                sev_str = sev.value if hasattr(sev, "value") else (str(sev) if sev is not None else "high")
                findings_data.append({
                    "category": c_val,
                    "issue": f.issue,
                    "explanation": f.explanation,
                    "required_change": f.required_change,
                    "severity": sev_str,
                })
                findings_bullets.append(f"- [{c_val}] {f.issue}: {f.explanation}")

        findings_text = "\n".join(findings_bullets) if findings_bullets else "- No specific findings reported."

        diagnostics = {
            "status": "failed",
            "reason": "rework_exhausted",
            "failure_category": cat_str,
            "section": section_name,
            "completed_sections": completed,
            "validation_findings": findings_data,
            "section_status": (
                self._state.get_section_status(section_name).value
                if self._state.get_section_status(section_name)
                else "needs_revision"
            ),
        }

        output_text = (
            f"BRD workflow halted: not all template sections could be completed. "
            f"Section '{section_name}' failed validation after rework was exhausted due to {cat_str.replace('_', ' ')} issues.\n"
            f"Completed sections ({len(completed)}/{len(self.sections)}): {', '.join(completed) if completed else 'None'}.\n"
            f"Validation findings:\n{findings_text}"
        )
        return diagnostics, output_text

    def interpret_section_validation(
        self,
        validation_result: Optional[ValidationResult] = None,
        section_name: Optional[str] = None,
        context: Optional[AgentContext] = None,
        current_content: Optional[str] = None,
        rework_attempt: int = 1,
    ) -> SectionReworkStrategy:
        """Intelligently interpret section validation findings and formulate rework strategy.

        Responsibilities:
        - Sub-agent validates section against requirements and returns ValidationResult.
        - Lead Agent interprets the findings and determines the rework strategy:
          * Identifies whether rework is required.
          * Formulates actionable rework guidance.
          * Specifies targeted adjustments to content.
          * Classifies unresolved failures into approved DEF-011 categories:
            EVIDENCE_GAP, GENERATION_QUALITY, TEMPLATE_CONFIGURATION.
        - Application enforces the deterministic retry limit (max 2 attempts) and executes updates.
        """
        # Handle if section_name and validation_result were passed inverted as positional arguments
        if isinstance(validation_result, str) and (section_name is None or isinstance(section_name, ValidationResult)):
            validation_result, section_name = section_name, validation_result

        target_sec = section_name or self._state.current_section or "Current Section"
        effective_ctx = context or AgentContext(project_id=self.project_id)
        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None)

        val_res = validation_result or self._state.get_section_validation_result(target_sec)
        if val_res is None:
            raise ValueError(f"No validation result available for section '{target_sec}' to interpret.")

        if val_res.is_valid:
            strategy = SectionReworkStrategy(
                section_name=target_sec,
                requires_rework=False,
                reasoning=f"Section '{target_sec}' satisfies all validation requirements.",
            )
        else:
            guidance = val_res.rework_feedback or f"Address validation findings for {target_sec}."
            specific_adjs = [
                (getattr(f, "required_change", None) or getattr(f, "issue", None) or getattr(f, "observation", str(f)))
                for f in val_res.findings
            ]

            # Filter benign administrative metadata and documentation-quality findings
            substantive_findings = [
                f for f in val_res.findings
                if not is_benign_administrative_metadata_finding(f, target_sec)
                and not is_documentation_quality_finding(f, target_sec)
            ]

            if not substantive_findings:
                strategy = SectionReworkStrategy(
                    section_name=target_sec,
                    requires_rework=False,
                    reasoning=f"Section '{target_sec}' satisfies requirements (documentation observations and administrative placeholders are accepted).",
                )
                return strategy

            # DEF-011: Classification of Unresolved Problem
            template_findings = []
            for f in substantive_findings:
                cat_val = f.category.value if hasattr(f.category, "value") else str(f.category)
                issue_lower = (f.issue or "").lower()
                expl_lower = (f.explanation or "").lower()
                if (
                    cat_val == ValidationCategory.TEMPLATE_COMPLIANCE.value
                    or "contradict" in issue_lower
                    or "contradict" in expl_lower
                    or "template constraint" in issue_lower
                    or "configuration" in issue_lower
                    or "structural inconsistency" in issue_lower
                ):
                    template_findings.append(f)

            evidence_gap_categories = {
                ValidationCategory.GROUNDING.value,
                ValidationCategory.COMPLETENESS.value,
                ValidationCategory.REQUIREMENT_COVERAGE.value,
            }
            evidence_gap_findings = []
            for f in substantive_findings:
                cat_val = f.category.value if hasattr(f.category, "value") else str(f.category)
                issue_lower = (f.issue or "").lower()
                expl_lower = (f.explanation or "").lower()
                if cat_val in evidence_gap_categories:
                    if any(w in issue_lower or w in expl_lower for w in ["missing", "not found", "ungrounded", "unverified", "evidence", "specify", "unknown", "lack of", "absent", "gap"]):
                        evidence_gap_findings.append(f)
                    elif cat_val == ValidationCategory.GROUNDING.value:
                        evidence_gap_findings.append(f)

            requires_retrieval = False
            retrieval_query = None
            missing_items: list[str] = []
            clarification_question = None

            if template_findings and not evidence_gap_findings:
                failure_category = SectionFailureCategory.TEMPLATE_CONFIGURATION
                reasoning = (
                    f"Section '{target_sec}' validation identified template or configuration contradictions: "
                    f"{', '.join(getattr(f, 'issue', str(f)) for f in template_findings[:2])}."
                )
            elif evidence_gap_findings:
                failure_category = SectionFailureCategory.EVIDENCE_GAP
                missing_items = [
                    (getattr(f, "issue", None) or getattr(f, "required_change", str(f)))
                    for f in evidence_gap_findings
                ]
                clarification_question = self.construct_section_clarification_question(
                    section_name=target_sec,
                    validation_result=val_res,
                    missing_items=missing_items,
                )
                if self.has_rag_capability:
                    requires_retrieval = True
                    retrieval_query = self.construct_section_rework_query(
                        section=target_sec,
                        missing_items=missing_items,
                        context=effective_ctx,
                    )
                    reasoning = (
                        f"Section '{target_sec}' validation identified missing or ungrounded information. "
                        f"Rework required with targeted RAG retrieval for: {', '.join(missing_items[:2])}."
                    )
                else:
                    reasoning = (
                        f"Section '{target_sec}' validation identified evidence gap: {', '.join(missing_items[:2])}. "
                        f"Direct user clarification will be required if rework cannot be resolved."
                    )
            else:
                failure_category = SectionFailureCategory.GENERATION_QUALITY
                reasoning = f"Section '{target_sec}' validation status is {val_res.outcome.value}. Rework required for writing/compliance."

            strategy = SectionReworkStrategy(
                section_name=target_sec,
                requires_rework=True,
                rework_guidance=guidance,
                specific_adjustments=specific_adjs,
                reasoning=reasoning,
                requires_retrieval=requires_retrieval,
                retrieval_query=retrieval_query,
                missing_evidence_items=missing_items,
                failure_category=failure_category,
                clarification_question=clarification_question,
            )

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="SECTION_VALIDATION_INTERPRETED",
            section=target_sec,
            requires_rework=strategy.requires_rework,
            failure_category=strategy.failure_category.value if strategy.failure_category else None,
        )
        return strategy

    async def interpret_section_validation_async(
        self,
        validation_result: Optional[ValidationResult] = None,
        section_name: Optional[str] = None,
        context: Optional[AgentContext] = None,
        current_content: Optional[str] = None,
        rework_attempt: int = 1,
    ) -> SectionReworkStrategy:
        """Asynchronously interpret section validation findings and formulate rework strategy."""
        return self.interpret_section_validation(
            validation_result=validation_result,
            section_name=section_name,
            context=context,
            current_content=current_content,
            rework_attempt=rework_attempt,
        )

    def generate_and_validate_section(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
        auto_progress: bool = False,
    ) -> tuple[SectionGenerationResult, ValidationResult]:
        """Execute the generate -> validate -> rework retry loop for a BRD section.

        Workflow:
        1. Generate initial section via Section Generation Sub-Agent.
        2. Validate section via Section Validation Sub-Agent.
        3. If VALID: accept section, mark completed, optionally auto-progress, and return.
        4. If NEEDS_REWORK: store feedback, update section, re-validate up to max_rework_attempts.
        5. Protect against uncontrolled infinite retry loops via max_rework_attempts safeguard.

        Args:
            section: Target section name. Defaults to self.state.current_section.
            available_information: Optional specific information/evidence.
            section_requirements: Optional specific requirements.
            template_structure: Optional template structure.
            existing_content: Optional baseline content for updates.
            rework_feedback: Optional initial rework feedback.
            context: Optional tenant AgentContext.
            max_rework_attempts: Maximum number of rework/retry cycles if validation fails. Defaults to 2.
            auto_progress: Whether to automatically advance to next section upon VALID validation. Defaults to False.

        Returns:
            tuple[SectionGenerationResult, ValidationResult]: Latest generation and validation results.
        """
        # Step 1: Initial generation (or update if existing_content was provided)
        gen_result = self.generate_section(
            section=section,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
        )

        target_sec = section or self._state.current_section or gen_result.section_name

        # Step 2: Validate generated section
        val_result = self.validate_section(
            section=target_sec,
            section_content=gen_result.content,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            context=context,
        )

        if val_result.is_valid:
            if auto_progress:
                self.progress_section(context=context)
            return gen_result, val_result

        # Step 3: Rework loop if validation indicated NEEDS_REWORK
        canonical_sec = self._state.current_section or target_sec
        for attempt in range(1, max_rework_attempts + 1):
            logger.info(
                "Rework started for section: %s (attempt %d/%d)",
                canonical_sec,
                attempt,
                max_rework_attempts,
            )
            # Re-generate/update with rework feedback
            gen_result = self.update_section(
                section=canonical_sec,
                new_information=available_information,
                rework_feedback=val_result.rework_feedback,
                existing_content=gen_result.content,
                context=context,
            )

            # Re-validate updated section
            val_result = self.validate_section(
                section=canonical_sec,
                section_content=gen_result.content,
                available_information=available_information,
                section_requirements=section_requirements,
                template_structure=template_structure,
                prior_rework_feedback=val_result.rework_feedback,
                context=context,
            )

            if val_result.is_valid:
                break

        if val_result.is_valid and auto_progress:
            self.progress_section(context=context)

        return gen_result, val_result

    async def generate_and_validate_section_async(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
        auto_progress: bool = False,
    ) -> tuple[SectionGenerationResult, ValidationResult]:
        """Asynchronously execute the generate -> validate -> rework retry loop for a BRD section."""
        # Step 1: Initial generation (or update if existing_content was provided)
        gen_result = await self.generate_section_async(
            section=section,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
        )

        target_sec = section or self._state.current_section or gen_result.section_name

        # Step 2: Validate generated section
        val_result = await self.validate_section_async(
            section=target_sec,
            section_content=gen_result.content,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            context=context,
        )

        if val_result.is_valid:
            if auto_progress:
                self.progress_section(context=context)
            return gen_result, val_result

        # Step 3: Rework loop if validation indicated NEEDS_REWORK
        canonical_sec = self._state.current_section or target_sec
        for attempt in range(1, max_rework_attempts + 1):
            logger.info(
                "Rework started for section: %s (attempt %d/%d)",
                canonical_sec,
                attempt,
                max_rework_attempts,
            )
            # Re-generate/update with rework feedback
            gen_result = await self.update_section_async(
                section=canonical_sec,
                new_information=available_information,
                rework_feedback=val_result.rework_feedback,
                existing_content=gen_result.content,
                context=context,
            )

            # Re-validate updated section
            val_result = await self.validate_section_async(
                section=canonical_sec,
                section_content=gen_result.content,
                available_information=available_information,
                section_requirements=section_requirements,
                template_structure=template_structure,
                prior_rework_feedback=val_result.rework_feedback,
                context=context,
            )

            if val_result.is_valid:
                break

        if val_result.is_valid and auto_progress:
            self.progress_section(context=context)

        return gen_result, val_result

    def process_current_section(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
    ) -> tuple[SectionGenerationResult, ValidationResult, Optional[SectionProgressionResult]]:
        """Process current section through generate -> validate -> rework, then progress if VALID.

        Workflow:
        1. Generate and validate current section (with rework retry up to max_rework_attempts).
        2. If VALID: deterministically progress to the next section and record progression result.
        3. If NEEDS_REWORK: return without progressing.
        """
        gen_result, val_result = self.generate_and_validate_section(
            section=section,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
            max_rework_attempts=max_rework_attempts,
            auto_progress=False,
        )
        prog_result: Optional[SectionProgressionResult] = None
        if val_result.is_valid:
            prog_result = self.progress_section(context=context)
        return gen_result, val_result, prog_result

    async def process_current_section_async(
        self,
        section: Optional[str] = None,
        available_information: Optional[Sequence[Any] | str] = None,
        section_requirements: Optional[Sequence[str] | str] = None,
        template_structure: Optional[str] = None,
        existing_content: Optional[str] = None,
        rework_feedback: Optional[str] = None,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
    ) -> tuple[SectionGenerationResult, ValidationResult, Optional[SectionProgressionResult]]:
        """Asynchronously process current section through generate -> validate -> rework, then progress if VALID."""
        gen_result, val_result = await self.generate_and_validate_section_async(
            section=section,
            available_information=available_information,
            section_requirements=section_requirements,
            template_structure=template_structure,
            existing_content=existing_content,
            rework_feedback=rework_feedback,
            context=context,
            max_rework_attempts=max_rework_attempts,
            auto_progress=False,
        )
        prog_result: Optional[SectionProgressionResult] = None
        if val_result.is_valid:
            prog_result = self.progress_section(context=context)
        return gen_result, val_result, prog_result

    def process_all_sections(
        self,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
        auto_assemble: bool = False,
        auto_validate_final: bool = False,
        auto_recover_final: bool = False,
    ) -> list[tuple[SectionGenerationResult, ValidationResult, SectionProgressionResult]]:
        """Deterministically iterate through and process all top-level template sections sequentially.

        Workflow:
        1. Initialize progression at first uncompleted section.
        2. Iteratively generate and validate each section.
        3. If VALID: deterministically advance to the next section.
        4. Terminates when section_processing_complete is True or when a section fails validation.
        5. If auto_assemble is True and all sections completed, automatically assembles the full BRD.
        6. If auto_validate_final is True, executes final validation (with optional auto_recover_final).

        Returns:
            list[tuple[SectionGenerationResult, ValidationResult, SectionProgressionResult]]:
                Ordered execution trace of all processed sections.
        """
        results: list[tuple[SectionGenerationResult, ValidationResult, SectionProgressionResult]] = []

        if not self._state.current_section and not self.is_section_processing_complete:
            self.initialize_section_progression(context=context)

        while not self.is_section_processing_complete and self._state.current_section:
            cur_sec = self._state.current_section
            gen_res, val_res, prog_res = self.process_current_section(
                section=cur_sec,
                context=context,
                max_rework_attempts=max_rework_attempts,
            )
            if prog_res is not None:
                results.append((gen_res, val_res, prog_res))
            else:
                # Validation failed after max reworks; stop sequence
                break

        if auto_assemble and self.is_section_processing_complete:
            self.assemble_brd(context=context)
            if auto_validate_final:
                if auto_recover_final:
                    self.validate_and_recover_final_brd(context=context)
                else:
                    self.validate_final_brd(context=context)

        return results

    async def process_all_sections_async(
        self,
        context: Optional[AgentContext] = None,
        max_rework_attempts: int = 2,
        auto_assemble: bool = False,
        auto_validate_final: bool = False,
        auto_recover_final: bool = False,
    ) -> list[tuple[SectionGenerationResult, ValidationResult, SectionProgressionResult]]:
        """Asynchronously iterate through and process all top-level template sections sequentially."""
        results: list[tuple[SectionGenerationResult, ValidationResult, SectionProgressionResult]] = []

        if not self._state.current_section and not self.is_section_processing_complete:
            self.initialize_section_progression(context=context)

        while not self.is_section_processing_complete and self._state.current_section:
            cur_sec = self._state.current_section
            gen_res, val_res, prog_res = await self.process_current_section_async(
                section=cur_sec,
                context=context,
                max_rework_attempts=max_rework_attempts,
            )
            if prog_res is not None:
                results.append((gen_res, val_res, prog_res))
            else:
                break

        if auto_assemble and self.is_section_processing_complete:
            await self.assemble_brd_async(context=context)
            if auto_validate_final:
                if auto_recover_final:
                    await self.validate_and_recover_final_brd_async(context=context)
                else:
                    await self.validate_final_brd_async(context=context)

        return results

    async def run_workflow_async(
        self,
        request: Optional[AgentRunRequest | str] = None,
        context: Optional[AgentContext] = None,
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        max_section_rework_attempts: int = 2,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
        initial_state: Optional[BRDAgentState] = None,
        prior_messages: Optional[Sequence[Any]] = None,
        consolidate_clarification: bool = False,
    ) -> AgentRunResponse:
        """Execute the Application-Owned BRD Workflow lifecycle asynchronously.

        The application owns and enforces the 9-phase workflow sequence, gates, and bounded loops:
        - Phase 1: Context & Objective Initialization
        - Phase 2: Action Decision (Lead Agent Intelligence)
        - Phase 3: Action Execution (Application executes RAG, Direct Work, or Delegation)
        - Phase 4: Evidence Evaluation (Evaluation Sub-Agent) & Interpretation (Lead Agent)
        - Phase 5: Section Iteration Loop (Sequential processing with Max 2 Reworks & Progression Gate)
        - Phase 6: Document Assembly Gate (Assembly only when all sections complete)
        - Phase 7: Final Validation (Final Validation Sub-Agent)
        - Phase 8: Final Validation Interpretation (Lead Agent) & Recovery (Max 3 Cycles)
        - Phase 9: Workflow Completion & Normalized Response
        """
        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx is None:
            effective_ctx = AgentContext(project_id=self.project_id)
        elif not effective_ctx.project_id and self.project_id:
            effective_ctx = AgentContext(
                project_id=self.project_id,
                conversation_id=effective_ctx.conversation_id,
                user_id=effective_ctx.user_id,
                project_name=getattr(effective_ctx, "project_name", None),
                project_description=getattr(effective_ctx, "project_description", None),
                available_documents=list(getattr(effective_ctx, "available_documents", [])),
                metadata=effective_ctx.metadata,
            )

        if initial_state is not None:
            self._state = initial_state
        elif isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)
        elif (
            effective_ctx and (
                (self._state.metadata.get("conversation_id") and effective_ctx.conversation_id and self._state.metadata.get("conversation_id") != effective_ctx.conversation_id)
                or (self._state.metadata.get("project_id") and effective_ctx.project_id and self._state.metadata.get("project_id") != effective_ctx.project_id)
            )
        ):
            # Guard against cross-conversation and cross-project state pollution if an agent instance is reused
            self._state = BRDAgentState.initialize_from_template(sections=self.sections)

        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None) or str(uuid.uuid4())
        if effective_ctx.metadata is not None:
            effective_ctx.metadata["agent_run_id"] = agent_run_id
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id
        if effective_ctx.conversation_id:
            self._state.metadata["conversation_id"] = effective_ctx.conversation_id
        if getattr(effective_ctx, "project_name", None):
            self._state.metadata["project_name"] = effective_ctx.project_name
        if getattr(effective_ctx, "project_description", None):
            self._state.metadata["project_description"] = effective_ctx.project_description
        if getattr(effective_ctx, "available_documents", None):
            self._state.metadata["available_documents"] = list(effective_ctx.available_documents)

        self._telemetry_tracker = LLMTelemetryTracker(run_id=agent_run_id)
        telemetry_token = set_current_telemetry_tracker(self._telemetry_tracker)

        input_prompt = request.input_text if isinstance(request, AgentRunRequest) else (str(request) if request is not None else None)
        self._ingest_user_conversation_evidence(input_prompt=input_prompt, prior_messages=prior_messages)

        effective_consolidate = consolidate_clarification or bool(self._state.metadata.get("consolidate_clarification", False))
        is_consolidated_resume = bool(self._state.metadata.pop("pending_consolidated_clarification", None))
        is_resuming = (
            self._state.is_waiting_for_user
            or bool(self._state.pending_clarification)
            or bool(self._state.metadata.get("pending_clarification_section"))
            or is_consolidated_resume
        )
        if (is_resuming or (self._state.current_section and not self.is_section_processing_complete)) and self._state.objective:
            effective_obj = objective or self._state.objective
        else:
            effective_obj = objective or input_prompt or self._state.objective or "Generate Business Requirements Document"
        self._state.objective = effective_obj

        if current_task is not None:
            self._state.current_task = current_task

        # DEF-013: Fail explicitly if RAG capability is unavailable for grounded BRD generation
        if not self.has_rag_capability:
            logger.error(
                "BRD workflow aborted: RAG capability is unavailable. Grounded BRD generation requires project knowledge retrieval."
            )
            return AgentRunResponse(
                output_text="Error: Project knowledge retrieval (RAG) is unavailable. Grounded BRD generation cannot proceed without access to project documentation.",
                context=effective_ctx,
                state=self._state,
                success=False,
                error="RAG capability unavailable for grounded BRD generation",
            )

        is_section_resume = False
        if is_consolidated_resume:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="WORKFLOW_RESUMING_FROM_CONSOLIDATED_CLARIFICATION",
            )
            if input_prompt:
                self.receive_user_clarification(
                    answer=input_prompt,
                    context=effective_ctx,
                )
            unresolved_gaps = self._state.metadata.pop("unresolved_section_gaps", {})
            for sec in unresolved_gaps:
                existing = self._state.get_section_content(sec)
                update_res = await self.update_section_async(
                    section=sec,
                    rework_feedback=f"Incorporate user clarification: {input_prompt}",
                    existing_content=existing,
                    context=effective_ctx,
                )
                if update_res.content:
                    self._state.set_section_content(sec, update_res.content)
                val_res = await self.validate_section_async(
                    section=sec,
                    section_content=update_res.content or existing or "",
                    context=effective_ctx,
                )
                self._state.update_section_status(sec, BRDSectionStatus.COMPLETED)
            self._state.unresolved_information.clear()
            self._state.clear_waiting_for_user()
        elif is_resuming:
            # Resuming controlled BRD workflow after user clarification
            pending_q = self._state.pending_clarification
            pending_sec = self._state.metadata.pop("pending_clarification_section", None)
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="WORKFLOW_RESUMING_FROM_CLARIFICATION",
                resolved_item=pending_q,
                section=pending_sec,
            )
            if input_prompt:
                self.receive_user_clarification(
                    answer=input_prompt,
                    resolved_item=pending_q,
                    context=effective_ctx,
                )
            if pending_sec:
                self._state.set_current_section(pending_sec)
                is_section_resume = True
            elif self._state.current_section and not self.is_section_processing_complete:
                is_section_resume = True

        if is_section_resume:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="WORKFLOW_RESUMED_AT_SECTION",
                section=self._state.current_section,
                objective=effective_obj,
            )
        else:
            if not is_resuming:
                # Phase 1: Initial Context & Objective
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="WORKFLOW_PHASE_STARTED",
                    phase="1_INITIAL_CONTEXT",
                    objective=effective_obj,
                )

                # Phase 2: Action Decision (Lead Agent Intelligence)
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="WORKFLOW_PHASE_STARTED",
                    phase="2_ACTION_DECISION",
                )
                action_decision = await self.decide_action_async(
                    objective=effective_obj,
                    context=effective_ctx,
                    input_text=input_prompt,
                )

                # Phase 3: Action Execution (Application executes selected action)
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="WORKFLOW_PHASE_STARTED",
                    phase="3_ACTION_EXECUTION",
                    action_type=action_decision.action_type.value,
                )
                if action_decision.action_type == ActionType.RAG and self.has_rag_capability:
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.TOOL,
                        event_name="RAG_SEARCH_STARTED",
                        query=action_decision.query,
                    )
                    rag_resp, _ = self.retry_with_rag(
                        query=action_decision.query,
                        context=effective_ctx,
                        evaluate_after=False,
                    )
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.TOOL,
                        event_name="RAG_SEARCH_COMPLETED",
                        success=rag_resp.success,
                    )
                elif action_decision.action_type == ActionType.DELEGATION:
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.APPLICATION,
                        event_name="DELEGATION_STARTED",
                    )
                    await self.delegate_async(
                        tasks=action_decision.delegated_tasks or effective_obj,
                        context=effective_ctx,
                        objective=effective_obj,
                    )
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.APPLICATION,
                        event_name="DELEGATION_COMPLETED",
                    )
                else:
                    await self.perform_direct_work_async(
                        objective=action_decision.direct_work_objective or effective_obj,
                        instructions=action_decision.direct_work_instructions,
                        context=effective_ctx,
                    )

            # Phase 4: Evidence Evaluation & Lead Agent Interpretation
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="WORKFLOW_PHASE_STARTED",
                phase="4_EVIDENCE_EVALUATION",
            )
            eval_result = await self.evaluate_async(context=effective_ctx)
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.SPECIALIZED_AGENT,
                event_name="EVALUATION_COMPLETED",
                outcome=eval_result.outcome.value,
                gaps_count=len(eval_result.missing_information) + len(eval_result.unresolved_information),
            )

            gap_decision = await self.interpret_evaluation_async(
                evaluation_result=eval_result,
                objective=effective_obj,
                context=effective_ctx,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="EVALUATION_INTERPRETED",
                action=gap_decision.action.value,
                reasoning=gap_decision.reasoning,
            )

            if gap_decision.action == GapResolutionAction.RAG and self.has_rag_capability:
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="RAG_GAP_RECOVERY_STARTED",
                    query=gap_decision.query,
                )
                rag_resp, eval_result2 = self.retry_with_rag(
                    query=gap_decision.query,
                    context=effective_ctx,
                    evaluate_after=True,
                )
                if eval_result2:
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.SPECIALIZED_AGENT,
                        event_name="REEVALUATION_COMPLETED",
                        outcome=eval_result2.outcome.value,
                    )
                    if eval_result2.outcome != EvaluationOutcome.SUFFICIENT:
                        for gap in list(eval_result2.missing_information) + list(eval_result2.unresolved_information):
                            if is_substantive_business_clarification_item(gap):
                                self._state.add_unresolved(gap)
                        gap_decision = GapResolutionDecision(
                            action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
                            reasoning="Proceeding to section generation with available evidence; unresolved items recorded.",
                            identified_gaps=list(eval_result2.missing_information) + list(eval_result2.unresolved_information),
                        )

            if gap_decision.action == GapResolutionAction.ASK_USER:
                question = gap_decision.clarification_question or "Clarification required from user."
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="USER_CLARIFICATION_REQUIRED",
                    question=question,
                )
                self._state.set_waiting_for_user(question)
                return AgentRunResponse(
                    output_text=question,
                    context=effective_ctx,
                    state=self._state,
                    success=True,
                )

        # Phase 5: Section Iteration Loop (Sequential Processing across Template Sections)
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_PHASE_STARTED",
            phase="5_SECTION_ITERATION",
        )
        if not self._state.current_section and not self.is_section_processing_complete:
            self.initialize_section_progression(context=effective_ctx)

        while not self.is_section_processing_complete and self._state.current_section:
            cur_sec = self._state.current_section
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="SECTION_PROCESSING_STARTED",
                section=cur_sec,
            )

            # DEF-008: Retrieve section-specific project evidence prior to generation
            if self.has_rag_capability:
                await self.retrieve_section_evidence_async(
                    section=cur_sec,
                    context=effective_ctx,
                )

            gen_res = await self.generate_section_async(section=cur_sec, context=effective_ctx)
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.SPECIALIZED_AGENT,
                event_name="SECTION_GENERATED",
                section=cur_sec,
            )

            val_res = await self.validate_section_async(
                section=cur_sec,
                section_content=gen_res.content,
                context=effective_ctx,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.SPECIALIZED_AGENT,
                event_name="SECTION_VALIDATED",
                section=cur_sec,
                outcome=val_res.outcome.value,
            )

            rework_strategy = await self.interpret_section_validation_async(
                validation_result=val_res,
                section_name=cur_sec,
                context=effective_ctx,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="SECTION_VALIDATION_INTERPRETED",
                section=cur_sec,
                requires_rework=rework_strategy.requires_rework,
                failure_category=rework_strategy.failure_category.value if rework_strategy.failure_category else None,
            )

            if rework_strategy.requires_rework:
                for attempt in range(1, max_section_rework_attempts + 1):
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.APPLICATION,
                        event_name="SECTION_REWORK_STARTED",
                        section=cur_sec,
                        attempt=attempt,
                        max_attempts=max_section_rework_attempts,
                    )
                    # DEF-010: If rework requires additional evidence, retrieve it via RAG before regenerating
                    if rework_strategy.requires_retrieval and self.has_rag_capability and rework_strategy.retrieval_query:
                        log_trace_event(
                            logger,
                            agent_run_id=agent_run_id,
                            actor=TraceActor.APPLICATION,
                            event_name="SECTION_REWORK_RAG_RETRIEVAL_STARTED",
                            section=cur_sec,
                            query=rework_strategy.retrieval_query,
                            attempt=attempt,
                        )
                        await self.retrieve_section_evidence_async(
                            section=cur_sec,
                            query=rework_strategy.retrieval_query,
                            context=effective_ctx,
                        )

                    gen_res = await self.update_section_async(
                        section=cur_sec,
                        rework_feedback=rework_strategy.rework_guidance,
                        existing_content=gen_res.content,
                        context=effective_ctx,
                    )
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.SPECIALIZED_AGENT,
                        event_name="SECTION_UPDATED",
                        section=cur_sec,
                        attempt=attempt,
                    )
                    val_res = await self.validate_section_async(
                        section=cur_sec,
                        section_content=gen_res.content,
                        context=effective_ctx,
                    )
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.SPECIALIZED_AGENT,
                        event_name="SECTION_REVALIDATED",
                        section=cur_sec,
                        attempt=attempt,
                        outcome=val_res.outcome.value,
                    )
                    rework_strategy = await self.interpret_section_validation_async(
                        validation_result=val_res,
                        section_name=cur_sec,
                        context=effective_ctx,
                    )
                    log_trace_event(
                        logger,
                        agent_run_id=agent_run_id,
                        actor=TraceActor.LEAD_AGENT,
                        event_name="SECTION_VALIDATION_INTERPRETED",
                        section=cur_sec,
                        attempt=attempt,
                        requires_rework=rework_strategy.requires_rework,
                        failure_category=rework_strategy.failure_category.value if rework_strategy.failure_category else None,
                    )
                    if not rework_strategy.requires_rework:
                        break

            # Progression Gate: advance only if VALID
            if val_res.is_valid:
                self._state.update_section_status(cur_sec, BRDSectionStatus.COMPLETED)
                prog_res = self.progress_section(context=effective_ctx)
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="SECTION_PROGRESSED",
                    section=cur_sec,
                    next_section=self._state.current_section,
                )
            else:
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="SECTION_PROGRESSION_BLOCKED",
                    section=cur_sec,
                    reason="rework_limit_exhausted",
                    failure_category=(
                        rework_strategy.failure_category.value
                        if isinstance(rework_strategy.failure_category, SectionFailureCategory)
                        else rework_strategy.failure_category
                    ),
                )
                # DEF-011: Differentiated Failure Handling After Section Rework Exhaustion
                if rework_strategy.failure_category == SectionFailureCategory.EVIDENCE_GAP:
                    clarification_counts = self._state.metadata.setdefault("section_clarification_counts", {})
                    sec_clarification_count = clarification_counts.get(cur_sec, 0)
                    if sec_clarification_count >= 1:
                        # Bounded loop: clarification already attempted for this section, terminate with diagnostic
                        pass
                    elif effective_consolidate:
                        sections = list(self._state.template_sections or extract_brd_sections(load_brd_template()))
                        cur_idx = -1
                        for i, s in enumerate(sections):
                            if s == cur_sec or s.lower() == cur_sec.lower():
                                cur_idx = i
                                break
                        next_sec = sections[cur_idx + 1] if (cur_idx >= 0 and cur_idx + 1 < len(sections)) else None

                        # Record genuine substantive gap, preserve draft
                        substantive_items = [
                            item for item in rework_strategy.missing_evidence_items
                            if is_substantive_business_clarification_item(item)
                        ]
                        for item in substantive_items:
                            self._state.add_unresolved(item)
                        if substantive_items:
                            self._state.metadata.setdefault("unresolved_section_gaps", {}).setdefault(cur_sec, []).extend(substantive_items)
                        if gen_res.content:
                            self._state.set_section_content(cur_sec, gen_res.content)
                        self._state.update_section_status(cur_sec, BRDSectionStatus.COMPLETED)

                        if next_sec is not None:
                            self._state.set_current_section(next_sec, auto_in_progress=True)
                            log_trace_event(
                                logger,
                                agent_run_id=agent_run_id,
                                actor=TraceActor.APPLICATION,
                                event_name="SECTION_DRAFTED_WITH_UNRESOLVED_GAPS",
                                section=cur_sec,
                                next_section=next_sec,
                                unresolved_count=len(substantive_items),
                            )
                            continue
                        else:
                            # Last section of drafting pass reached; stop section loop to enter consolidation gate
                            break
                    else:
                        clarification_counts[cur_sec] = sec_clarification_count + 1
                        clarification_q = (
                            rework_strategy.clarification_question
                            or self.construct_section_clarification_question(
                                section_name=cur_sec,
                                validation_result=val_res,
                                missing_items=rework_strategy.missing_evidence_items,
                            )
                        )
                        self._state.set_waiting_for_user(clarification_q)
                        self._state.metadata["pending_clarification_section"] = cur_sec
                        self._state.update_section_status(cur_sec, BRDSectionStatus.NEEDS_REVISION)

                        log_trace_event(
                            logger,
                            agent_run_id=agent_run_id,
                            actor=TraceActor.APPLICATION,
                            event_name="USER_CLARIFICATION_REQUIRED",
                            section=cur_sec,
                            question=clarification_q,
                        )
                        return AgentRunResponse(
                            output_text=clarification_q,
                            context=effective_ctx,
                            state=self._state,
                            success=True,
                        )

                # For GENERATION_QUALITY, TEMPLATE_CONFIGURATION, or exhausted clarification:
                diagnostics, diag_text = self._format_section_failure_diagnostic(
                    section_name=cur_sec,
                    failure_category=rework_strategy.failure_category or SectionFailureCategory.GENERATION_QUALITY,
                    validation_result=val_res,
                )
                self._state.metadata["failure_diagnostics"] = diagnostics
                self._state.update_section_status(cur_sec, BRDSectionStatus.NEEDS_REVISION)

                return AgentRunResponse(
                    output_text=diag_text,
                    success=False,
                    error=f"Section rework exhausted for '{cur_sec}' due to {diagnostics['failure_category']}",
                    context=effective_ctx,
                    state=self._state,
                    diagnostics=diagnostics,
                )

        # Phase 6: Document Assembly Gate
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_PHASE_STARTED",
            phase="6_DOCUMENT_ASSEMBLY",
        )
        if not self.is_section_processing_complete:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="DOCUMENT_ASSEMBLY_BLOCKED",
                reason="sections_incomplete",
                remaining_sections=len(self.get_remaining_sections()),
            )
            return AgentRunResponse(
                output_text="BRD workflow paused: not all template sections could be completed.",
                success=False,
                context=effective_ctx,
                state=self._state,
            )

        assembly_res = await self.assemble_brd_async(context=effective_ctx)
        if getattr(assembly_res, "assembled_document", None):
            self._state.set_assembled_brd(assembly_res.assembled_document)
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="DOCUMENT_ASSEMBLED",
            success=getattr(assembly_res, "assembly_complete", getattr(assembly_res, "is_complete", True)),
        )

        # Phase 7: Final Validation
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_PHASE_STARTED",
            phase="7_FINAL_VALIDATION",
        )
        final_val_res = await self.validate_final_brd_async(context=effective_ctx, auto_recover=False)
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.SPECIALIZED_AGENT,
            event_name="FINAL_VALIDATION_COMPLETED",
            outcome=final_val_res.outcome.value,
        )

        # Phase 8: Final Validation Interpretation & Bounded Recovery Loop
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_PHASE_STARTED",
            phase="8_FINAL_RECOVERY",
        )
        cycle = 0
        while not final_val_res.is_valid and cycle < max_recovery_cycles:
            recovery_strategy = await self.interpret_final_validation_async(
                final_validation_result=final_val_res,
                context=effective_ctx,
            )
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.LEAD_AGENT,
                event_name="FINAL_VALIDATION_INTERPRETED",
                cycle=cycle + 1,
                requires_recovery=recovery_strategy.requires_recovery,
                affected_sections=recovery_strategy.affected_sections,
            )
            if not recovery_strategy.requires_recovery or not recovery_strategy.affected_sections:
                break

            cycle += 1
            self._state.increment_final_validation_recovery_cycle()
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="FINAL_RECOVERY_CYCLE_STARTED",
                cycle=cycle,
                max_cycles=max_recovery_cycles,
            )

            cycle_sections_passed = True
            failed_sec_name = ""
            for sec in recovery_strategy.affected_sections:
                sec_guidance = recovery_strategy.section_guidance.get(sec, "")
                sec_passed = False
                for attempt in range(1, max_section_rework_attempts + 1):
                    await asyncio.sleep(1.0)
                    gen_res = await self.update_section_async(
                        section=sec,
                        rework_feedback=sec_guidance,
                        existing_content=self._state.get_section_content(sec),
                        context=effective_ctx,
                    )
                    val_res = await self.validate_section_async(
                        section=sec,
                        section_content=gen_res.content,
                        prior_rework_feedback=sec_guidance,
                        context=effective_ctx,
                    )
                    if val_res.is_valid:
                        self._state.update_section_status(sec, BRDSectionStatus.COMPLETED)
                        sec_passed = True
                        break
                    else:
                        sec_guidance = val_res.rework_feedback or sec_guidance

                if not sec_passed:
                    cycle_sections_passed = False
                    failed_sec_name = sec
                    logger.warning(
                        "Recovery rework exhausted for section '%s' after %d attempts; section remains invalid.",
                        sec,
                        max_section_rework_attempts,
                    )
                    break

            if not cycle_sections_passed:
                self._state.final_validation_recovery_exhausted = True
                diag = {
                    "phase": "8_FINAL_RECOVERY",
                    "failed_section": failed_sec_name,
                    "cycle": cycle,
                    "error": f"Section '{failed_sec_name}' could not reach a valid state after {max_section_rework_attempts} rework attempts.",
                }
                self._state.metadata["recovery_failure"] = diag
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="FINAL_RECOVERY_FAILED",
                    diagnostics=diag,
                )
                if self._telemetry_tracker:
                    self._state.metadata["llm_execution_summary"] = self._telemetry_tracker.get_summary().to_dict()
                return AgentRunResponse(
                    output_text=f"BRD recovery halted: section '{failed_sec_name}' could not reach a valid state after {max_section_rework_attempts} rework attempts. Assembly aborted to preserve document integrity.",
                    success=False,
                    context=effective_ctx,
                    state=self._state,
                )

            # INVARIANT: DO NOT assemble unless ALL sections are completed
            if not self.is_section_processing_complete:
                self._state.final_validation_recovery_exhausted = True
                logger.error("Cannot assemble BRD in recovery: not all template sections are completed.")
                if self._telemetry_tracker:
                    self._state.metadata["llm_execution_summary"] = self._telemetry_tracker.get_summary().to_dict()
                return AgentRunResponse(
                    output_text="Cannot assemble BRD in recovery: not all template sections are completed.",
                    success=False,
                    context=effective_ctx,
                    state=self._state,
                )

            await self.assemble_brd_async(context=effective_ctx)
            final_val_res = await self.validate_final_brd_async(context=effective_ctx, auto_recover=False)
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.SPECIALIZED_AGENT,
                event_name="FINAL_REVALIDATION_COMPLETED",
                cycle=cycle,
                outcome=final_val_res.outcome.value,
            )

        if not final_val_res.is_valid and cycle >= max_recovery_cycles:
            self._state.final_validation_recovery_exhausted = True
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="FINAL_RECOVERY_EXHAUSTED",
                cycles=cycle,
            )

        # Phase 9: Completion
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_PHASE_STARTED",
            phase="9_COMPLETION",
        )
        final_doc = self._state.assembled_brd or "BRD generation completed."
        final_output = final_doc

        # Continuation mechanism: if substantive unresolved information exists, ask consolidated clarification after delivering BRD V1
        substantive_unresolved = [
            u for u in self._state.unresolved_information
            if is_substantive_business_clarification_item(u)
        ]
        if substantive_unresolved and not self._state.metadata.get("clarification_gate_completed"):
            self._state.metadata["clarification_gate_completed"] = True
            clarification_q = self.construct_consolidated_clarification_question(substantive_unresolved)
            self._state.set_waiting_for_user(clarification_q)
            self._state.metadata["pending_consolidated_clarification"] = True
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="USER_CONSOLIDATED_CLARIFICATION_REQUIRED",
                question=clarification_q,
                unresolved_items=substantive_unresolved,
            )
            final_output = f"{final_doc}\n\n---\n\n{clarification_q}"

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_COMPLETED",
            success=True,
            is_valid=final_val_res.is_valid,
        )

        if self._telemetry_tracker:
            self._state.metadata["llm_execution_summary"] = self._telemetry_tracker.get_summary().to_dict()
            logger.info(self._telemetry_tracker.format_log_summary())

        return AgentRunResponse(
            output_text=final_output,
            context=effective_ctx,
            model=getattr(self.model, "model_name", str(self.model)),
            success=True,
            state=self._state,
        )

    def run_workflow(
        self,
        request: Optional[AgentRunRequest | str] = None,
        context: Optional[AgentContext] = None,
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        max_section_rework_attempts: int = 2,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
        initial_state: Optional[BRDAgentState] = None,
        prior_messages: Optional[Sequence[Any]] = None,
        consolidate_clarification: bool = False,
    ) -> AgentRunResponse:
        """Execute the Application-Owned BRD Workflow lifecycle synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.run_workflow_async(
                        request=request,
                        context=context,
                        objective=objective,
                        current_task=current_task,
                        max_section_rework_attempts=max_section_rework_attempts,
                        max_recovery_cycles=max_recovery_cycles,
                        initial_state=initial_state,
                        prior_messages=prior_messages,
                        consolidate_clarification=consolidate_clarification,
                    ),
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self.run_workflow_async(
                    request=request,
                    context=context,
                    objective=objective,
                    current_task=current_task,
                    max_section_rework_attempts=max_section_rework_attempts,
                    max_recovery_cycles=max_recovery_cycles,
                    initial_state=initial_state,
                    prior_messages=prior_messages,
                    consolidate_clarification=consolidate_clarification,
                )
            )

    async def stream_workflow_async(
        self,
        request: Optional[AgentRunRequest | str] = None,
        context: Optional[AgentContext] = None,
        objective: Optional[str] = None,
        current_task: Optional[str] = None,
        max_section_rework_attempts: int = 2,
        max_recovery_cycles: int = MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
        initial_state: Optional[BRDAgentState] = None,
        prior_messages: Optional[Sequence[Any]] = None,
        consolidate_clarification: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream progress events and document content throughout the BRD workflow lifecycle."""
        effective_ctx = request.context if isinstance(request, AgentRunRequest) else context
        if effective_ctx is None:
            effective_ctx = AgentContext(project_id=self.project_id)
        elif not effective_ctx.project_id and self.project_id:
            effective_ctx = AgentContext(
                project_id=self.project_id,
                conversation_id=effective_ctx.conversation_id,
                user_id=effective_ctx.user_id,
                project_name=getattr(effective_ctx, "project_name", None),
                project_description=getattr(effective_ctx, "project_description", None),
                available_documents=list(getattr(effective_ctx, "available_documents", [])),
                metadata=effective_ctx.metadata,
            )

        if initial_state is not None:
            self._state = initial_state
        elif isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)
        elif (
            effective_ctx and (
                (self._state.metadata.get("conversation_id") and effective_ctx.conversation_id and self._state.metadata.get("conversation_id") != effective_ctx.conversation_id)
                or (self._state.metadata.get("project_id") and effective_ctx.project_id and self._state.metadata.get("project_id") != effective_ctx.project_id)
            )
        ):
            # Guard against cross-conversation and cross-project state pollution if an agent instance is reused
            self._state = BRDAgentState.initialize_from_template(sections=self.sections)

        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None) or str(uuid.uuid4())
        if effective_ctx.metadata is not None:
            effective_ctx.metadata["agent_run_id"] = agent_run_id
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id
        if effective_ctx.conversation_id:
            self._state.metadata["conversation_id"] = effective_ctx.conversation_id
        if getattr(effective_ctx, "project_name", None):
            self._state.metadata["project_name"] = effective_ctx.project_name
        if getattr(effective_ctx, "project_description", None):
            self._state.metadata["project_description"] = effective_ctx.project_description
        if getattr(effective_ctx, "available_documents", None):
            self._state.metadata["available_documents"] = list(effective_ctx.available_documents)

        self._telemetry_tracker = LLMTelemetryTracker(run_id=agent_run_id)
        telemetry_token = set_current_telemetry_tracker(self._telemetry_tracker)

        input_prompt = request.input_text if isinstance(request, AgentRunRequest) else (str(request) if request is not None else None)
        self._ingest_user_conversation_evidence(input_prompt=input_prompt, prior_messages=prior_messages)

        effective_consolidate = consolidate_clarification or bool(self._state.metadata.get("consolidate_clarification", False))
        is_consolidated_resume = bool(self._state.metadata.pop("pending_consolidated_clarification", None))
        is_resuming = (
            self._state.is_waiting_for_user
            or bool(self._state.pending_clarification)
            or bool(self._state.metadata.get("pending_clarification_section"))
            or is_consolidated_resume
        )
        if (is_resuming or (self._state.current_section and not self.is_section_processing_complete)) and self._state.objective:
            effective_obj = objective or self._state.objective
        else:
            effective_obj = objective or input_prompt or self._state.objective or "Generate Business Requirements Document"
        self._state.objective = effective_obj

        if current_task is not None:
            self._state.current_task = current_task

        # DEF-013: Fail explicitly if RAG capability is unavailable for grounded BRD generation
        if not self.has_rag_capability:
            logger.error("BRD workflow aborted: RAG capability is unavailable.")
            yield {
                "type": "progress",
                "phase": "1_INITIAL_CONTEXT",
                "actor": TraceActor.APPLICATION,
                "message": "Error: RAG capability is unavailable. Grounded BRD generation cannot proceed.",
            }
            yield {
                "type": "content",
                "content": "Error: Project knowledge retrieval (RAG) is unavailable. Grounded BRD generation cannot proceed without access to project documentation.",
            }
            return

        is_section_resume = False
        if is_consolidated_resume:
            yield {
                "type": "progress",
                "phase": "5_SECTION_ITERATION",
                "actor": TraceActor.APPLICATION,
                "message": "Incorporating user clarification and updating affected sections...",
            }
            if input_prompt:
                self.receive_user_clarification(
                    answer=input_prompt,
                    context=effective_ctx,
                )
            unresolved_gaps = self._state.metadata.pop("unresolved_section_gaps", {})
            for sec in unresolved_gaps:
                yield {
                    "type": "progress",
                    "phase": "5_SECTION_ITERATION",
                    "actor": TraceActor.APPLICATION,
                    "section": sec,
                    "message": f"Updating section with user clarification: {sec}",
                }
                existing = self._state.get_section_content(sec)
                update_res = await self.update_section_async(
                    section=sec,
                    rework_feedback=f"Incorporate user clarification: {input_prompt}",
                    existing_content=existing,
                    context=effective_ctx,
                )
                if update_res.content:
                    self._state.set_section_content(sec, update_res.content)
                val_res = await self.validate_section_async(
                    section=sec,
                    section_content=update_res.content or existing or "",
                    context=effective_ctx,
                )
                self._state.update_section_status(sec, BRDSectionStatus.COMPLETED)
            self._state.unresolved_information.clear()
            self._state.clear_waiting_for_user()
        elif is_resuming:
            pending_q = self._state.pending_clarification
            pending_sec = self._state.metadata.pop("pending_clarification_section", None)
            if input_prompt:
                self.receive_user_clarification(
                    answer=input_prompt,
                    resolved_item=pending_q,
                    context=effective_ctx,
                )
            if pending_sec:
                self._state.set_current_section(pending_sec)
                is_section_resume = True
            elif self._state.current_section and not self.is_section_processing_complete:
                is_section_resume = True

        if is_section_resume:
            yield {
                "type": "progress",
                "phase": "5_SECTION_ITERATION",
                "actor": TraceActor.APPLICATION,
                "section": self._state.current_section,
                "message": f"Resuming workflow at section: {self._state.current_section} with user clarification",
            }
        else:
            if is_resuming:
                yield {
                    "type": "progress",
                    "phase": "4_EVIDENCE_EVALUATION",
                    "actor": TraceActor.APPLICATION,
                    "message": "Incorporating user clarification and re-evaluating evidence sufficiency...",
                }
            else:
                yield {
                    "type": "progress",
                    "phase": "1_INITIAL_CONTEXT",
                    "actor": TraceActor.APPLICATION,
                    "message": f"Starting BRD workflow for objective: {effective_obj}",
                }

                yield {
                    "type": "progress",
                    "phase": "2_ACTION_DECISION",
                    "actor": TraceActor.LEAD_AGENT,
                    "message": "Analyzing objective and determining action strategy...",
                }
                action_decision = await self.decide_action_async(objective=effective_obj, context=effective_ctx, input_text=input_prompt)

                yield {
                    "type": "progress",
                    "phase": "3_ACTION_EXECUTION",
                    "actor": TraceActor.APPLICATION,
                    "message": f"Executing action: {action_decision.action_type.value}",
                }
                if action_decision.action_type == ActionType.RAG and self.has_rag_capability:
                    self.retry_with_rag(query=action_decision.query, context=effective_ctx, evaluate_after=False)
                elif action_decision.action_type == ActionType.DELEGATION:
                    await self.delegate_async(tasks=action_decision.delegated_tasks or effective_obj, context=effective_ctx)
                else:
                    await self.perform_direct_work_async(objective=action_decision.direct_work_objective or effective_obj, context=effective_ctx)

            yield {
                "type": "progress",
                "phase": "4_EVIDENCE_EVALUATION",
                "actor": TraceActor.SPECIALIZED_AGENT,
                "message": "Evaluating evidence sufficiency for BRD requirements...",
            }
            eval_result = await self.evaluate_async(context=effective_ctx)
            gap_decision = await self.interpret_evaluation_async(evaluation_result=eval_result, objective=effective_obj, context=effective_ctx)

            if gap_decision.action == GapResolutionAction.RAG and self.has_rag_capability:
                yield {
                    "type": "progress",
                    "phase": "4_EVIDENCE_EVALUATION",
                    "actor": TraceActor.APPLICATION,
                    "message": "Retrieving missing specifications via project knowledge base...",
                }
                _, eval_result2 = self.retry_with_rag(query=gap_decision.query, context=effective_ctx, evaluate_after=True)
                if eval_result2 and eval_result2.outcome != EvaluationOutcome.SUFFICIENT:
                    for gap in list(eval_result2.missing_information) + list(eval_result2.unresolved_information):
                        if is_substantive_business_clarification_item(gap):
                            self._state.add_unresolved(gap)
                    gap_decision = GapResolutionDecision(
                        action=GapResolutionAction.PROCEED_TO_SECTION_GENERATION,
                        reasoning="Proceeding to section generation with available evidence; unresolved items recorded.",
                        identified_gaps=list(eval_result2.missing_information) + list(eval_result2.unresolved_information),
                    )

            if gap_decision.action == GapResolutionAction.ASK_USER:
                question = gap_decision.clarification_question or "Clarification required from user."
                self._state.set_waiting_for_user(question)
                yield {
                    "type": "progress",
                    "phase": "4_EVIDENCE_EVALUATION",
                    "actor": TraceActor.APPLICATION,
                    "message": f"Clarification requested: {question}",
                }
                yield {
                    "type": "content",
                    "content": question,
                }
                return

        # Phase 5: Section Iteration Loop
        if not self._state.current_section and not self.is_section_processing_complete:
            self.initialize_section_progression(context=effective_ctx)

        while not self.is_section_processing_complete and self._state.current_section:
            cur_sec = self._state.current_section
            yield {
                "type": "progress",
                "phase": "5_SECTION_ITERATION",
                "actor": TraceActor.APPLICATION,
                "section": cur_sec,
                "message": f"Drafting section: {cur_sec}",
            }

            # DEF-008: Retrieve section-specific project evidence prior to generation
            if self.has_rag_capability:
                yield {
                    "type": "progress",
                    "phase": "5_SECTION_ITERATION",
                    "actor": TraceActor.APPLICATION,
                    "section": cur_sec,
                    "message": f"Retrieving project knowledge for section: {cur_sec}",
                }
                await self.retrieve_section_evidence_async(
                    section=cur_sec,
                    context=effective_ctx,
                )

            gen_res = await self.generate_section_async(section=cur_sec, context=effective_ctx)
            val_res = await self.validate_section_async(section=cur_sec, section_content=gen_res.content, context=effective_ctx)
            rework_strategy = await self.interpret_section_validation_async(validation_result=val_res, section_name=cur_sec, context=effective_ctx)

            if rework_strategy.requires_rework:
                for attempt in range(1, max_section_rework_attempts + 1):
                    yield {
                        "type": "progress",
                        "phase": "5_SECTION_ITERATION",
                        "actor": TraceActor.APPLICATION,
                        "section": cur_sec,
                        "message": f"Reworking section: {cur_sec} (attempt {attempt}/{max_section_rework_attempts})",
                    }
                    # DEF-010: Retrieve additional evidence if required
                    if rework_strategy.requires_retrieval and self.has_rag_capability and rework_strategy.retrieval_query:
                        yield {
                            "type": "progress",
                            "phase": "5_SECTION_ITERATION",
                            "actor": TraceActor.APPLICATION,
                            "section": cur_sec,
                            "message": f"Retrieving additional project knowledge for section rework: {cur_sec}...",
                        }
                        await self.retrieve_section_evidence_async(
                            section=cur_sec,
                            query=rework_strategy.retrieval_query,
                            context=effective_ctx,
                        )

                    gen_res = await self.update_section_async(section=cur_sec, rework_feedback=rework_strategy.rework_guidance, existing_content=gen_res.content, context=effective_ctx)
                    val_res = await self.validate_section_async(section=cur_sec, section_content=gen_res.content, context=effective_ctx)
                    rework_strategy = await self.interpret_section_validation_async(validation_result=val_res, section_name=cur_sec, context=effective_ctx)
                    if not rework_strategy.requires_rework:
                        break

            if val_res.is_valid:
                self._state.update_section_status(cur_sec, BRDSectionStatus.COMPLETED)
                self.progress_section(context=effective_ctx)
                yield {
                    "type": "progress",
                    "phase": "5_SECTION_ITERATION",
                    "actor": TraceActor.APPLICATION,
                    "section": cur_sec,
                    "message": f"Section completed: {cur_sec}",
                }
            else:
                # DEF-011: Differentiated Failure Handling After Section Rework Exhaustion
                if rework_strategy.failure_category == SectionFailureCategory.EVIDENCE_GAP:
                    clarification_counts = self._state.metadata.setdefault("section_clarification_counts", {})
                    sec_clarification_count = clarification_counts.get(cur_sec, 0)
                    if sec_clarification_count >= 1:
                        # Bounded loop: clarification already attempted for this section, terminate with diagnostic
                        pass
                    elif effective_consolidate:
                        sections = list(self._state.template_sections or extract_brd_sections(load_brd_template()))
                        cur_idx = -1
                        for i, s in enumerate(sections):
                            if s == cur_sec or s.lower() == cur_sec.lower():
                                cur_idx = i
                                break
                        next_sec = sections[cur_idx + 1] if (cur_idx >= 0 and cur_idx + 1 < len(sections)) else None

                        # Record genuine substantive gap, preserve draft
                        substantive_items = [
                            item for item in rework_strategy.missing_evidence_items
                            if is_substantive_business_clarification_item(item)
                        ]
                        for item in substantive_items:
                            self._state.add_unresolved(item)
                        if substantive_items:
                            self._state.metadata.setdefault("unresolved_section_gaps", {}).setdefault(cur_sec, []).extend(substantive_items)
                        if gen_res.content:
                            self._state.set_section_content(cur_sec, gen_res.content)
                        self._state.update_section_status(cur_sec, BRDSectionStatus.COMPLETED)

                        if next_sec is not None:
                            self._state.set_current_section(next_sec, auto_in_progress=True)
                            yield {
                                "type": "progress",
                                "phase": "5_SECTION_ITERATION",
                                "actor": TraceActor.APPLICATION,
                                "section": cur_sec,
                                "message": f"Section drafted with unresolved business gaps deferred to consolidation: {cur_sec}",
                            }
                            continue
                        else:
                            # Last section of drafting pass reached; continue to assembly
                            break
                    else:
                        clarification_counts[cur_sec] = sec_clarification_count + 1
                        clarification_q = (
                            rework_strategy.clarification_question
                            or self.construct_section_clarification_question(
                                section_name=cur_sec,
                                validation_result=val_res,
                                missing_items=rework_strategy.missing_evidence_items,
                            )
                        )
                        self._state.set_waiting_for_user(clarification_q)
                        self._state.metadata["pending_clarification_section"] = cur_sec
                        self._state.update_section_status(cur_sec, BRDSectionStatus.NEEDS_REVISION)

                        yield {
                            "type": "progress",
                            "phase": "5_SECTION_ITERATION",
                            "actor": TraceActor.APPLICATION,
                            "section": cur_sec,
                            "message": f"Clarification requested for section '{cur_sec}': {clarification_q}",
                        }
                        yield {"type": "content", "content": clarification_q}
                        return

                diagnostics, diag_text = self._format_section_failure_diagnostic(
                    section_name=cur_sec,
                    failure_category=rework_strategy.failure_category or SectionFailureCategory.GENERATION_QUALITY,
                    validation_result=val_res,
                )
                self._state.metadata["failure_diagnostics"] = diagnostics
                self._state.update_section_status(cur_sec, BRDSectionStatus.NEEDS_REVISION)

                yield {
                    "type": "progress",
                    "phase": "5_SECTION_ITERATION",
                    "actor": TraceActor.APPLICATION,
                    "section": cur_sec,
                    "message": f"Section progression halted for '{cur_sec}': {diagnostics['failure_category']}",
                }
                yield {"type": "content", "content": diag_text}
                return

        yield {
            "type": "progress",
            "phase": "6_DOCUMENT_ASSEMBLY",
            "actor": TraceActor.APPLICATION,
            "message": "Assembling complete Business Requirements Document...",
        }
        assembly_res = await self.assemble_brd_async(context=effective_ctx)
        if getattr(assembly_res, "assembled_document", None):
            self._state.set_assembled_brd(assembly_res.assembled_document)

        yield {
            "type": "progress",
            "phase": "7_FINAL_VALIDATION",
            "actor": TraceActor.SPECIALIZED_AGENT,
            "message": "Executing document-level final validation...",
        }
        final_val_res = await self.validate_final_brd_async(context=effective_ctx, auto_recover=False)

        cycle = 0
        while not final_val_res.is_valid and cycle < max_recovery_cycles:
            recovery_strategy = await self.interpret_final_validation_async(final_validation_result=final_val_res, context=effective_ctx)
            if not recovery_strategy.requires_recovery or not recovery_strategy.affected_sections:
                break
            cycle += 1
            self._state.increment_final_validation_recovery_cycle()
            yield {
                "type": "progress",
                "phase": "8_FINAL_RECOVERY",
                "actor": TraceActor.APPLICATION,
                "message": f"Executing recovery cycle {cycle}/{max_recovery_cycles} for sections: {', '.join(recovery_strategy.affected_sections)}",
            }
            cycle_sections_passed = True
            failed_sec_name = ""
            for sec in recovery_strategy.affected_sections:
                guidance = recovery_strategy.section_guidance.get(sec, "")
                sec_passed = False
                for attempt in range(1, max_section_rework_attempts + 1):
                    yield {
                        "type": "progress",
                        "phase": "8_FINAL_RECOVERY",
                        "actor": TraceActor.APPLICATION,
                        "section": sec,
                        "message": f"Updating and validating affected section: {sec} (attempt {attempt}/{max_section_rework_attempts})",
                    }
                    await asyncio.sleep(1.0)
                    gen_res = await self.update_section_async(
                        section=sec,
                        rework_feedback=guidance,
                        existing_content=self._state.get_section_content(sec),
                        context=effective_ctx,
                    )
                    val_res = await self.validate_section_async(
                        section=sec,
                        section_content=gen_res.content,
                        prior_rework_feedback=guidance,
                        context=effective_ctx,
                    )
                    if val_res.is_valid:
                        self._state.update_section_status(sec, BRDSectionStatus.COMPLETED)
                        sec_passed = True
                        yield {
                            "type": "progress",
                            "phase": "8_FINAL_RECOVERY",
                            "actor": TraceActor.APPLICATION,
                            "section": sec,
                            "message": f"Section '{sec}' successfully validated in recovery (attempt {attempt}).",
                        }
                        break
                    else:
                        guidance = val_res.rework_feedback or guidance

                if not sec_passed:
                    cycle_sections_passed = False
                    failed_sec_name = sec
                    logger.warning(
                        "Recovery rework exhausted for section '%s' after %d attempts; section remains invalid.",
                        sec,
                        max_section_rework_attempts,
                    )
                    break

            if not cycle_sections_passed:
                self._state.final_validation_recovery_exhausted = True
                diag = {
                    "phase": "8_FINAL_RECOVERY",
                    "failed_section": failed_sec_name,
                    "cycle": cycle,
                    "error": f"Section '{failed_sec_name}' could not reach a valid state after {max_section_rework_attempts} rework attempts.",
                }
                self._state.metadata["recovery_failure"] = diag
                log_trace_event(
                    logger,
                    agent_run_id=agent_run_id,
                    actor=TraceActor.APPLICATION,
                    event_name="FINAL_RECOVERY_FAILED",
                    diagnostics=diag,
                )
                yield {
                    "type": "progress",
                    "phase": "8_FINAL_RECOVERY",
                    "actor": TraceActor.APPLICATION,
                    "section": failed_sec_name,
                    "message": f"Recovery rework exhausted for section '{failed_sec_name}': section remains invalid. Assembly aborted.",
                }
                yield {
                    "type": "content",
                    "content": f"\n\n**BRD Generation Paused**: Section '{failed_sec_name}' could not satisfy quality and compliance requirements after {max_section_rework_attempts} rework attempts. Document assembly aborted to maintain document integrity.",
                }
                if self._telemetry_tracker:
                    summary = self._telemetry_tracker.get_summary()
                    self._state.metadata["llm_execution_summary"] = summary.to_dict()
                    yield {
                        "type": "execution_summary",
                        "execution_summary": summary.to_dict(),
                    }
                return

            if not self.is_section_processing_complete:
                self._state.final_validation_recovery_exhausted = True
                logger.error("Cannot assemble BRD in recovery: not all template sections are completed.")
                yield {
                    "type": "progress",
                    "phase": "8_FINAL_RECOVERY",
                    "actor": TraceActor.APPLICATION,
                    "message": "Cannot assemble BRD in recovery: not all template sections are completed.",
                }
                if self._telemetry_tracker:
                    summary = self._telemetry_tracker.get_summary()
                    self._state.metadata["llm_execution_summary"] = summary.to_dict()
                    yield {
                        "type": "execution_summary",
                        "execution_summary": summary.to_dict(),
                    }
                return

            yield {
                "type": "progress",
                "phase": "6_DOCUMENT_ASSEMBLY",
                "actor": TraceActor.APPLICATION,
                "message": "Re-assembling complete Business Requirements Document...",
            }
            await self.assemble_brd_async(context=effective_ctx)
            final_val_res = await self.validate_final_brd_async(context=effective_ctx, auto_recover=False)

        yield {
            "type": "progress",
            "phase": "9_COMPLETION",
            "actor": TraceActor.APPLICATION,
            "message": "BRD workflow complete.",
        }
        final_doc = self._state.assembled_brd or "BRD generation completed."
        yield {"type": "content", "content": final_doc}

        substantive_unresolved = [
            u for u in self._state.unresolved_information
            if is_substantive_business_clarification_item(u)
        ]
        if substantive_unresolved and not self._state.metadata.get("clarification_gate_completed"):
            self._state.metadata["clarification_gate_completed"] = True
            clarification_q = self.construct_consolidated_clarification_question(substantive_unresolved)
            self._state.set_waiting_for_user(clarification_q)
            self._state.metadata["pending_consolidated_clarification"] = True
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="USER_CONSOLIDATED_CLARIFICATION_REQUIRED",
                question=clarification_q,
                unresolved_items=substantive_unresolved,
            )
            yield {
                "type": "progress",
                "phase": "9_COMPLETION",
                "actor": TraceActor.APPLICATION,
                "message": f"Consolidated clarification requested for continuation: {clarification_q}",
            }
            yield {"type": "content", "content": f"\n\n---\n\n{clarification_q}"}

        if self._telemetry_tracker:
            summary = self._telemetry_tracker.get_summary()
            self._state.metadata["llm_execution_summary"] = summary.to_dict()
            logger.info(self._telemetry_tracker.format_log_summary())
            yield {
                "type": "execution_summary",
                "execution_summary": summary.to_dict(),
            }


def create_brd_lead_agent(
    runtime: Optional[Any] = None,
    config: Optional[AgentConfig] = None,
    model: Optional[BaseChatModel] = None,
    tools: Optional[Sequence[Any]] = None,
    system_instruction: Optional[str] = None,
    system_prompt: Optional[str] = None,
    template: Optional[str] = None,
    state: Optional[BRDAgentState] = None,
    rag_service: Optional[RAGService] = None,
    enable_rag: bool = True,
    evaluator: Optional[BRDEvaluationAgent] = None,
    section_generator: Optional[BRDSectionGenerationAgent] = None,
    section_validator: Optional[BRDSectionValidationAgent] = None,
    final_validator: Optional[BRDFinalValidationAgent] = None,
    store: Optional[BaseStore] = None,
    project_id: Optional[str] = None,
    memory_sources: Optional[Sequence[str]] = None,
    enable_memory: bool = True,
) -> BRDLeadAgent:
    """Factory function to instantiate the BRD Lead Agent.

    Args:
        runtime: Deprecated parameter retained for backward compatibility.
        config: Optional AgentConfig instance.
        model: Optional pre-configured BaseChatModel.
        tools: Optional sequence of tools.
        system_instruction: Optional system instruction override.
        system_prompt: Deprecated alias for system_instruction for backward compatibility.
        template: Optional BRD template override.
        state: Optional BRDAgentState working state instance.
        rag_service: Optional pre-configured RAGService instance to inject for knowledge retrieval.
        enable_rag: Whether to equip RAG knowledge retrieval tool (defaults to True).
        evaluator: Optional pre-configured BRDEvaluationAgent instance.
        section_generator: Optional pre-configured BRDSectionGenerationAgent instance.
        section_validator: Optional pre-configured BRDSectionValidationAgent instance.
        final_validator: Optional pre-configured BRDFinalValidationAgent instance.
        store: Optional LangGraph BaseStore instance for long-term memory.
        project_id: Optional project identifier for project-scoped memory.
        memory_sources: Optional sequence of memory file paths to preload.
        enable_memory: Whether to enable long-term memory (defaults to True).

    Returns:
        Configured BRDLeadAgent instance.
    """
    return BRDLeadAgent(
        runtime=runtime,
        config=config,
        model=model,
        tools=tools,
        system_instruction=system_instruction,
        system_prompt=system_prompt,
        template=template,
        state=state,
        rag_service=rag_service,
        enable_rag=enable_rag,
        evaluator=evaluator,
        section_generator=section_generator,
        section_validator=section_validator,
        final_validator=final_validator,
        store=store,
        project_id=project_id,
        memory_sources=memory_sources,
        enable_memory=enable_memory,
    )
