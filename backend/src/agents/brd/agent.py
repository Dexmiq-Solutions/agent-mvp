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

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
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
    extract_brd_sections,
    extract_section_requirements,
    extract_section_template,
    get_brd_template_path,
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


@dataclass
class SectionReworkStrategy:
    """Strategy formulated by the Lead Agent upon interpreting section validation findings."""

    section_name: str
    requires_rework: bool
    rework_guidance: str = ""
    specific_adjustments: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "requires_rework": self.requires_rework,
            "rework_guidance": self.rework_guidance,
            "specific_adjustments": list(self.specific_adjustments),
            "reasoning": self.reasoning,
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
            prompt = (
                f"You are the BRD Lead Agent. Analyze the current objective and decide the best action.\n"
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
            prompt = (
                f"You are the BRD Lead Agent. Analyze the current objective and decide the best action.\n"
                f"Objective: {effective_obj}\n"
                f"Section: {section_name or self._state.current_section or 'General'}\n"
                f"Available tools: RAG equipped={self.has_rag_capability}\n\n"
                f"Choose exactly one action_type:\n"
                f"- 'rag': Search project knowledge base for existing documents, specs, or background.\n"
                f"- 'direct_work': Perform direct analytical synthesis and requirements drafting.\n"
                f"- 'delegation': Decompose a complex objective into multiple delegated subtasks.\n\n"
                f"Respond in valid JSON with fields: 'action_type', 'query', 'direct_work_objective', 'direct_work_instructions', 'delegated_tasks', 'reasoning'."
            )
            msg = await self._model.ainvoke([HumanMessage(content=prompt)])
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
        self._state.add_evidence({
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
        self._state.add_evidence({
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
        run_workflow: bool = False,
    ) -> AgentRunResponse:
        """Execute a synchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            current_task: Optional immediate task context override to associate with this execution.
            delegate: If True, execute via delegation and sub-agents.
            run_workflow: If True, execute via the application-owned full BRD workflow runner.

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

        # Route to application-owned full BRD workflow if requested
        should_run_workflow = run_workflow or (
            effective_ctx and bool(effective_ctx.metadata.get("run_workflow", False))
        )
        if should_run_workflow:
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
        run_workflow: bool = False,
    ) -> AgentRunResponse:
        """Execute an asynchronous interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            current_task: Optional immediate task context override to associate with this execution.
            delegate: If True, execute via delegation and sub-agents.
            run_workflow: If True, execute via the application-owned full BRD workflow runner.

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

        # Route to application-owned full BRD workflow if requested
        should_run_workflow = run_workflow or (
            effective_ctx and bool(effective_ctx.metadata.get("run_workflow", False))
        )
        if should_run_workflow:
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
        run_workflow: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute an asynchronous streaming interaction cycle via the DeepAgents harness.

        Args:
            request: AgentRunRequest or raw input string prompt.
            context: Optional AgentContext for project isolation.
            prior_messages: Optional sequence of prior BaseMessage instances to seed
                thread state if uninitialized.
            current_task: Optional immediate task context override to associate with this execution.
            run_workflow: If True, stream via the application-owned full BRD workflow runner.

        Yields:
            dict[str, Any]: Incremental content chunks, e.g. {"type": "content", "content": "..."}.
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

        # Route to application-owned full BRD workflow if requested
        should_run_workflow = run_workflow or (
            effective_ctx and bool(effective_ctx.metadata.get("run_workflow", False))
        )
        if should_run_workflow:
            prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
            async for event in self.stream_workflow_async(
                request=prompt_text,
                context=effective_ctx,
                current_task=current_task,
            ):
                yield event
            return

        token = set_current_agent_context(effective_ctx)
        exec_config: dict[str, Any] = {}
        if effective_ctx.conversation_id:
            exec_config["configurable"] = {"thread_id": effective_ctx.conversation_id}
        elif self._checkpointer is not None:
            exec_config["configurable"] = {"thread_id": "default"}

        has_thread_state = False
        if self._checkpointer is not None and exec_config.get("configurable"):
            if hasattr(self._graph, "get_state"):
                try:
                    state = self._graph.get_state(exec_config)
                    if state and state.values.get("messages"):
                        has_thread_state = True
                    elif prior_messages and hasattr(self._graph, "update_state"):
                        self._graph.update_state(exec_config, {"messages": list(prior_messages)})
                        has_thread_state = True
                except Exception as seed_exc:
                    logger.warning("Failed to inspect/seed checkpointer state: %s", seed_exc)

        prompt_text = request.input_text if isinstance(request, AgentRunRequest) else str(request)
        if has_thread_state:
            inputs = {"messages": [HumanMessage(content=prompt_text)]}
        elif prior_messages:
            inputs = {"messages": list(prior_messages) + [HumanMessage(content=prompt_text)]}
        else:
            inputs = {"messages": [HumanMessage(content=prompt_text)]}

        try:
            async for msg, metadata in self._graph.astream(
                inputs,
                config=exec_config if exec_config else None,
                stream_mode="messages",
            ):
                if isinstance(msg, (AIMessageChunk, AIMessage)):
                    if not getattr(msg, "tool_calls", None):
                        content = getattr(msg, "content", "")
                        if isinstance(content, str) and content:
                            yield {"type": "content", "content": content}
                        elif isinstance(content, list):
                            text_parts = [
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            ]
                            combined = "".join(text_parts)
                            if combined:
                                yield {"type": "content", "content": combined}
        finally:
            reset_current_agent_context(token)

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

        eval_result = None
        if evaluate_after and response.action_result:
            eval_result = self.evaluate(
                action_result=response.action_result,
                context=effective_ctx,
            )

        return response, eval_result

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

        self._state.add_evidence({
            "source": "user_clarification",
            "content": answer,
            "resolved_item": resolved_item,
        })

        if resolved_item:
            self._state.resolve_unresolved(resolved_item)

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
            },
        )

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
            info_payload = info_parts if info_parts else "*(No prior evidence collected)*"

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
            info_payload = info_parts if info_parts else "*(No prior evidence collected)*"

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
            },
        )

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

    def interpret_section_validation(
        self,
        validation_result: Optional[ValidationResult] = None,
        section_name: Optional[str] = None,
        context: Optional[AgentContext] = None,
        current_content: Optional[str] = None,
    ) -> SectionReworkStrategy:
        """Intelligently interpret section validation findings and formulate rework strategy.

        Responsibilities:
        - Sub-agent validates section against requirements and returns ValidationResult.
        - Lead Agent interprets the findings and determines the rework strategy:
          * Identifies whether rework is required.
          * Formulates actionable rework guidance.
          * Specifies targeted adjustments to content.
        - Application enforces the deterministic retry limit (max 2 attempts) and executes updates.
        """
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
            strategy = SectionReworkStrategy(
                section_name=target_sec,
                requires_rework=True,
                rework_guidance=guidance,
                specific_adjustments=specific_adjs,
                reasoning=f"Section '{target_sec}' validation status is {val_res.outcome.value}. Rework required.",
            )

        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.LEAD_AGENT,
            event_name="SECTION_VALIDATION_INTERPRETED",
            section=target_sec,
            requires_rework=strategy.requires_rework,
        )
        return strategy

    async def interpret_section_validation_async(
        self,
        validation_result: Optional[ValidationResult] = None,
        section_name: Optional[str] = None,
        context: Optional[AgentContext] = None,
        current_content: Optional[str] = None,
    ) -> SectionReworkStrategy:
        """Asynchronously interpret section validation findings and formulate rework strategy."""
        return self.interpret_section_validation(
            validation_result=validation_result,
            section_name=section_name,
            context=context,
            current_content=current_content,
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
        if initial_state is not None:
            self._state = initial_state
        elif isinstance(request, AgentRunRequest) and request.state is not None:
            if isinstance(request.state, BRDAgentState):
                self._state = request.state
            elif isinstance(request.state, dict):
                self._state = BRDAgentState.from_dict(request.state)

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

        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None) or str(uuid.uuid4())
        if effective_ctx.metadata is not None:
            effective_ctx.metadata["agent_run_id"] = agent_run_id
        if effective_ctx.project_id:
            self._state.metadata["project_id"] = effective_ctx.project_id

        input_prompt = request.input_text if isinstance(request, AgentRunRequest) else (str(request) if request is not None else None)
        effective_obj = objective or input_prompt or self._state.objective or "Generate Business Requirements Document"
        self._state.objective = effective_obj
        if current_task is not None:
            self._state.current_task = current_task

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
        elif gap_decision.action == GapResolutionAction.ASK_USER:
            log_trace_event(
                logger,
                agent_run_id=agent_run_id,
                actor=TraceActor.APPLICATION,
                event_name="USER_CLARIFICATION_REQUIRED",
                question=gap_decision.clarification_question,
            )
            self._state.metadata["pending_clarification"] = gap_decision.clarification_question

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
                )
                break

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

            for sec in recovery_strategy.affected_sections:
                sec_guidance = recovery_strategy.section_guidance.get(sec, "")
                gen_res = await self.update_section_async(
                    section=sec,
                    rework_feedback=sec_guidance,
                    existing_content=self._state.get_section_content(sec),
                    context=effective_ctx,
                )
                val_res = await self.validate_section_async(
                    section=sec,
                    section_content=gen_res.content,
                    context=effective_ctx,
                )
                if not val_res.is_valid:
                    for attempt in range(1, max_section_rework_attempts + 1):
                        gen_res = await self.update_section_async(
                            section=sec,
                            rework_feedback=val_res.rework_feedback,
                            existing_content=gen_res.content,
                            context=effective_ctx,
                        )
                        val_res = await self.validate_section_async(
                            section=sec,
                            section_content=gen_res.content,
                            context=effective_ctx,
                        )
                        if val_res.is_valid:
                            break

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
        final_output = self._state.assembled_brd or "BRD generation completed."
        log_trace_event(
            logger,
            agent_run_id=agent_run_id,
            actor=TraceActor.APPLICATION,
            event_name="WORKFLOW_COMPLETED",
            success=True,
            is_valid=final_val_res.is_valid,
        )

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
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream progress events and document content throughout the BRD workflow lifecycle."""
        if initial_state is not None:
            self._state = initial_state
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

        agent_run_id = (effective_ctx.metadata.get("agent_run_id") if effective_ctx.metadata else None) or str(uuid.uuid4())
        input_prompt = request.input_text if isinstance(request, AgentRunRequest) else (str(request) if request is not None else None)
        effective_obj = objective or input_prompt or self._state.objective or "Generate Business Requirements Document"

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
            self.retry_with_rag(query=gap_decision.query, context=effective_ctx, evaluate_after=True)
        elif gap_decision.action == GapResolutionAction.ASK_USER:
            yield {
                "type": "progress",
                "phase": "4_EVIDENCE_EVALUATION",
                "actor": TraceActor.APPLICATION,
                "message": f"Clarification requested: {gap_decision.clarification_question}",
            }

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
                break

        if not self.is_section_processing_complete:
            yield {
                "type": "progress",
                "phase": "6_DOCUMENT_ASSEMBLY",
                "actor": TraceActor.APPLICATION,
                "message": "Assembly blocked: not all sections completed.",
            }
            yield {"type": "content", "content": "BRD workflow halted before assembly."}
            return

        yield {
            "type": "progress",
            "phase": "6_DOCUMENT_ASSEMBLY",
            "actor": TraceActor.APPLICATION,
            "message": "Assembling complete Business Requirements Document...",
        }
        assembly_res = await self.assemble_brd_async(context=effective_ctx)

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
            for sec in recovery_strategy.affected_sections:
                guidance = recovery_strategy.section_guidance.get(sec, "")
                gen_res = await self.update_section_async(section=sec, rework_feedback=guidance, existing_content=self._state.get_section_content(sec), context=effective_ctx)
                val_res = await self.validate_section_async(section=sec, section_content=gen_res.content, context=effective_ctx)
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
