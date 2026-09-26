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

from enum import Enum
from pathlib import Path
import re
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

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
from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import (
    ActionResult,
    ActionSource,
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
    reset_current_agent_context,
    set_current_agent_context,
)
from observability.logging import get_logger
from services.rag_service import RAGService
from tools import get_default_tools
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import create_search_project_knowledge_tool, search_project_knowledge
from deepagents.backends.protocol import BackendProtocol
from langgraph.store.base import BaseStore
from agents.runtime.memory import (
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


class WorkflowDecision(str, Enum):
    """Workflow decision determined by the BRD Lead Agent following evaluation."""

    PROCEED_TO_SECTION_GENERATION = "proceed_to_section_generation"
    RAG = "rag"
    ASK_USER = "ask_user"


class BRDLeadAgent:
    """Domain-specific Agent responsible for accomplishing the BRD generation objective.

    The BRD Lead Agent is the primary agent for the BRD generation use case.
    It operates across three foundational pillars:
    1. System Instruction (system_instruction.md): Defines how the Agent behaves.
    2. BRD Template (brd_template.md): Defines what the Agent must produce.
    3. Agent State (BRDAgentState): Defines what the Agent currently knows and remembers.

    It executes within the DeepAgents execution harness managed by AgentRuntime,
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
        runtime: Optional[AgentRuntime] = None,
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
    ) -> None:
        """Initialize the BRD Lead Agent.

        Args:
            runtime: Optional existing AgentRuntime harness. If provided, runtime execution
                is delegated to this instance.
            config: Optional AgentConfig instance. Used if runtime is not provided.
            model: Optional pre-configured BaseChatModel. Used if runtime is not provided.
            tools: Optional sequence of tools. Defaults to get_default_tools() if runtime is not provided.
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
        """
        resolved_instruction = system_instruction or system_prompt or load_system_instruction()
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
            self._store = store if store is not None else (
                getattr(runtime, "store", None) or get_default_memory_store()
            )
            self._memory_sources = [
                normalize_memory_path(p) for p in (memory_sources or [DEFAULT_PROJECT_MEMORY_FILE])
            ]
            ns_factory = create_project_namespace_factory(
                bound_project_id=self._project_id,
                agent=self,
            )
            self._memory_backend = (
                getattr(runtime, "backend", None)
                if runtime is not None and getattr(runtime, "backend", None) is not None
                else ProjectMemoryStoreBackend(
                    namespace=ns_factory,
                    store=self._store,
                )
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

        if runtime is not None:
            self._runtime = runtime
            # Configure runtime system instruction if not already set
            if getattr(self._runtime, "_system_prompt", None) != resolved_instruction:
                self._runtime._system_prompt = resolved_instruction
        else:
            if tools is not None:
                resolved_tools = list(tools)
            elif not enable_rag:
                resolved_tools = [echo_diagnostic_tool]
            elif rag_service is not None:
                resolved_tools = [
                    echo_diagnostic_tool,
                    create_search_project_knowledge_tool(rag_service=rag_service),
                ]
            else:
                resolved_tools = get_default_tools()

            runtime_kwargs: dict[str, Any] = {}
            if self._enable_memory:
                runtime_kwargs["store"] = self._store
                runtime_kwargs["backend"] = self._memory_backend
                runtime_kwargs["memory"] = self._memory_sources
                runtime_kwargs["permissions"] = self._memory_permissions
                runtime_kwargs["middleware"] = [self._memory_middleware]

            self._runtime = AgentRuntime(
                config=config,
                model=model,
                tools=resolved_tools,
                system_prompt=resolved_instruction,
                **runtime_kwargs,
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
    def runtime(self) -> AgentRuntime:
        """Return the underlying AgentRuntime execution harness."""
        return self._runtime

    @property
    def config(self) -> AgentConfig:
        """Return the active Agent configuration."""
        return self._runtime.config

    @property
    def model(self) -> BaseChatModel:
        """Return the active language model."""
        return self._runtime.model

    @property
    def tools(self) -> list[Any]:
        """Return the list of active tools."""
        return self._runtime.tools

    @property
    def graph(self) -> Any:
        """Return the compiled DeepAgents graph."""
        return self._runtime.graph

    @property
    def sections(self) -> list[str]:
        """Return ordered required BRD sections dynamically extracted from the template."""
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

                # 1. Identify affected sections from current validation findings
                candidate_affected = current_result.affected_sections
                resolved_affected = resolve_affected_sections(candidate_affected, self.sections)

                if not resolved_affected:
                    logger.warning(
                        "Final validation recovery identified no resolvable affected sections (project_id: %s, cycle: %d, candidates: %s); stopping recovery safely",
                        project_id,
                        cycle,
                        candidate_affected,
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

                    sec_feedback = format_section_rework_guidance(sec, current_result)

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

                candidate_affected = current_result.affected_sections
                resolved_affected = resolve_affected_sections(candidate_affected, self.sections)

                if not resolved_affected:
                    logger.warning(
                        "Async final validation recovery identified no resolvable affected sections (project_id: %s, cycle: %d, candidates: %s); stopping recovery safely",
                        project_id,
                        cycle,
                        candidate_affected,
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

                    sec_feedback = format_section_rework_guidance(sec, current_result)

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

        response = self._runtime.execute(request=request, context=effective_ctx)
        # Converge onto common Action Result boundary
        action_res = response.to_action_result()
        response.action_result = action_res
        self._state.set_action_result(action_res)
        response.state = self._state
        return response

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

        response = await self._runtime.execute_async(request=request, context=effective_ctx)
        action_res = response.to_action_result()
        response.action_result = action_res
        self._state.set_action_result(action_res)
        response.state = self._state
        return response

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

    def decide_next_step(
        self,
        evaluation_result: Optional[EvaluationResult] = None,
        can_rag_resolve: Optional[bool] = None,
    ) -> WorkflowDecision:
        """Decide the next workflow action based on an EvaluationResult.

        Architectural Rule:
        The Evaluation Sub-Agent evaluates. The BRD Lead Agent decides what happens next:
        - If SUFFICIENT: Proceed toward BRD section generation.
        - If INSUFFICIENT:
            - If RAG can reasonably provide the missing information: execute RAG.
            - If RAG cannot reasonably provide it: ask the user for clarification.

        Args:
            evaluation_result: Optional specific EvaluationResult to base the decision on.
                Defaults to self.state.latest_evaluation_result.
            can_rag_resolve: Optional explicit boolean flag indicating whether RAG can
                resolve the gaps. If omitted, heuristic assessment is performed.

        Returns:
            WorkflowDecision: PROCEED_TO_SECTION_GENERATION, RAG, or ASK_USER.

        Raises:
            ValueError: If no EvaluationResult is provided or available in state.
        """
        eval_res = evaluation_result or self._state.latest_evaluation_result
        if eval_res is None:
            raise ValueError(
                "No EvaluationResult provided or available in state to decide next step."
            )

        if eval_res.is_sufficient:
            logger.info("Evaluation outcome is SUFFICIENT -> Proceeding toward section generation")
            return WorkflowDecision.PROCEED_TO_SECTION_GENERATION

        logger.info("Evaluation outcome is INSUFFICIENT -> Assessing gap resolution path")

        if not self.has_rag_capability:
            logger.info("RAG capability not equipped -> Choosing ASK_USER")
            return WorkflowDecision.ASK_USER

        if can_rag_resolve is not None:
            decision = WorkflowDecision.RAG if can_rag_resolve else WorkflowDecision.ASK_USER
            logger.info("Explicit RAG feasibility provided -> Choosing %s", decision.value)
            return decision

        is_rag_feasible = self._can_rag_reasonably_provide(
            missing_items=eval_res.missing_information,
            unresolved_items=eval_res.unresolved_information,
        )
        decision = WorkflowDecision.RAG if is_rag_feasible else WorkflowDecision.ASK_USER
        logger.info("Lead Agent assessed gap feasibility -> Choosing %s", decision.value)
        return decision

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


def create_brd_lead_agent(
    runtime: Optional[AgentRuntime] = None,
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
        runtime: Optional existing AgentRuntime instance.
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
