"""AI Agents subsystem package."""

from agents.brd.agent import (
    BRDLeadAgent,
    create_brd_lead_agent,
    get_system_instruction_path,
    load_system_instruction,
)
from agents.brd.config import AgentConfig, create_agent_model
from agents.brd.context import (
    ActionResult,
    ActionSource,
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
)
from agents.brd.final_validation import (
    BRDFinalValidationAgent,
    FinalValidationCategory,
    FinalValidationContext,
    FinalValidationFinding,
    FinalValidationOutcome,
    FinalValidationResult,
    FinalValidationSeverity,
    FindingResolutionStatus,
)
from agents.brd.recovery import (
    FinalValidationRecoveryResult,
    MAX_FINAL_VALIDATION_RECOVERY_CYCLES,
    format_section_rework_guidance,
    resolve_affected_sections,
)
from agents.brd.rewriter import (
    BRDRewriterAgent,
    BRDRewriterContext,
    BRDRewriterResult,
    DocumentEdit,
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
from agents.brd.state import (
    BRDAgentState,
    BRDDeepAgentState,
    BRDSectionStatus,
    SectionStatus,
)
from agents.brd.template import (
    extract_brd_sections,
    extract_section_requirements,
    extract_section_template,
    get_brd_template_path,
    is_administrative_section,
    load_brd_template,
)

__all__ = [
    "ActionResult",
    "ActionSource",
    "AgentConfig",
    "AgentContext",
    "AgentRunRequest",
    "AgentRunResponse",
    "BRDAgentState",
    "BRDDeepAgentState",
    "BRDFinalValidationAgent",
    "BRDLeadAgent",
    "BRDRewriterAgent",
    "BRDRewriterContext",
    "BRDRewriterResult",
    "BRDSectionGenerationAgent",
    "BRDSectionStatus",
    "BRDSectionValidationAgent",
    "DocumentEdit",
    "FinalValidationCategory",
    "FinalValidationContext",
    "FinalValidationFinding",
    "FinalValidationOutcome",
    "FinalValidationRecoveryResult",
    "FinalValidationResult",
    "FinalValidationSeverity",
    "FindingResolutionStatus",
    "MAX_FINAL_VALIDATION_RECOVERY_CYCLES",
    "SectionGenerationContext",
    "SectionGenerationResult",
    "SectionOperation",
    "SectionStatus",
    "SectionValidationContext",
    "ValidationCategory",
    "ValidationFinding",
    "ValidationOutcome",
    "ValidationResult",
    "create_agent_model",
    "create_brd_lead_agent",
    "extract_brd_sections",
    "extract_section_requirements",
    "extract_section_template",
    "format_section_rework_guidance",
    "get_brd_template_path",
    "get_system_instruction_path",
    "is_administrative_section",
    "load_brd_template",
    "load_system_instruction",
    "resolve_affected_sections",
]
