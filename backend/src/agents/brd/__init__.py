"""AI Agents subsystem package."""

from agents.brd.agent import BRDLeadAgent, create_brd_lead_agent
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
from agents.brd.state import BRDAgentState, BRDSectionStatus

__all__ = [
    "ActionResult",
    "ActionSource",
    "AgentConfig",
    "AgentContext",
    "AgentRunRequest",
    "AgentRunResponse",
    "BRDAgentState",
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
    "SectionValidationContext",
    "ValidationCategory",
    "ValidationFinding",
    "ValidationOutcome",
    "ValidationResult",
    "create_agent_model",
    "create_brd_lead_agent",
    "format_section_rework_guidance",
    "resolve_affected_sections",
]
