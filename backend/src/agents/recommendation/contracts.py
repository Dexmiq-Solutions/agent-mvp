"""Contracts and data models for the Recommendation Agent subsystem.

Conforms strictly to Section 7 of docs/recommendation_agent_context.md.
Defines data structures for Phase 1: Understand the Request, including input payloads,
context structures, resolved references, route decisions, and clarification requests.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

ProfileId = Literal["explain", "recommend", "conflict", "communication"]
SourceKind = Literal["brd", "project_evidence", "industry", "assumption"]
HighStakesReason = Literal[
    "hard_to_undo",
    "compliance_or_legal",
    "client_facing",
    "large_cost_or_schedule",
]


class UserContext(BaseModel):
    """Authenticated user context, identity, and authorization boundary."""

    user_id: str
    tenant_id: str
    permissions: list[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    """Inbound request payload submitted to the Recommendation Workflow."""

    thread_id: str
    brd_id: str
    message: str
    user: UserContext
    run_id: Optional[str] = None


class ResolvedReference(BaseModel):
    """Entity or section reference extracted and resolved against the BRD section tree."""

    raw_text: str
    section_id: Optional[str] = None
    confidence: Literal["exact", "inferred", "ambiguous"]
    candidates: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """Targeted clarification questions returned to the user when facts are missing or unclear."""

    questions: list[str]
    missing_facts: list[str] = Field(default_factory=list)
    reason: str
    original_message: str
    profile: ProfileId


class RequestContext(BaseModel):
    """Context state assembled during Phase 1A (Load Context) and Phase 1B (Resolve References)."""

    request: AgentRequest
    history_summary: str = ""
    brd_version: str = ""
    resolved_references: list[ResolvedReference] = Field(default_factory=list)
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    pending_clarification: Optional[ClarificationRequest] = None
    user_provided_facts: dict[str, str] = Field(default_factory=dict)
    last_recommendation: Optional[dict[str, Any]] = None
    focus_entity: Optional[str] = None
    proceed_with_assumptions: bool = False
    assumptions: list[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    """Structured routing and stakes triage decision produced by the Intent Router Agent."""

    profile: ProfileId
    high_stakes: bool
    high_stakes_reasons: list[HighStakesReason] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    short_reason: str


class Phase1Result(BaseModel):
    """Result of Phase 1: Understand the Request.

    Either advances with ready_for_phase_2 containing the RouteDecision, or
    pauses with needs_user containing a ClarificationRequest, or marks failed.
    """

    status: Literal["ready_for_phase_2", "needs_user", "failed"]
    route_decision: Optional[RouteDecision] = None
    context: RequestContext
    clarification: Optional[ClarificationRequest] = None
    error_message: Optional[str] = None
