"""Recommendation Agent subsystem package.

Exports Phase 1 contracts, configuration, workflow, and agents.
"""

from agents.recommendation.config import (
    RecommendationConfig,
    create_recommendation_model,
    create_router_model,
)
from agents.recommendation.context_loader import load_context
from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    HighStakesReason,
    Phase1Result,
    ProfileId,
    RequestContext,
    ResolvedReference,
    RouteDecision,
    SourceKind,
    UserContext,
)
from agents.recommendation.reference_resolver import resolve_references
from agents.recommendation.router import IntentRouterAgent
from agents.recommendation.workflow import RecommendationWorkflow

__all__ = [
    "AgentRequest",
    "ClarificationRequest",
    "HighStakesReason",
    "IntentRouterAgent",
    "Phase1Result",
    "ProfileId",
    "RecommendationConfig",
    "RecommendationWorkflow",
    "RequestContext",
    "ResolvedReference",
    "RouteDecision",
    "SourceKind",
    "UserContext",
    "create_recommendation_model",
    "create_router_model",
    "load_context",
    "resolve_references",
]
