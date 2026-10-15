"""End-to-end unit tests for RecommendationWorkflow Phase 1 Orchestrator.

Covers:
- Step 1D missing/unclear check:
  - Clean path to 'ready_for_phase_2'.
  - Turn 1 clarification pause returning 'needs_user' with 1-3 targeted questions.
  - Turn 2 clarification resume to 'ready_for_phase_2'.
  - One-round clarification policy (D25) proceeding with explicit assumptions.
- Proof of isolation: Phase 1 execution does not invoke Phase 2, Lead Agent,
  Option Explorer, Critic, or Verification Agent.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    Phase1Result,
    RequestContext,
    RouteDecision,
    UserContext,
)
from agents.recommendation.router.agent import IntentRouterAgent
from agents.recommendation.workflow import RecommendationWorkflow
from models.brd import BRDModel, BRDSectionModel, BRDVersionModel
from models.project import ProjectModel


def make_request(message: str, thread_id: str = "conv-wf") -> AgentRequest:
    return AgentRequest(
        thread_id=thread_id,
        brd_id="proj-wf",
        message=message,
        user=UserContext(user_id="user-wf", tenant_id="user-wf"),
    )


def setup_mock_services():
    session = AsyncMock()
    project_service = AsyncMock()
    project_service.get_project.return_value = ProjectModel(id="proj-wf", user_id="user-wf")

    brd_service = AsyncMock()
    sec = BRDSectionModel(
        id="s-6",
        brd_version_id="v-1",
        section_key="6.0",
        title="6. Security and Compliance",
        order_index=6,
        content="Point 4: MFA requirements.",
    )
    draft_version = BRDVersionModel(id="v-1", brd_id="b-1", version_number=1, version_label="0.1", sections=[sec])
    brd_service.get_or_create_canonical_brd.return_value = (BRDModel(id="b-1"), draft_version)

    conversation_service = AsyncMock()
    conversation_service.list_messages.return_value = []

    return session, project_service, brd_service, conversation_service


@pytest.mark.asyncio
async def test_workflow_clean_path_to_ready_for_phase_2():
    session, project_service, brd_service, conversation_service = setup_mock_services()

    # Router returns high confidence explain decision
    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="explain",
            high_stakes=False,
            high_stakes_reasons=[],
            missing_facts=[],
            confidence="high",
            short_reason="Explain section 6",
        )
    )

    workflow = RecommendationWorkflow(router_agent=mock_router)
    request = make_request("Explain point 4 of section 6.")

    result: Phase1Result = await workflow.execute_phase1(
        request=request,
        session=session,
        conversation_service=conversation_service,
        brd_service=brd_service,
        project_service=project_service,
    )

    assert result.status == "ready_for_phase_2"
    assert result.route_decision is not None
    assert result.route_decision.profile == "explain"
    assert result.clarification is None
    assert len(result.context.resolved_references) == 1
    assert result.context.resolved_references[0].section_id == "6.0"


@pytest.mark.asyncio
async def test_workflow_missing_facts_pauses_for_clarification():
    session, project_service, brd_service, conversation_service = setup_mock_services()

    # Router returns communication profile with missing facts
    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="communication",
            high_stakes=True,
            high_stakes_reasons=["client_facing", "large_cost_or_schedule"],
            missing_facts=["cause", "new_date"],
            confidence="high",
            short_reason="Client delay message requires cause and new target date.",
        )
    )

    workflow = RecommendationWorkflow(router_agent=mock_router)
    request = make_request("We will miss the deadline. How do we tell the client?")

    result: Phase1Result = await workflow.execute_phase1(
        request=request,
        session=session,
        conversation_service=conversation_service,
        brd_service=brd_service,
        project_service=project_service,
    )

    assert result.status == "needs_user"
    assert result.clarification is not None
    assert result.clarification.profile == "communication"
    assert len(result.clarification.questions) == 2
    # Ensure questions have explanations/rationales
    for q in result.clarification.questions:
        assert "(" in q and ")" in q


@pytest.mark.asyncio
async def test_workflow_one_round_clarification_bound_d25():
    """Verify D25: after 1 clarification round, do not re-prompt even if facts remain missing."""
    session, project_service, brd_service, conversation_service = setup_mock_services()

    # Pre-populate context to simulate Turn 2 (resuming after a clarification)
    prior_clarification = ClarificationRequest(
        questions=["What is the new date?"],
        missing_facts=["new_date"],
        reason="Missing date",
        original_message="We are delayed",
        profile="communication",
    )

    # Router still flags a missing fact
    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="communication",
            high_stakes=True,
            high_stakes_reasons=["client_facing"],
            missing_facts=["new_date"],
            confidence="high",
            short_reason="Still missing formal date",
        )
    )

    workflow = RecommendationWorkflow(router_agent=mock_router)

    # Construct context that already has pending_clarification
    req = make_request("I don't know the date yet, just draft the initial notice.")
    ctx = RequestContext(
        request=req,
        pending_clarification=prior_clarification,
    )

    # Missing/unclear check on Turn 2
    result = workflow.missing_unclear_check(
        route=await mock_router.route(ctx),
        context=ctx,
    )

    # Must NOT ask again; must proceed with assumptions
    assert result.status == "ready_for_phase_2"
    assert ctx.proceed_with_assumptions is True
    assert any("new_date" in a for a in ctx.assumptions)


@pytest.mark.asyncio
async def test_workflow_isolation_from_later_phases():
    """Verify Phase 1 does not invoke any later-phase agents."""
    session, project_service, brd_service, conversation_service = setup_mock_services()

    mock_router = MagicMock(spec=IntentRouterAgent)
    mock_router.route = AsyncMock(
        return_value=RouteDecision(
            profile="explain",
            high_stakes=False,
            high_stakes_reasons=[],
            missing_facts=[],
            confidence="high",
            short_reason="Explain",
        )
    )

    workflow = RecommendationWorkflow(router_agent=mock_router)
    # Confirm workflow only holds the router agent
    assert hasattr(workflow, "router")
    assert not hasattr(workflow, "lead_agent")
    assert not hasattr(workflow, "option_explorer")
    assert not hasattr(workflow, "critic")
    assert not hasattr(workflow, "verifier")

    result = await workflow.execute_phase1(
        request=make_request("Explain section 6"),
        session=session,
        conversation_service=conversation_service,
        brd_service=brd_service,
        project_service=project_service,
    )

    assert result.status == "ready_for_phase_2"
