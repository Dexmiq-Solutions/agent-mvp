"""Recommendation Workflow implementation — Phase 1 Orchestrator.

Orchestrates the sequential execution of Phase 1:
Workflow Entry -> Load Context (1A) -> Resolve References (1B) -> Intent Router Agent (1C)
-> Missing/Unclear Check (1D) -> Phase 1 Result.
"""

from __future__ import annotations

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from agents.recommendation.config import RecommendationConfig
from agents.recommendation.context_loader import load_context
from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    Phase1Result,
    RequestContext,
    RouteDecision,
)
from agents.recommendation.reference_resolver import resolve_references
from agents.recommendation.router.agent import IntentRouterAgent
from observability.logging import get_logger
from services.brd_service import BRDService
from services.conversation_service import ConversationService
from services.project_service import ProjectService

logger = get_logger(__name__)


class RecommendationWorkflow:
    """Central entry point for the Recommendation Workflow.

    Independently initializable and executable. Executes Phase 1 to produce
    either a routing decision ready for Phase 2, or a targeted clarification response.
    """

    def __init__(
        self,
        router_agent: Optional[IntentRouterAgent] = None,
        config: Optional[RecommendationConfig] = None,
    ) -> None:
        """Initialize RecommendationWorkflow.

        Args:
            router_agent: Optional IntentRouterAgent instance.
            config: Optional RecommendationConfig instance.
        """
        self._config = config or RecommendationConfig.from_settings()
        self._router = router_agent or IntentRouterAgent(config=self._config)

    @property
    def router(self) -> IntentRouterAgent:
        """Return the active IntentRouterAgent instance."""
        return self._router

    async def execute_phase1(
        self,
        request: AgentRequest,
        session: AsyncSession,
        conversation_service: ConversationService,
        brd_service: BRDService,
        project_service: Optional[ProjectService] = None,
    ) -> Phase1Result:
        """Execute Phase 1: Understand the Request.

        Args:
            request: Inbound user request payload with user and thread identifiers.
            session: Active database session.
            conversation_service: Active ConversationService.
            brd_service: Active BRDService.
            project_service: Optional ProjectService.

        Returns:
            Phase1Result with status 'ready_for_phase_2', 'needs_user', or 'failed'.
        """
        logger.info(
            "Starting Recommendation Workflow Phase 1 (thread_id: %s, brd_id: %s)",
            request.thread_id,
            request.brd_id,
        )

        try:
            # Step 1A: Load Context (Deterministic)
            context, sections = await load_context(
                request=request,
                session=session,
                conversation_service=conversation_service,
                brd_service=brd_service,
                project_service=project_service,
            )

            # Step 1B: Resolve References (Deterministic)
            context = resolve_references(context=context, sections=sections)

            # Step 1C: Intent Router Agent (LLM, No retrieval tools)
            route_decision = await self._router.route(context=context)

            # Step 1D: Missing/Unclear Check (Deterministic control logic)
            return self.missing_unclear_check(
                route=route_decision,
                context=context,
            )

        except Exception as exc:
            logger.error(
                "Error during Recommendation Workflow Phase 1 execution: %s",
                exc,
                exc_info=True,
            )
            dummy_context = RequestContext(request=request)
            return Phase1Result(
                status="failed",
                context=dummy_context,
                error_message=str(exc),
            )

    def missing_unclear_check(
        self,
        route: RouteDecision,
        context: RequestContext,
    ) -> Phase1Result:
        """Execute Step 1D: Evaluate router findings and determine workflow branch.

        When required facts are missing or confidence is low:
        - If Turn 1 (no prior clarification in this thread): returns 'needs_user' with
          1 to 3 targeted questions explaining why each matters.
        - If Turn 2 (resuming from previous clarification): enforces D25 one-round policy;
          does not re-prompt, but proceeds with explicit assumptions recorded.
        """
        has_missing_facts = len(route.missing_facts) > 0
        is_low_confidence = route.confidence == "low"

        # If facts are complete and confidence is high, advance directly to Phase 2
        if not has_missing_facts and not is_low_confidence:
            logger.info(
                "Phase 1 complete: Profile '%s' selected (high_stakes=%s, confidence=high)",
                route.profile,
                route.high_stakes,
            )
            return Phase1Result(
                status="ready_for_phase_2",
                route_decision=route,
                context=context,
            )

        # Enforce One-Round Clarification Policy (D25)
        # If the context indicates we already paused and received clarification on the prior turn,
        # we do not interrogate the user again.
        if context.pending_clarification is not None:
            logger.info(
                "One-round clarification already completed. Proceeding to Phase 2 with explicit assumptions (D25)."
            )
            context.proceed_with_assumptions = True
            for fact in route.missing_facts:
                assumption_text = (
                    f"Assumed standard baseline constraints for '{fact}' because it was not specified."
                )
                if assumption_text not in context.assumptions:
                    context.assumptions.append(assumption_text)

            if is_low_confidence:
                context.assumptions.append(
                    f"Proceeding under assumption that request aligns with profile '{route.profile}'."
                )

            return Phase1Result(
                status="ready_for_phase_2",
                route_decision=route,
                context=context,
            )

        # Turn 1: Build 1 to 3 targeted clarification questions explaining why each matters
        questions = self._build_clarification_questions(route, context)

        clarification = ClarificationRequest(
            questions=questions[:3],
            missing_facts=route.missing_facts,
            reason=route.short_reason,
            original_message=context.request.message,
            profile=route.profile,
        )

        logger.info(
            "Phase 1 pausing for user clarification: %d questions asked (profile: %s)",
            len(clarification.questions),
            route.profile,
        )

        return Phase1Result(
            status="needs_user",
            route_decision=route,
            context=context,
            clarification=clarification,
        )

    def _build_clarification_questions(
        self,
        route: RouteDecision,
        context: RequestContext,
    ) -> list[str]:
        """Construct between 1 and 3 targeted clarification questions with rationales."""
        questions: list[str] = []

        # 1. Questions for ambiguous references
        for ref in context.resolved_references:
            if ref.confidence == "ambiguous":
                if ref.candidates:
                    cands_str = ", ".join(ref.candidates)
                    questions.append(
                        f"Which section did you mean: {cands_str}? "
                        f"(Needed to ground our analysis in the exact requirement text)."
                    )
                else:
                    questions.append(
                        f"Could you specify which section or requirement you are referring to? "
                        f"('{ref.raw_text}' could not be located in the current BRD)."
                    )

        # 2. Questions for missing facts (e.g. Communication profile)
        fact_rationales = {
            "cause": "What is the primary cause of the delay or change? (Needed to choose between a solution-first approach and an explanatory approach).",
            "impact": "What is the specific milestone or deliverable impact? (Needed to evaluate relationship risk and scope adjustments).",
            "new_date": "What is the revised target completion date? (Needed to provide a clear, accountable commitment in the message).",
            "mitigation": "What corrective or mitigation steps are already underway? (Needed to reassure the client and maintain trust).",
        }

        for fact in route.missing_facts:
            if fact in fact_rationales:
                questions.append(fact_rationales[fact])
            else:
                questions.append(
                    f"Could you provide information about '{fact}'? "
                    f"(Needed to formulate an accurate recommendation)."
                )

        # 3. Fallback question if query was simply low confidence without specific missing facts
        if not questions and route.confidence == "low":
            questions.append(
                "Could you clarify your specific objective? "
                "(For example: whether you need an explanation of a current requirement, "
                "recommendations for alternatives, resolution of a conflict, or client communication guidance)."
            )

        return questions
