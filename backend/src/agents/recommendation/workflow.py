"""Recommendation Workflow implementation — Phase 1 Orchestrator.

Orchestrates the sequential execution of Phase 1:
Workflow Entry -> Load Context (1A) -> Resolve References (1B) -> Intent Router Agent (1C)
-> Missing/Unclear Check (1D) -> Phase 1 Result.
"""

from __future__ import annotations

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

import uuid
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
from observability.llm_telemetry import get_current_telemetry_tracker, scoped_telemetry_context
from observability.logging import get_logger, scoped_run_context
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
        run_id = getattr(request, "run_id", None) or str(uuid.uuid4())

        async with scoped_run_context(run_id=run_id, workflow_name="RecommendationWorkflow"):
            logger.info(
                "Starting Recommendation Workflow Phase 1 (run_id: %s, thread_id: %s, brd_id: %s)",
                run_id,
                request.thread_id,
                request.brd_id,
            )

            try:
                # Step 1A: Load Context (Deterministic)
                logger.info(
                    "Executing Step 1A: Load Context (thread_id: %s, brd_id: %s)",
                    request.thread_id,
                    request.brd_id,
                )
                context, sections = await load_context(
                    request=request,
                    session=session,
                    conversation_service=conversation_service,
                    brd_service=brd_service,
                    project_service=project_service,
                )
                logger.info(
                    "Step 1A completed: Loaded %d canonical sections, history_summary_length=%d, prior_decisions=%d, focus_entity=%s",
                    len(sections),
                    len(context.history_summary),
                    len(context.decision_history),
                    context.focus_entity,
                )

                # Step 1B: Resolve References (Deterministic)
                logger.info(
                    "Executing Step 1B: Resolve References against %d canonical sections",
                    len(sections),
                )
                context = resolve_references(context=context, sections=sections)
                resolved_summary = [
                    f"'{r.raw_text}' -> section_id={r.section_id} (confidence={r.confidence})"
                    for r in context.resolved_references
                ]
                logger.info(
                    "Step 1B completed: Resolved %d references: %s",
                    len(context.resolved_references),
                    ", ".join(resolved_summary) if resolved_summary else "none",
                )

                # Step 1C: Intent Router Agent (LLM, No retrieval tools)
                logger.info("Executing Step 1C: Intent Router Agent invocation")
                with scoped_telemetry_context(
                    component="IntentRouterAgent",
                    phase="1C_INTENT_ROUTER",
                    operation="route",
                ):
                    route_decision = await self._router.route(context=context)

                logger.info(
                    "Step 1C completed: Intent Router produced decision: profile='%s', high_stakes=%s, confidence='%s', missing_facts=%s, rationale='%s'",
                    route_decision.profile,
                    route_decision.high_stakes,
                    route_decision.confidence,
                    route_decision.missing_facts,
                    route_decision.short_reason,
                )

                # Step 1D: Missing/Unclear Check (Deterministic control logic)
                logger.info("Executing Step 1D: Missing/Unclear Check evaluation")
                result = self.missing_unclear_check(
                    route=route_decision,
                    context=context,
                )
                logger.info(
                    "Step 1D completed: Workflow result status='%s'%s",
                    result.status,
                    f", clarification_questions={result.clarification.questions}" if result.clarification else "",
                )

                tracker = get_current_telemetry_tracker()
                if tracker:
                    logger.info(tracker.format_log_summary())

                logger.info(
                    "Recommendation Workflow Phase 1 finished successfully (run_id: %s, status: %s)",
                    run_id,
                    result.status,
                )
                return result

            except Exception as exc:
                logger.error(
                    "Error during Recommendation Workflow Phase 1 execution (run_id: %s): %s",
                    run_id,
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
