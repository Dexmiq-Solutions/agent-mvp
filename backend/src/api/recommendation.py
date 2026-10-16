"""API routes for Recommendation Workflow execution under strict project boundaries."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    Phase1Result,
    RouteDecision,
    UserContext,
)
from agents.recommendation.workflow import RecommendationWorkflow
from api.dependencies import (
    get_brd_service,
    get_conversation_service,
    get_current_user,
    get_db_session,
    get_recommendation_workflow,
)
from db.session import get_async_session_maker
from models.user import UserModel
from observability.logging import get_logger
from schemas.conversation import MessageCreate
from services.brd_service import BRDService
from services.conversation_service import ConversationService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/projects/{project_id}/conversations",
    tags=["recommendations"],
)


class RecommendationWorkflowResponse(BaseModel):
    """API response model for Recommendation Workflow execution."""

    status: str
    response_markdown: str
    route_decision: Optional[RouteDecision] = None
    clarification: Optional[ClarificationRequest] = None
    user_message_id: str
    assistant_message_id: Optional[str] = None
    duration_seconds: float = Field(default=0.0)
    error_message: Optional[str] = None


@router.post(
    "/{conversation_id}/recommendations",
    status_code=status.HTTP_200_OK,
    summary="Execute Recommendation Workflow Phase 1 (Understand the Request)",
    response_model=RecommendationWorkflowResponse,
)
async def execute_recommendation(
    project_id: str,
    conversation_id: str,
    payload: MessageCreate,
    current_user: UserModel = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
    brd_service: BRDService = Depends(get_brd_service),
    workflow: RecommendationWorkflow = Depends(get_recommendation_workflow),
    session: AsyncSession = Depends(get_db_session),
) -> RecommendationWorkflowResponse:
    """Execute Phase 1 of the Recommendation Workflow within project and tenant boundaries.

    Validates project and conversation existence under tenant ownership, acquires run-ownership lease,
    persists the user message, executes Phase 1, fences consequential writes against lease expiry,
    persists the resulting assistant turn with durable metadata, and releases the execution lease.
    """
    start_time = time.perf_counter()

    run_id = str(uuid.uuid4())
    logger.info(
        "Recommendation API request received (project_id: %s, conversation_id: %s, run_id: %s)",
        project_id,
        conversation_id,
        run_id,
    )

    # 1. Enforce Project Isolation & Atomically Validate Workflow + Acquire Run Ownership Lease
    await service.acquire_execution_lease(
        project_id=project_id,
        conversation_id=conversation_id,
        run_id=run_id,
        expected_workflow="recommendation",
        ttl_seconds=45,
    )

    # 2. Persist User Message tagged with workflow and run_id
    user_message = await service.create_message(
        project_id=project_id,
        conversation_id=conversation_id,
        content=payload.content,
        role="user",
        metadata={
            **(payload.metadata or {}),
            "workflow": "recommendation",
            "agent_run_id": run_id,
        },
    )

    # Pre-agent database boundary: commit user message turn
    await session.commit()

    # 3. Setup background heartbeat
    stop_heartbeat = asyncio.Event()

    async def _heartbeat_worker() -> None:
        bind_engine = getattr(session, "bind", None)
        session_maker = (
            get_async_session_maker(engine=bind_engine)
            if bind_engine is not None
            else get_async_session_maker()
        )
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=15.0)
                break
            except asyncio.TimeoutError:
                pass
            try:
                async with session_maker() as hb_session:
                    hb_service = ConversationService(session=hb_session)
                    renewed = await hb_service.heartbeat_execution_lease(
                        conversation_id=conversation_id,
                        run_id=run_id,
                        ttl_seconds=45,
                    )
                    if not renewed:
                        logger.warning(
                            "Heartbeat rejected: Lease expired or superseded for run '%s' on conversation '%s'",
                            run_id,
                            conversation_id,
                        )
                        break
            except Exception as exc:
                logger.warning("Heartbeat exception for run '%s': %s", run_id, exc)

    heartbeat_task = asyncio.create_task(_heartbeat_worker())

    try:
        # 4. Construct Inbound AgentRequest
        agent_request = AgentRequest(
            thread_id=conversation_id,
            brd_id=project_id,
            message=payload.content,
            user=UserContext(
                user_id=current_user.id,
                tenant_id=current_user.id,
                permissions=["project:read", "brd:read"],
            ),
            run_id=run_id,
        )

        # 5. Execute Phase 1 Workflow
        phase1_result: Phase1Result = await workflow.execute_phase1(
            request=agent_request,
            session=session,
            conversation_service=service,
            brd_service=brd_service,
        )

        duration = round(time.perf_counter() - start_time, 3)
        logger.info(
            "Recommendation API request processed (run_id: %s, duration: %.2fs, status: %s)",
            run_id,
            duration,
            phase1_result.status,
        )

        if phase1_result.status == "failed":
            logger.error(
                "Recommendation Workflow execution failed: %s",
                phase1_result.error_message,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Recommendation Workflow failed: {phase1_result.error_message}",
            )

        # 6. Pre-persistence lease verification under row lock
        await service.verify_and_renew_lease(
            conversation_id=conversation_id,
            expected_run_id=run_id,
            ttl_seconds=45,
        )

        # 7. Handle Branch Results & Persist Assistant Turn
        assistant_msg_id: Optional[str] = None

        if phase1_result.status == "needs_user" and phase1_result.clarification:
            clarification = phase1_result.clarification
            numbered_questions = "\n".join(
                f"{i + 1}. {q}" for i, q in enumerate(clarification.questions)
            )
            response_markdown = (
                f"Before proceeding with your request, I need clarification on the following:\n\n"
                f"{numbered_questions}"
            )

            assistant_msg = await service.create_message(
                project_id=project_id,
                conversation_id=conversation_id,
                content=response_markdown,
                role="assistant",
                metadata={
                    "workflow": "recommendation",
                    "agent_run_id": run_id,
                    "phase": "1_UNDERSTAND_THE_REQUEST",
                    "status": "needs_user",
                    "pending_clarification": clarification.model_dump(),
                    "route_decision": (
                        phase1_result.route_decision.model_dump()
                        if phase1_result.route_decision
                        else None
                    ),
                    "user_message_id": user_message.id,
                    "duration_seconds": duration,
                },
            )
            # Final pre-commit fencing assertion using real-time database clock
            await service.pre_commit_fencing_assertion(
                conversation_id=conversation_id,
                expected_run_id=run_id,
            )
            await session.commit()
            assistant_msg_id = assistant_msg.id

        else:
            # Phase 1 Complete (ready_for_phase_2)
            route = phase1_result.route_decision
            stakes_desc = "Yes" if route and route.high_stakes else "No"
            if route and route.high_stakes and route.high_stakes_reasons:
                stakes_desc += f" ({', '.join(route.high_stakes_reasons)})"

            assumptions_block = ""
            if phase1_result.context.assumptions:
                assumptions_list = "\n".join(f"- {a}" for a in phase1_result.context.assumptions)
                assumptions_block = f"\n\n**Explicit Assumptions Recorded:**\n{assumptions_list}"

            response_markdown = (
                f"### Phase 1: Request Understood\n"
                f"- **Selected Profile**: `{route.profile if route else 'unknown'}`\n"
                f"- **High Stakes**: {stakes_desc}\n"
                f"- **Routing Rationale**: {route.short_reason if route else 'N/A'}"
                f"{assumptions_block}\n\n"
                f"*(Phase 1 complete: Ready for Phase 2 — Apply the Profile)*"
            )

            assistant_msg = await service.create_message(
                project_id=project_id,
                conversation_id=conversation_id,
                content=response_markdown,
                role="assistant",
                metadata={
                    "workflow": "recommendation",
                    "agent_run_id": run_id,
                    "phase": "1_UNDERSTAND_THE_REQUEST",
                    "status": "ready_for_phase_2",
                    "route_decision": route.model_dump() if route else None,
                    "focus_entity": phase1_result.context.focus_entity,
                    "assumptions": phase1_result.context.assumptions,
                    "user_message_id": user_message.id,
                    "duration_seconds": duration,
                },
            )
            # Final pre-commit fencing assertion using real-time database clock
            await service.pre_commit_fencing_assertion(
                conversation_id=conversation_id,
                expected_run_id=run_id,
            )
            await session.commit()
            assistant_msg_id = assistant_msg.id

        return RecommendationWorkflowResponse(
            status=phase1_result.status,
            response_markdown=response_markdown,
            route_decision=phase1_result.route_decision,
            clarification=phase1_result.clarification,
            user_message_id=user_message.id,
            assistant_message_id=assistant_msg_id,
            duration_seconds=duration,
        )

    finally:
        stop_heartbeat.set()
        await heartbeat_task
        try:
            await service.release_execution_lease(
                conversation_id=conversation_id,
                run_id=run_id,
            )
        except Exception as exc:
            logger.warning("Failed to release lease for run '%s': %s", run_id, exc)
