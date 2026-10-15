"""Deterministic context loading implementation for Step 1A of Phase 1.

Responsible for:
- Enforcing project/tenant boundary isolation and failing closed on unauthorized access.
- Loading canonical BRD version metadata and section hierarchy.
- Reconstructing prior conversation messages, pending clarifications, focus entities,
  and previously accepted decision records from message metadata.
- Merging clarification answers into user_provided_facts when resuming after clarification.
"""

from __future__ import annotations

from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from agents.recommendation.contracts import (
    AgentRequest,
    ClarificationRequest,
    RequestContext,
)
from exceptions.project import ProjectNotFoundError
from models.brd import BRDSectionModel
from observability.logging import get_logger
from services.brd_service import BRDService
from services.conversation_service import ConversationService
from services.project_service import ProjectService

logger = get_logger(__name__)


async def load_context(
    request: AgentRequest,
    session: AsyncSession,
    conversation_service: ConversationService,
    brd_service: BRDService,
    project_service: Optional[ProjectService] = None,
) -> tuple[RequestContext, list[BRDSectionModel]]:
    """Execute Step 1A: Deterministically load context for Recommendation Workflow.

    Args:
        request: Validated AgentRequest containing thread_id, brd_id (project_id),
            message, and UserContext.
        session: Active database session.
        conversation_service: Service for conversation and message operations.
        brd_service: Service for BRD canonical version and section management.
        project_service: Optional ProjectService instance.

    Returns:
        Tuple of (RequestContext, list of BRDSectionModel).

    Raises:
        ProjectNotFoundError: If the project does not exist or user lacks tenant access.
    """
    project_id = request.brd_id
    user_id = request.user.user_id

    # 1. Enforce Tenant Isolation Boundary (Fail Closed)
    p_service = project_service or ProjectService(session=session)
    try:
        project = await p_service.get_project(project_id=project_id, user_id=user_id)
        if project is None:
            logger.warning(
                "Access denied or project not found: project_id=%s, user_id=%s",
                project_id,
                user_id,
            )
            raise ProjectNotFoundError(f"Project '{project_id}' not found or access denied.")
    except Exception as exc:
        logger.warning(
            "Tenant validation failed for project_id=%s, user_id=%s: %s",
            project_id,
            user_id,
            exc,
        )
        raise ProjectNotFoundError(
            f"Project '{project_id}' access validation failed: {str(exc)}"
        ) from exc

    # 2. Retrieve Canonical BRD, Active Working Draft, and Section Tree
    _, current_draft = await brd_service.get_or_create_canonical_brd(project_id)
    raw_sections = current_draft.sections if current_draft and current_draft.sections else []
    sorted_sections: list[BRDSectionModel] = sorted(
        raw_sections, key=lambda s: getattr(s, "order_index", 0)
    )
    brd_version_label = (
        current_draft.version_label if current_draft and current_draft.version_label else "0.1"
    )

    # 3. Retrieve Conversation Message History
    existing_messages = await conversation_service.list_messages(
        project_id=project_id,
        conversation_id=request.thread_id,
    )

    # 4. Extract Prior State from Message Metadata
    pending_clarification: Optional[ClarificationRequest] = None
    last_recommendation: Optional[dict[str, Any]] = None
    focus_entity: Optional[str] = None
    decision_history: list[dict[str, Any]] = []
    user_provided_facts: dict[str, str] = {}

    history_lines: list[str] = []
    # Inspect chronological messages to rebuild conversational state
    for msg in existing_messages:
        role = msg.role
        content = msg.content or ""
        meta = msg.meta if hasattr(msg, "meta") else (getattr(msg, "message_metadata", None) or {})

        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")

            # Check for clarification request stored in metadata
            if "pending_clarification" in meta and meta["pending_clarification"]:
                try:
                    pending_clarification = ClarificationRequest.model_validate(
                        meta["pending_clarification"]
                    )
                except Exception as exc:
                    logger.debug("Could not parse pending_clarification from message: %s", exc)

            # Check for last recommendation
            if "recommendation" in meta and meta["recommendation"]:
                last_recommendation = meta["recommendation"]
            elif "last_recommendation" in meta and meta["last_recommendation"]:
                last_recommendation = meta["last_recommendation"]

            # Check for focus entity
            if "focus_entity" in meta and meta["focus_entity"]:
                focus_entity = meta["focus_entity"]

            # Check for decision records
            if "decision_record" in meta and meta["decision_record"]:
                decision_history.append(meta["decision_record"])
            if "decision_history" in meta and isinstance(meta["decision_history"], list):
                decision_history.extend(meta["decision_history"])

    # 5. Clarification Continuity: Merge Answer on Next Turn
    # If the immediately preceding turn was awaiting clarification, the user's current message
    # is the clarification answer.
    if pending_clarification is not None:
        user_provided_facts["clarification_answer"] = request.message
        for fact_name in pending_clarification.missing_facts:
            user_provided_facts[fact_name] = request.message

    # Format concise history summary (last 6 message turns)
    recent_history = history_lines[-6:] if len(history_lines) > 6 else history_lines
    history_summary = "\n".join(recent_history)

    request_context = RequestContext(
        request=request,
        history_summary=history_summary,
        brd_version=brd_version_label,
        resolved_references=[],
        decision_history=decision_history,
        pending_clarification=pending_clarification,
        user_provided_facts=user_provided_facts,
        last_recommendation=last_recommendation,
        focus_entity=focus_entity,
    )

    return request_context, sorted_sections
