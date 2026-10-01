import asyncio
from collections.abc import AsyncIterator
import json
import time
from typing import Any, Optional, Union
import uuid

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from agents.brd.agent import BRDLeadAgent
from agents.brd.context import AgentContext
from api.dependencies import get_brd_lead_agent, get_conversation_service
from observability.logging import get_logger
from schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageResponse,
)
from services.conversation_service import ConversationService

logger = get_logger(__name__)


router = APIRouter(prefix="/projects/{project_id}/conversations", tags=["Conversations"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ConversationResponse,
    summary="Create a new conversation",
)
async def create_conversation(
    project_id: str,
    payload: ConversationCreate = ConversationCreate(),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Create a new conversation strictly scoped within the project boundary."""
    conversation = await service.create_conversation(
        project_id=project_id,
        title=payload.title,
    )
    return ConversationResponse(
        id=conversation.id,
        project_id=conversation.project_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages_count=0,
    )


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=list[ConversationResponse],
    summary="List all conversations for a project",
)
async def list_conversations(
    project_id: str,
    limit: int = Query(100, ge=1, le=100, description="Maximum conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip"),
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationResponse]:
    """Retrieve all conversations belonging to the specified project boundary."""
    conversations = await service.list_conversations(
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return [
        ConversationResponse(
            id=c.id,
            project_id=c.project_id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            messages_count=len(c.messages) if c.messages else 0,
        )
        for c in conversations
    ]


@router.get(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    response_model=ConversationDetailResponse,
    summary="Get conversation details and message history",
)
async def get_conversation(
    project_id: str,
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailResponse:
    """Retrieve a conversation and its messages under strict project isolation."""
    conversation = await service.get_conversation(
        project_id=project_id,
        conversation_id=conversation_id,
    )
    messages = await service.list_messages(
        project_id=project_id,
        conversation_id=conversation_id,
    )
    return ConversationDetailResponse(
        id=conversation.id,
        project_id=conversation.project_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages_count=len(messages),
        messages=[MessageResponse.from_model(m) for m in messages],
    )


@router.patch(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    response_model=ConversationResponse,
    summary="Update conversation title",
)
async def update_conversation(
    project_id: str,
    conversation_id: str,
    payload: ConversationUpdate,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Update mutable attributes of a conversation under project isolation."""
    conversation = await service.update_conversation(
        project_id=project_id,
        conversation_id=conversation_id,
        title=payload.title,
    )
    count = await service.get_messages_count(project_id, conversation_id)
    return ConversationResponse(
        id=conversation.id,
        project_id=conversation.project_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages_count=count,
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    project_id: str,
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    """Delete a conversation and cascade deletion to all its messages."""
    await service.delete_conversation(
        project_id=project_id,
        conversation_id=conversation_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/messages",
    status_code=status.HTTP_201_CREATED,
    summary="Send a message turn and optionally stream the BRD Agent response",
)
async def create_message(
    project_id: str,
    conversation_id: str,
    payload: MessageCreate,
    stream: bool = Query(default=True, description="Whether to stream the agent response for user messages"),
    service: ConversationService = Depends(get_conversation_service),
    agent: BRDLeadAgent = Depends(get_brd_lead_agent),
) -> Any:
    """Send and persist a message turn within a conversation under project boundary isolation.

    When a user message is sent with streaming enabled, the BRD Lead Agent executes the
    controlled BRD workflow directly. The response and phase progress are streamed back via
    Server-Sent Events (SSE). The final assistant response is automatically persisted in the
    conversation message history.
    Non-user messages or requests with stream=false are persisted directly and return HTTP 201.
    """
    # 1. Enforce Project Isolation & Conversation validation using existing services
    conversation = await service.get_conversation(
        project_id=project_id,
        conversation_id=conversation_id,
    )

    # Determine streaming mode: only user messages trigger the agent execution
    if payload.role != "user":
        effective_stream = False
    elif payload.stream is not None:
        effective_stream = payload.stream
    else:
        effective_stream = stream

    # 2. Persist message turn
    # If not streaming (e.g. assistant message turn or stream=false), persist and return immediately
    if not effective_stream:
        message = await service.create_message(
            project_id=project_id,
            conversation_id=conversation_id,
            content=payload.content,
            role=payload.role,
            metadata=payload.metadata,
        )
        return MessageResponse.from_model(message)

    # Persist the user message first using existing message infrastructure
    user_message = await service.create_message(
        project_id=project_id,
        conversation_id=conversation_id,
        content=payload.content,
        role="user",
        metadata=payload.metadata,
    )

    # 3. Create agent execution context preserving application-controlled project_id and conversation_id
    agent_run_id = str(uuid.uuid4())
    context_metadata: dict[str, Any] = {
        "user_message_id": user_message.id,
        "conversation_id": conversation_id,
        "agent_run_id": agent_run_id,
        **(payload.metadata or {}),
    }

    agent_context = AgentContext(
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=payload.metadata.get("user_id") if payload.metadata else None,
        metadata=context_metadata,
    )

    # 4. BRDLeadAgent executes under project boundary isolation

    # 5. Retrieve prior conversation message history for multi-turn thread continuity
    existing_messages = await service.list_messages(
        project_id=project_id,
        conversation_id=conversation_id,
    )
    prior_messages: list[Any] = []
    for m in existing_messages:
        if m.id == user_message.id:
            continue
        if m.role == "user":
            prior_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            prior_messages.append(AIMessage(content=m.content))

    # 6. Stream agent response back to frontend and collect assistant output
    async def event_generator() -> AsyncIterator[str]:
        accumulated_parts: list[str] = []
        start_time = time.perf_counter()
        try:
            async for event in agent.stream_async(
                request=payload.content,
                context=agent_context,
                prior_messages=prior_messages,
            ):
                if event.get("type") == "content":
                    token = event.get("content", "")
                    accumulated_parts.append(token)
                    chunk_data = json.dumps({"type": "content", "content": token})
                    yield f"event: message\ndata: {chunk_data}\n\n"
                elif event.get("type") in ("progress", "status"):
                    progress_data = json.dumps(event)
                    yield f"event: progress\ndata: {progress_data}\n\n"

            final_text = "".join(accumulated_parts).strip()
            if not final_text:
                final_text = "*(No response generated)*"

            # 7. Persist final assistant response using existing message infrastructure
            assistant_message = await service.create_message(
                project_id=project_id,
                conversation_id=conversation_id,
                content=final_text,
                role="assistant",
                metadata={
                    "user_message_id": user_message.id,
                    "agent_run_id": agent_run_id,
                    "conversation_id": conversation_id,
                    "project_id": project_id,
                    "duration_seconds": round(time.perf_counter() - start_time, 3),
                },
            )

            logger.info(
                "Assistant response persisted (project_id: %s, conversation_id: %s, user_msg_id: %s, asst_msg_id: %s, agent_run_id: %s, duration: %.2fs)",
                project_id,
                conversation_id,
                user_message.id,
                assistant_message.id,
                agent_run_id,
                time.perf_counter() - start_time,
            )

            # 8. Return completion event to the frontend
            done_data = json.dumps({
                "type": "done",
                "message": MessageResponse.from_model(assistant_message).model_dump(mode="json"),
            })
            yield f"event: done\ndata: {done_data}\n\n"

        except asyncio.CancelledError:
            logger.warning(
                "Client disconnected during streaming response (project_id: %s, conversation_id: %s, agent_run_id: %s)",
                project_id,
                conversation_id,
                agent_run_id,
            )
            raise
        except Exception as exc:
            logger.error(
                "Error during agent streaming response (project_id: %s, conversation_id: %s, agent_run_id: %s): %s",
                project_id,
                conversation_id,
                agent_run_id,
                exc,
                exc_info=True,
            )
            error_data = json.dumps({
                "type": "error",
                "error": f"Agent execution failed: {str(exc)}",
            })
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@router.get(
    "/{conversation_id}/messages",
    status_code=status.HTTP_200_OK,
    response_model=list[MessageResponse],
    summary="List messages in chronological order",
)
async def list_messages(
    project_id: str,
    conversation_id: str,
    limit: int = Query(100, ge=1, le=100, description="Maximum messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    service: ConversationService = Depends(get_conversation_service),
) -> list[MessageResponse]:
    """Retrieve ordered message history for a conversation under project isolation."""
    messages = await service.list_messages(
        project_id=project_id,
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
    )
    return [MessageResponse.from_model(m) for m in messages]
