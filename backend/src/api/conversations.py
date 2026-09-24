"""FastAPI router for project-scoped Conversation and Message operations."""

from fastapi import APIRouter, Depends, Query, Response, status

from api.dependencies import get_conversation_service
from schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageResponse,
)
from services.conversation_service import ConversationService


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
    response_model=MessageResponse,
    summary="Send and persist a message turn",
)
async def create_message(
    project_id: str,
    conversation_id: str,
    payload: MessageCreate,
    service: ConversationService = Depends(get_conversation_service),
) -> MessageResponse:
    """Persist a message turn within a conversation under project boundary isolation."""
    message = await service.create_message(
        project_id=project_id,
        conversation_id=conversation_id,
        content=payload.content,
        role=payload.role,
        metadata=payload.metadata,
    )
    return MessageResponse.from_model(message)



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
