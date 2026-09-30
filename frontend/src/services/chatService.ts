// =============================================================================
// Chat Service — CRUD operations for conversations and messages
// =============================================================================

import apiClient from './api';
import type {
  Conversation,
  CreateConversationPayload,
  UpdateConversationPayload,
  Message,
  SendMessagePayload,
  PaginatedResponse,
} from '../types';

function base(projectId: string) {
  return `projects/${projectId}/conversations`;
}

/** Fetch a paginated list of conversations for a project. */
export async function listConversations(
  projectId: string,
  limit = 20,
  offset = 0,
): Promise<PaginatedResponse<Conversation>> {
  const { data } = await apiClient.get<PaginatedResponse<Conversation>>(
    base(projectId),
    { params: { limit, offset } },
  );
  return data;
}

/** Retrieve a single conversation by ID, including its messages. */
export async function getConversation(
  projectId: string,
  conversationId: string,
): Promise<Conversation> {
  const { data } = await apiClient.get<Conversation>(
    `${base(projectId)}/${conversationId}`,
  );
  return data;
}

/** Create a new conversation thread in a project. */
export async function createConversation(
  projectId: string,
  payload: CreateConversationPayload,
): Promise<Conversation> {
  const { data } = await apiClient.post<Conversation>(
    base(projectId),
    payload,
  );
  return data;
}

/** Rename a conversation. */
export async function updateConversation(
  projectId: string,
  conversationId: string,
  payload: UpdateConversationPayload,
): Promise<Conversation> {
  const { data } = await apiClient.patch<Conversation>(
    `${base(projectId)}/${conversationId}`,
    payload,
  );
  return data;
}

/** Delete a conversation. */
export async function deleteConversation(
  projectId: string,
  conversationId: string,
): Promise<void> {
  await apiClient.delete(`${base(projectId)}/${conversationId}`);
}

/**
 * Send a message to a conversation.
 * 
 * @param generate If true, triggers the RAG pipeline and AI response generation.
 *                 If false, simply appends the user message to the thread.
 */
export async function sendMessage(
  projectId: string,
  conversationId: string,
  payload: SendMessagePayload,
  generate: boolean = true,
): Promise<Message> {
  const { data } = await apiClient.post<Message>(
    `${base(projectId)}/${conversationId}/messages`,
    payload,
    { 
      params: { generate },
      // Generative turns might take longer to process
      timeout: generate ? 120_000 : 30_000, 
    },
  );
  return data;
}
