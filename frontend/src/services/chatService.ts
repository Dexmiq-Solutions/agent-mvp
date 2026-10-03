// =============================================================================
// Chat Service — CRUD operations for conversations and messages
// =============================================================================

import { mockChatService } from './mockData';
import type {
  Conversation,
  CreateConversationPayload,
  UpdateConversationPayload,
  Message,
  SendMessagePayload,
  PaginatedResponse,
} from '../types';

/** Fetch a paginated list of conversations for a project. */
export async function listConversations(
  projectId: string,
  limit = 20,
  offset = 0,
): Promise<PaginatedResponse<Conversation>> {
  return mockChatService.listConversations(projectId, limit, offset);
}

/** Retrieve a single conversation by ID, including its messages. */
export async function getConversation(
  projectId: string,
  conversationId: string,
): Promise<Conversation> {
  return mockChatService.getConversation(projectId, conversationId);
}

/** Create a new conversation thread in a project. */
export async function createConversation(
  projectId: string,
  payload: CreateConversationPayload,
): Promise<Conversation> {
  return mockChatService.createConversation(projectId, payload);
}

/** Rename a conversation. */
export async function updateConversation(
  projectId: string,
  conversationId: string,
  payload: UpdateConversationPayload,
): Promise<Conversation> {
  return mockChatService.updateConversation(projectId, conversationId, payload);
}

/** Delete a conversation. */
export async function deleteConversation(
  projectId: string,
  conversationId: string,
): Promise<void> {
  return mockChatService.deleteConversation(projectId, conversationId);
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
  return mockChatService.sendMessage(projectId, conversationId, payload, generate);
}
