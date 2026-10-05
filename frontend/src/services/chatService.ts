// =============================================================================
// Chat Service — CRUD operations for conversations and messages
// =============================================================================

import apiClient from './api.ts';
import { tokenService } from './tokenService.ts';
import { refreshTokens } from './authService.ts';
import type {
  Conversation,
  CreateConversationPayload,
  UpdateConversationPayload,
  Message,
  SendMessagePayload,
  PaginatedResponse,
} from '../types/index.ts';

function base(projectId: string) {
  return `projects/${projectId}/conversations`;
}

/** Fetch a paginated list of conversations for a project. */
export async function listConversations(
  projectId: string,
  limit = 20,
  offset = 0,
): Promise<PaginatedResponse<Conversation>> {
  // Backend returns a plain array, so we wrap it for the frontend
  const { data } = await apiClient.get<Conversation[]>(base(projectId), {
    params: { limit, offset },
  });
  const items = Array.isArray(data) ? data : [];
  return {
    items,
    total: items.length,
    limit,
    offset,
  };
}

/** Retrieve a single conversation by ID, including its messages. */
export async function getConversation(
  projectId: string,
  conversationId: string,
): Promise<Conversation> {
  const { data } = await apiClient.get<Conversation>(`${base(projectId)}/${conversationId}`);
  return data;
}

/** Create a new conversation thread in a project. */
export async function createConversation(
  projectId: string,
  payload: CreateConversationPayload,
): Promise<Conversation> {
  const { data } = await apiClient.post<Conversation>(base(projectId), payload);
  return data;
}

/** Rename a conversation. */
export async function updateConversation(
  projectId: string,
  conversationId: string,
  payload: UpdateConversationPayload,
): Promise<Conversation> {
  const { data } = await apiClient.patch<Conversation>(`${base(projectId)}/${conversationId}`, payload);
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
  _generate: boolean = true,
): Promise<Message> {
  const { data } = await apiClient.post<Message>(`${base(projectId)}/${conversationId}/messages`, payload, {
    params: { stream: false },
    // Long timeout for AI generation
    timeout: 120_000,
  });
  return data;
}

export interface StreamHandlers {
  onProgress?: (data: any) => void;
  onMessage?: (content: string) => void;
  onDone?: (message: Message) => void;
  onError?: (error: Error) => void;
}

/** Stream the assistant's response via SSE */
export async function streamMessage(
  projectId: string,
  conversationId: string,
  payload: SendMessagePayload,
  handlers: StreamHandlers
): Promise<void> {
  const env = typeof import.meta !== 'undefined' && 'env' in import.meta ? (import.meta as { env?: Record<string, string> }).env : undefined;
  const baseURL = env?.VITE_API_BASE_URL || '/api';
  const url = `${baseURL.replace(/\/$/, '')}/projects/${projectId}/conversations/${conversationId}/messages?stream=true`;
  
  const getHeaders = (token: string | null) => {
    const h: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) {
      h.Authorization = `Bearer ${token}`;
    }
    return h;
  };

  try {
    let token = tokenService.getAccessToken();
    let response = await fetch(url, {
      method: 'POST',
      headers: getHeaders(token),
      body: JSON.stringify(payload),
    });

    // Handle expired token with automatic refresh retry
    if (response.status === 401) {
      try {
        const newTokens = await refreshTokens();
        token = newTokens.access_token;
        response = await fetch(url, {
          method: 'POST',
          headers: getHeaders(token),
          body: JSON.stringify(payload),
        });
      } catch (refreshErr) {
        throw new Error('Authentication expired. Please log in again.');
      }
    }

    if (!response.ok) {
      const text = await response.text();
      let msg = text;
      try {
        const errJson = JSON.parse(text);
        if (errJson.detail) msg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } catch (e) {}
      throw new Error(msg || `HTTP Error ${response.status}`);
    }

    if (!response.body) throw new Error('No response body');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || ''; // Keep the last incomplete block in the buffer

      for (const block of lines) {
        if (!block.trim()) continue;
        const blockLines = block.split('\n');
        let eventType = 'message';
        let eventData = '';

        for (const line of blockLines) {
          if (line.startsWith('event: ')) {
            eventType = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            eventData += line.substring(6);
          }
        }

        if (eventData) {
          try {
            const parsed = JSON.parse(eventData);
            if (eventType === 'progress' && handlers.onProgress) {
              handlers.onProgress(parsed);
            } else if (eventType === 'message' && handlers.onMessage) {
              handlers.onMessage(parsed.content || '');
            } else if (eventType === 'done' && handlers.onDone) {
              handlers.onDone(parsed.message);
            } else if (eventType === 'error' && handlers.onError) {
              handlers.onError(new Error(parsed.error || 'Stream error'));
            }
          } catch (e) {
            console.warn('Failed to parse SSE data:', eventData);
          }
        }
      }
    }
  } catch (error) {
    if (handlers.onError) {
      handlers.onError(error instanceof Error ? error : new Error(String(error)));
    } else {
      throw error;
    }
  }
}

