// =============================================================================
// Chat Service — CRUD operations for conversations and messages
// =============================================================================

import apiClient, { getApiBaseUrl } from './api.ts';
import { tokenService } from './tokenService.ts';
import { refreshTokens } from './authService.ts';
import type {
  Conversation,
  CreateConversationPayload,
  UpdateConversationPayload,
  Message,
  SendMessagePayload,
  PaginatedResponse,
  WorkflowProgressEvent,
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
  onProgress?: (data: WorkflowProgressEvent) => void;
  onMessage?: (content: string) => void;
  onDone?: (message: Message) => void;
  onError?: (error: Error) => void;
  signal?: AbortSignal;
}

/** Stream the assistant's response via SSE */
export async function streamMessage(
  projectId: string,
  conversationId: string,
  payload: SendMessagePayload,
  handlers: StreamHandlers
): Promise<void> {
  const baseURL = getApiBaseUrl();
  const url = `${baseURL}/projects/${projectId}/conversations/${conversationId}/messages?stream=true`;
  
  const getHeaders = (token: string | null) => {
    const h: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    };
    if (token && token !== 'undefined' && token !== 'null') {
      h.Authorization = `Bearer ${token}`;
    }
    return h;
  };

  try {
    if (handlers.signal?.aborted) return;

    let token = tokenService.getAccessToken();
    let response = await fetch(url, {
      method: 'POST',
      headers: getHeaders(token),
      body: JSON.stringify(payload),
      signal: handlers.signal,
    });

    // Handle expired token with automatic refresh retry
    if (response.status === 401 && !handlers.signal?.aborted) {
      try {
        const newTokens = await refreshTokens();
        token = newTokens.access_token;
        response = await fetch(url, {
          method: 'POST',
          headers: getHeaders(token),
          body: JSON.stringify(payload),
          signal: handlers.signal,
        });
      } catch (refreshErr) {
        throw new Error('Authentication expired. Please log in again.');
      }
    }

    if (handlers.signal?.aborted) return;

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
      if (handlers.signal?.aborted) {
        await reader.cancel().catch(() => {});
        return;
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/(?:\r?\n){2}/);
      buffer = blocks.pop() || ''; // Keep the last incomplete block in the buffer

      for (const block of blocks) {
        if (!block.trim()) continue;
        const blockLines = block.split(/\r?\n/);
        let eventType = 'message';
        let eventData = '';

        for (const line of blockLines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('event:')) {
            eventType = trimmed.replace(/^event:\s*/, '').trim();
          } else if (trimmed.startsWith('data:')) {
            eventData += trimmed.replace(/^data:\s*/, '');
          }
        }

        if (eventData) {
          try {
            const parsed = JSON.parse(eventData);
            if ((eventType === 'progress' || parsed.type === 'progress') && handlers.onProgress) {
              handlers.onProgress(parsed);
            } else if ((eventType === 'message' || parsed.type === 'content') && handlers.onMessage) {
              handlers.onMessage(parsed.content || '');
            } else if ((eventType === 'done' || parsed.type === 'done') && handlers.onDone) {
              handlers.onDone(parsed.message);
            } else if ((eventType === 'error' || parsed.type === 'error') && handlers.onError) {
              handlers.onError(new Error(parsed.error || 'Stream error'));
            }
          } catch (e) {
            // Raw text fallback if not JSON
            if (eventType === 'message' && eventData && handlers.onMessage) {
              handlers.onMessage(eventData);
            }
          }
        }
      }
    }
  } catch (error) {
    if (handlers.signal?.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
      // Aborted intentionally by client; cleanly return
      return;
    }
    if (handlers.onError) {
      handlers.onError(error instanceof Error ? error : new Error(String(error)));
    } else {
      throw error;
    }
  }
}

