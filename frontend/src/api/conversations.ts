import { apiRequest, API_BASE } from './client.ts';
import type { Conversation, ConversationDetail, Message } from '../types/index.ts';

export async function listConversations(projectId: string): Promise<Conversation[]> {
  return apiRequest<Conversation[]>(`/projects/${projectId}/conversations`);
}

export async function createConversation(projectId: string, title?: string): Promise<Conversation> {
  return apiRequest<Conversation>(`/projects/${projectId}/conversations`, {
    method: 'POST',
    body: JSON.stringify(title ? { title } : {}),
  });
}

export async function getConversation(projectId: string, conversationId: string): Promise<ConversationDetail> {
  return apiRequest<ConversationDetail>(`/projects/${projectId}/conversations/${conversationId}`);
}

export async function listMessages(projectId: string, conversationId: string): Promise<Message[]> {
  return apiRequest<Message[]>(`/projects/${projectId}/conversations/${conversationId}/messages`);
}

export interface SendMessageStreamOptions {
  projectId: string;
  conversationId: string;
  content: string;
  onToken?: (token: string) => void;
  onDone?: (finalMessage: Message) => void;
  onError?: (error: string) => void;
}

export async function sendMessageStream({
  projectId,
  conversationId,
  content,
  onToken,
  onDone,
  onError,
}: SendMessageStreamOptions): Promise<void> {
  const url = `${API_BASE}/projects/${projectId}/conversations/${conversationId}/messages?stream=true`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      role: 'user',
      content,
    }),
  });

  if (!response.ok) {
    let errorDetail = `Message request failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = String(errJson.detail);
    } catch {
      const text = await response.text().catch(() => '');
      if (text) errorDetail = text;
    }
    onError?.(errorDetail);
    throw new Error(errorDetail);
  }

  if (!response.body) {
    const errorMsg = 'Response body is empty';
    onError?.(errorMsg);
    throw new Error(errorMsg);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      // Keep incomplete trailing line in buffer
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          currentEvent = 'message';
          continue;
        }

        if (trimmed.startsWith('event:')) {
          currentEvent = trimmed.replace('event:', '').trim();
        } else if (trimmed.startsWith('data:')) {
          const rawData = trimmed.replace('data:', '').trim();
          try {
            const parsed = JSON.parse(rawData);
            if (currentEvent === 'message' || parsed.type === 'content') {
              if (parsed.content) {
                onToken?.(parsed.content);
              }
            } else if (currentEvent === 'done' || parsed.type === 'done') {
              if (parsed.message) {
                onDone?.(parsed.message as Message);
              }
            } else if (currentEvent === 'error' || parsed.type === 'error') {
              const errMsg = parsed.error || 'Unknown agent error';
              onError?.(errMsg);
            }
          } catch {
            // If data is raw text
            if (currentEvent === 'message' && rawData) {
              onToken?.(rawData);
            }
          }
        }
      }
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    onError?.(errMsg);
    throw err;
  }
}
