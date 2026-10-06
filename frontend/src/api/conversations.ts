import { apiRequest, API_BASE } from './client.ts';
import { tokenService } from '../services/tokenService.ts';
import type { Conversation, ConversationDetail, LLMExecutionSummary, Message, WorkflowProgressEvent } from '../types/index.ts';

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
  onProgress?: (progress: WorkflowProgressEvent) => void;
  onExecutionSummary?: (summary: LLMExecutionSummary) => void;
  onDone?: (finalMessage: Message) => void;
  onError?: (error: string) => void;
  signal?: AbortSignal;
}

export async function sendMessageStream({
  projectId,
  conversationId,
  content,
  onToken,
  onProgress,
  onExecutionSummary,
  onDone,
  onError,
  signal,
}: SendMessageStreamOptions): Promise<void> {
  const url = `${API_BASE}/projects/${projectId}/conversations/${conversationId}/messages?stream=true`;
  const token = tokenService.getAccessToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const response = await fetch(url, {
    method: 'POST',
    headers,
    signal,
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
  let streamCompleted = false;

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
            } else if (currentEvent === 'progress' || parsed.type === 'progress') {
              onProgress?.(parsed as WorkflowProgressEvent);
            } else if (currentEvent === 'execution_summary' || parsed.type === 'execution_summary') {
              onExecutionSummary?.((parsed.execution_summary || parsed) as LLMExecutionSummary);
            } else if (currentEvent === 'done' || parsed.type === 'done') {
              streamCompleted = true;
              if (parsed.message) {
                onDone?.(parsed.message as Message);
              }
            } else if (currentEvent === 'error' || parsed.type === 'error') {
              streamCompleted = true;
              const errMsg = parsed.error || 'Unknown agent error';
              onError?.(errMsg);
            }
            // Unknown event types are safely ignored without throwing
          } catch {
            // If data is raw text
            if (currentEvent === 'message' && rawData) {
              onToken?.(rawData);
            }
          }
        }
      }
    }

    if (!streamCompleted) {
      const unexpectedError = 'Stream closed unexpectedly before completion';
      onError?.(unexpectedError);
      throw new Error(unexpectedError);
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err);
    onError?.(errMsg);
    throw err;
  }
}
