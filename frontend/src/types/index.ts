export interface Project {
  id: string;
  name: string;
  description?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ProjectCreatePayload {
  name: string;
  description?: string;
}

export interface Conversation {
  id: string;
  project_id: string;
  title: string;
  created_at?: string | null;
  updated_at?: string | null;
  messages_count?: number | null;
}

export interface WorkflowProgressEvent {
  type: 'progress';
  phase: string;
  actor?: string;
  message: string;
  section?: string;
}

export interface WorkflowStateMetadata {
  waiting_for_user?: boolean;
  pending_clarification?: string | null;
  current_section?: string | null;
  section_progress?: Record<string, 'Not Started' | 'In Progress' | 'Completed' | 'Needs Revision' | string>;
  template_sections?: string[];
  failure_diagnostics?: Record<string, unknown> | null;
  is_complete?: boolean;
  assembled_brd?: string | null;
  [key: string]: unknown;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata?: {
    workflow_state?: WorkflowStateMetadata;
    user_message_id?: string;
    agent_run_id?: string;
    duration_seconds?: number;
    [key: string]: unknown;
  };
  created_at?: string | null;
}

export type AssistantWorkflowStatus =
  | 'RUNNING'
  | 'WAITING_FOR_CLARIFICATION'
  | 'HALTED'
  | 'COMPLETED'
  | 'NORMAL';

export function getMessageWorkflowStatus(message: Message): AssistantWorkflowStatus {
  if (message.role !== 'assistant') return 'NORMAL';
  const ws = message.metadata?.workflow_state;
  if (!ws) return 'NORMAL';
  if (ws.waiting_for_user) return 'WAITING_FOR_CLARIFICATION';
  if (ws.failure_diagnostics) return 'HALTED';
  if (
    ws.is_complete ||
    (ws.section_progress &&
      Object.keys(ws.section_progress).length > 0 &&
      Object.values(ws.section_progress).every((s) => s === 'Completed'))
  ) {
    return 'COMPLETED';
  }
  return 'NORMAL';
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface DocumentVersion {
  id: string;
  document_id: string;
  project_id: string;
  version_number: number;
  original_filename: string;
  status: 'pending' | 'indexing' | 'ready' | 'failed' | string;
  content_type?: string | null;
  size_bytes?: number | null;
  error_message?: string | null;
  indexed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface SourceDocument {
  id: string;
  project_id: string;
  name: string;
  created_at?: string | null;
  updated_at?: string | null;
  latest_version?: DocumentVersion | null;
  versions_count: number;
}

