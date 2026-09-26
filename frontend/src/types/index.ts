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

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata?: Record<string, unknown>;
  created_at?: string | null;
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
