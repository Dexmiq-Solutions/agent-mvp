// =============================================================================
// Dexmiq AI Agent Workspace — Type Definitions
// Aligned with Backend API Contract v1.0.0
// =============================================================================

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/** Root workspace entity. All conversations and sources belong to a project. */
export interface Project {
  id: string;            // UUID
  name: string;          // 1–255 characters
  description: string | null;
  created_at: string;    // ISO-8601
  updated_at: string;    // ISO-8601
}

/** Payload for creating a new project. */
export interface CreateProjectPayload {
  name: string;
  description?: string | null;
}

/** Payload for updating an existing project (partial). */
export interface UpdateProjectPayload {
  name?: string;
  description?: string | null;
}

// ---------------------------------------------------------------------------
// Source Document & Versioning
// ---------------------------------------------------------------------------

/** Processing status of a document version in the RAG pipeline. */
export type DocumentVersionStatus = 'pending' | 'indexing' | 'ready' | 'failed';

/** A specific version of an uploaded source document. */
export interface DocumentVersion {
  id: string;                           // UUID
  document_id: string;                  // UUID
  project_id: string;                   // UUID
  version_number: number;
  original_filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentVersionStatus;
  error_message: string | null;
  indexed_at: string | null;            // ISO-8601
  created_at: string;                   // ISO-8601
  updated_at: string;                   // ISO-8601
}

/** A knowledge-base source document belonging to a project. */
export interface SourceDocument {
  id: string;                           // UUID
  project_id: string;                   // UUID
  name: string;
  created_at: string;                   // ISO-8601
  updated_at: string;                   // ISO-8601
  latest_version: DocumentVersion | null;
  versions_count: number;
}

/** Payload for renaming a source document. */
export interface UpdateSourcePayload {
  name: string;
}

// ---------------------------------------------------------------------------
// Conversation & Messages
// ---------------------------------------------------------------------------

/** Role of a message in a conversation turn. */
export type MessageRole = 'user' | 'assistant';

/** A single message turn inside a conversation. */
export interface Message {
  id: string;                           // UUID
  conversation_id: string;              // UUID
  role: MessageRole;
  content: string;
  metadata: TelemetryMetadata;
  created_at: string;                   // ISO-8601
}

/** A conversation thread scoped to a project. */
export interface Conversation {
  id: string;                           // UUID
  project_id: string;                   // UUID
  title: string;
  created_at: string;                   // ISO-8601
  updated_at: string;                   // ISO-8601
  messages_count: number;
  messages: Message[];
}

/** Payload for creating a new conversation. */
export interface CreateConversationPayload {
  title?: string;
}

/** Payload for updating a conversation (e.g. rename). */
export interface UpdateConversationPayload {
  title: string;
}

/** Payload for sending a message in a conversation. */
export interface SendMessagePayload {
  content: string;
}

// ---------------------------------------------------------------------------
// Telemetry / Metadata
// ---------------------------------------------------------------------------

/**
 * Metadata returned alongside assistant messages.
 * Contains RAG retrieval context, model info, and quality-gate results.
 * The backend may include additional keys; we type the known ones here.
 */
export interface TelemetryMetadata {
  model?: string;
  retrieval_results?: RetrievalResult[];
  groundedness_score?: number;
  processing_time_ms?: number;
  [key: string]: unknown;               // Allow extra backend-supplied fields
}

/** A single retrieval result from the hybrid RAG pipeline. */
export interface RetrievalResult {
  chunk_text: string;
  source_document_id: string;
  source_document_name: string;
  relevance_score: number;
}

// ---------------------------------------------------------------------------
// Paginated Response Wrapper
// ---------------------------------------------------------------------------

/** Generic paginated list response from the backend. */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// API Error
// ---------------------------------------------------------------------------

/** Structured error shape returned by the backend. */
export interface ApiError {
  detail: string | { msg: string; type: string }[];
}
