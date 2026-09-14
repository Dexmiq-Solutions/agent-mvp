# Conversation & Message Flow Architecture

## 1. Overview & Purpose

The **Conversation & Message Flow** layer provides the persistent, application-level conversation boundary for project-scoped interactions in Dexmiq's RAG-backed system. 

It establishes an isolated, auditable record of multi-turn user dialogues within a tenant project, decoupling conversation storage and lifecycle from retrieval execution and future model generation.

```
                  ┌───────────────────────┐
                  │        Project        │  (Tenant Isolation Root)
                  └──────────┬────────────┘
                             │ 1
                             ▼ *
                  ┌───────────────────────┐
                  │     Conversation      │  (Dialogue Context Boundary)
                  └──────────┬────────────┘
                             │ 1
                             ▼ *
                  ┌───────────────────────┐
                  │        Message        │  (Append-Only Chronological Turn)
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │      RAG Service      │  (Project-Scoped Retrieval)
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │   Generation Service  │  (Future Phase: LLM Response)
                  └───────────────────────┘
```

---

## 2. Relational Hierarchy & Ownership

The system strictly enforces a three-level hierarchical entity ownership:

```
Project (id)
  └── Conversation (id, project_id)
        └── Message (id, conversation_id, role, content, metadata, created_at)
```

### Relational Model Guarantees:
1. **Single Ownership**:
   - A `Conversation` belongs to exactly one `Project`.
   - A `Message` belongs to exactly one `Conversation`.
2. **No Redundant Keys**:
   - `MessageModel` contains `conversation_id`, avoiding a duplicate `project_id` column.
   - The authoritative source of project ownership is `Message.conversation_id -> Conversation.project_id`.
3. **Cascading Deletions**:
   - Foreign key constraints on both PostgreSQL and SQLite specify `ON DELETE CASCADE`. Deleting a project cascades to its conversations and messages; deleting a conversation cascades to all contained messages.

---

## 3. Database Schema & Indexing

### Conversations Table (`conversations`)
| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | `VARCHAR(255)` | No | Primary Key (UUIDv4) |
| `project_id` | `VARCHAR(255)` | No | Foreign Key (`projects.id`, ON DELETE CASCADE) |
| `title` | `VARCHAR(255)` | Yes | Display title (default: "New Conversation") |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Yes | Creation timestamp |
| `updated_at` | `TIMESTAMP WITH TIME ZONE` | Yes | Modification timestamp |

**Indexes**:
- `ix_conversations_project_id`: Fast lookup of conversations by project.
- `ix_conversations_project_id_updated_at`: Fast sorted listing (`updated_at DESC`) under project isolation.

### Messages Table (`messages`)
| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | `VARCHAR(255)` | No | Primary Key (UUIDv4) |
| `conversation_id` | `VARCHAR(255)` | No | Foreign Key (`conversations.id`, ON DELETE CASCADE) |
| `role` | `VARCHAR(50)` | No | Role identifier: `'user'` or `'assistant'` |
| `content` | `TEXT` | No | Textual payload of the conversational turn |
| `metadata` | `JSON` | Yes | Extensible structured metadata |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Yes | Creation timestamp |

**Indexes**:
- `ix_messages_conversation_id`: Fast lookup of turns by conversation.
- `ix_messages_conversation_id_created_at`: Deterministic chronological ordering (`created_at ASC`).

---

## 4. Project Isolation Boundary

Project isolation is a mandatory architectural invariant across all operations:

1. **Existence Verification**: All operations verify that the target `project_id` exists before proceeding.
2. **Ownership Validation**:
   - Whenever accessing a conversation, `conversation.project_id == project_id` is asserted.
   - If the conversation does not exist, `ConversationNotFoundError` (HTTP 404) is raised.
   - If the conversation exists but belongs to a different project, `ProjectConversationMismatchError` (mapped to HTTP 404) is raised. Cross-project data access is strictly prevented.
3. **Indirect Message Ownership**:
   - Message creation and retrieval endpoints require both `project_id` and `conversation_id`.
   - The conversation is validated first under the project boundary, ensuring messages can never be accessed or created across project boundaries.

---

## 5. Lifecycles

### Conversation Lifecycle
1. **Creation**: `POST /projects/{project_id}/conversations`
   - Validates project existence.
   - Assigns a stable UUIDv4 identifier.
   - Sets initial title (defaults to `"New Conversation"` if omitted).
   - Flushes transaction and returns `ConversationResponse` with 0 initial messages.
2. **Listing**: `GET /projects/{project_id}/conversations`
   - Scoped strictly to `project_id`.
   - Returns paginated list sorted by `updated_at DESC`.
3. **Retrieval**: `GET /projects/{project_id}/conversations/{conversation_id}`
   - Verifies project ownership.
   - Returns conversation details along with chronological message history.
4. **Update**: `PATCH /projects/{project_id}/conversations/{conversation_id}`
   - Updates conversation title and touches `updated_at`.
5. **Deletion**: `DELETE /projects/{project_id}/conversations/{conversation_id}`
   - Verifies project ownership.
   - Deletes conversation entity; relational cascade deletes all dependent messages.

### Message Lifecycle
1. **Creation**: `POST /projects/{project_id}/conversations/{conversation_id}/messages`
   - Verifies project and conversation ownership.
   - Validates role (`"user"` or `"assistant"`).
   - Validates non-empty content.
   - Persists message and updates conversation `updated_at`.
   - Returns persisted `MessageResponse`.
2. **Retrieval**: `GET /projects/{project_id}/conversations/{conversation_id}/messages`
   - Verifies project and conversation ownership.
   - Returns messages ordered by `created_at ASC, id ASC`.

---

## 6. Message Role Semantics

Roles are strictly constrained to core conversational participants:
- **`user`**: The human user's input/query.
- **`assistant`**: The AI assistant's generated response.

Agent-specific roles (such as `agent`, `tool`, `system`, or `retriever`) are intentionally excluded to keep the conversation model agnostic and decoupled.

---

## 7. Service Boundary & RAG Integration

The Conversation layer sits directly between the application API and the RAG/Generation services:

```
[User Request]
       │
       ▼
[Conversation API]
       │
       ▼
[ConversationService]
       │ (1. Persist User Message)
       ├──────────────────────────────────────────┐
       │ (2. Retrieve Context via Helper)         │
       ▼                                          ▼
[get_conversation_context()]             [PostgreSQL Database]
       │                                 (Persistent conversations & messages)
       ▼
[RAGService.retrieve(project_id, query)] 
       │
       ▼
[Retrieved Context & Chunks]
       │
       ▼
[Future: GenerationService.generate()]
       │
       ▼
[ConversationService.create_message(role='assistant')]
```

### Architectural Guarantees:
- **No Vector Search in Conversation Service**: `ConversationService` does not import Qdrant, embeddings, or chunk models.
- **No Direct LLM Invocations**: `ConversationService` performs no prompt construction or model inference.
- **Context Helper**: `ConversationService.get_conversation_context(project_id, conversation_id, max_messages)` provides a clean interface for subsequent generation stages to retrieve recent turns.

---

## 8. Transactions & Concurrency

1. **Short Transaction Windows**: Database sessions are flushed or committed immediately upon message or conversation persistence. Database transactions are never held open across network calls, embeddings, or vector retrieval.
2. **Deterministic Message Ordering**: Messages are queried using `ORDER BY created_at ASC, id ASC` to guarantee deterministic ordering even when timestamps coincide.
3. **Single Query Retrieval**: Message histories are retrieved in a single batched query, preventing N+1 overhead.

---

## 9. Observability & Tracing

All conversation operations use the centralized logger (`app.core.logging.get_logger`):
- Events log identifiers: `project_id`, `conversation_id`, `message_id`, and `role`.
- **Privacy Protection**: Message contents are never logged in plaintext; only message length (`content_length`) is captured.

---

## 10. Explicit Exclusions & Deferred Work

The following items are intentionally excluded from this phase and deferred to subsequent layers:
- **LLM Generation**: Assistant answer synthesis is deferred to the Generation Service.
- **Title Generation**: Automatic conversation titling via LLM is deferred.
- **History Summarization**: Token-budgeted history compression is deferred.
- **Agent Orchestration**: LangGraph, DeepAgents, MCP, and multi-agent coordination are not part of this layer.
