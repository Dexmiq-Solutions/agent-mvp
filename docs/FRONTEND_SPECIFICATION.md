# Frontend Seed & Technical Implementation Specification

**Project**: Dexmiq AI Agent Workspace (Agent MVP)  
**Document Version**: 1.0.0  
**Target Audience**: Incoming Frontend Engineer  
**Date**: September 16, 2026  
**Backend Authority**: `c:\DEXMIQ_PROJECTS\agent-mvp\backend` (FastAPI / SQLAlchemy / PostgreSQL / Qdrant / Voyage / OpenAI)  
**Frontend Location**: `c:\DEXMIQ_PROJECTS\agent-mvp\frontend` (Greenfield Web Application)

---

## 1. Executive Summary & Document Intent

Welcome to the **Dexmiq AI Agent Workspace** project!

This document is your single source of truth for engineering the initial frontend web application. It bridges the **product and UX vision** (a project-centric AI workspace modeled after ChatGPT Projects and Claude Workspaces) with the **actual technical reality of the backend codebase**.

### Key Rules of Engagement:
1. **The Backend Repository is Technical Truth**: Every endpoint, model, field, and constraint described here is audited directly from the backend codebase. Do not invent endpoints or fields that are not documented here.
2. **The Product UX is Strategic Direction**: The UI layout, screen hierarchy, and interaction patterns reflect the intended user experience. Where the backend does not yet support a specific UI element (e.g., search query filtering or user authentication), this document explicitly marks it as a **Backend Gap** and provides the client-side bridge.
3. **No Unimplemented Features as Working UX**: Never present planned agent features (such as token streaming, autonomous agent tool loops, or voice synthesis) as fully working features if the backend currently operates synchronously. Clearly manage user expectations through loading states and telemetry displays.

---

## A. Product Overview

### What the Product Is
The application is a **Project-Centric AI Agent Workspace**. Instead of interacting with a generic, disconnected chatbot, the user organizes their work into discrete **Projects** (workspaces).

### The Project-Centric Model
A **Project** acts as the root boundary for:
1. **Conversations (Chats)**: Threaded dialogues where the user consults an AI assistant equipped with context retrieved from the project's knowledge base.
2. **Sources (Knowledge Base)**: Project-scoped documents (markdown, text, Word documents) uploaded by the user that are ingested, chunked, vectorized, and searched via a hybrid Retrieval-Augmented Generation (RAG) pipeline.
3. **Future Agent Execution**: Autonomous agent workflows, tool calling, and human-in-the-loop task orchestration (currently in architectural staging).

```
┌────────────────────────────────────────────────────────┐
│                   Dexmiq Application                   │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │   Project A   │           │   Project B   │
      └───────┬───────┘           └───────┬───────┘
              │                                   │
      ┌───────┴───────┐                   ┌───────┴───────┐
      ▼               ▼                   ▼               ▼
┌───────────┐   ┌───────────┐       ┌───────────┐   ┌───────────┐
│   Chats   │   │  Sources  │       │   Chats   │   │  Sources  │
│ (Threads) │   │    (KB)   │       │ (Threads) │   │    (KB)   │
└─────┬─────┘   └─────┬─────┘       └─────┬─────┘   └─────┬─────┘
      │               │                   │               │
      ▼               ▼                   ▼               ▼
  Messages        Chunks &            Messages        Chunks &
  (Turns)          Vectors            (Turns)          Vectors
```

### High-Level User Journey
1. **Initial Entry**: When a user first opens the app, they create a named project workspace.
2. **Projects Hub**: A selector screen lists all available projects, with quick creation and search.
3. **Project Home**: Entering a project reveals the project dashboard containing a prominent **New Chat** composer, a **Chats** tab (previous conversations), and a **Sources** tab (knowledge base documents).
4. **Seamless Transition**: Submitting a query in the Project Home composer immediately creates a new conversation and seamlessly transitions the user into the active conversation interface.
5. **RAG-Powered Chat**: When a message is sent with `generate=true`, the backend retrieves relevant passages exclusively from that project's uploaded sources, feeds them to the LLM, passes the answer through a groundedness quality gate, and returns an assistant response turn.

---

## B. Current Implementation Status Matrix

The backend is built with **FastAPI**, **SQLAlchemy (asyncpg)**, **PostgreSQL Content Store**, **Supabase Object Storage**, **Qdrant Vector Store**, **Voyage Embeddings**, and an **OpenAI-compatible LLM Gateway**.

| Capability | Backend Status | Relevant Code / Endpoints | Frontend Availability | Implementation Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Project Creation** | ✅ Implemented | `POST /projects` | ✅ Usable Now | Validates name (1-255 chars, non-whitespace). |
| **Project Listing** | ✅ Implemented | `GET /projects` | ✅ Usable Now | Supports `limit` & `offset`. Ordered by `created_at DESC`. |
| **Project Retrieval** | ✅ Implemented | `GET /projects/{project_id}` | ✅ Usable Now | Returns project metadata. |
| **Project Update** | ✅ Implemented | `PATCH /projects/{project_id}` | ✅ Usable Now | Updates `name` and `description`. |
| **Project Deletion** | ✅ Implemented | `DELETE /projects/{project_id}` | ✅ Usable Now | Cascades to conversations, messages, and storage files. |
| **Project Search** | ⚠️ Partial | `GET /projects` | ⚠️ Client-Side Only | Backend lacks `?query=` param; frontend filters client-side. |
| **Project Pinning / Sharing** | ❌ Not Implemented | None | ❌ Planned | UI shows pins & tabs; no backend field. LocalStorage fallback. |
| **Source Upload** | ✅ Implemented | `POST /projects/{id}/sources` | ✅ Usable Now | Multipart file upload (`file`, optional `name`). Stores in Supabase. |
| **Source Listing** | ✅ Implemented | `GET /projects/{id}/sources` | ✅ Usable Now | Lists documents with `latest_version` and status. |
| **Source Retrieval** | ✅ Implemented | `GET /projects/{id}/sources/{id}` | ✅ Usable Now | Returns document detail with all version records. |
| **Source Rename** | ✅ Implemented | `PATCH /projects/{id}/sources/{id}` | ✅ Usable Now | Modifies document display `name`. |
| **Source Deletion** | ✅ Implemented | `DELETE /projects/{id}/sources/{id}`| ✅ Usable Now | Cascades to versions, chunks, and storage objects. |
| **Source Versioning** | ✅ Implemented | `POST .../sources/{id}/versions` | ✅ Usable Now | Uploads new file revision to existing document. |
| **Source Processing (RAG Indexing)** | ✅ Implemented | `AUTO_PROCESS_DOCUMENTS=True` | ✅ Usable Now | Triggered automatically on upload. Supports `.txt`, `.md`, `.docx`. |
| **Manual Re-indexing / Retry** | ❌ Not Implemented | None | ❌ Backend Gap | No endpoint to retry failed indexing without re-uploading. |
| **PDF Ingestion** | ❌ Not Implemented | `rag/parsing/` | ❌ Backend Gap | `PDFParser` not registered in registry. Only `.md`, `.txt`, `.docx`. |
| **Conversation Creation** | ✅ Implemented | `POST /projects/{id}/conversations`| ✅ Usable Now | Accepts optional `title` (default: "New Conversation"). |
| **Conversation Listing** | ✅ Implemented | `GET /projects/{id}/conversations` | ✅ Usable Now | Ordered by `updated_at DESC`. Includes `messages_count`. |
| **Conversation Detail** | ✅ Implemented | `GET /projects/{id}/conversations/{id}`| ✅ Usable Now | Returns full chronological message history. |
| **Conversation Title Update** | ✅ Implemented | `PATCH .../conversations/{id}` | ✅ Usable Now | Renames conversation title. |
| **Conversation Deletion** | ✅ Implemented | `DELETE .../conversations/{id}` | ✅ Usable Now | Cascades to all messages. |
| **Send Message (Plain Turn)** | ✅ Implemented | `POST .../messages?generate=false` | ✅ Usable Now | Persists user or assistant turn without RAG/LLM. |
| **Send Message + AI Generation** | ✅ Implemented | `POST .../messages?generate=true` | ✅ Usable Now | Persists user turn, executes RAG + LLM, returns assistant turn. |
| **Message Streaming (SSE / WS)** | ❌ Not Implemented | `LLM_STREAMING_ENABLED=False` | ❌ Backend Gap | Calls are synchronous REST HTTP POST requests. |
| **Single Message Edit/Delete** | ❌ Not Implemented | None | ❌ Backend Gap | No endpoints exist for modifying individual turns. |
| **Direct Knowledge Retrieval** | ✅ Implemented | `POST /projects/{id}/retrieval` | ✅ Usable Now | Can be used for raw chunk inspection or debug views. |
| **User Authentication / Login** | ❌ Not Implemented | None | ❌ Backend Gap | All endpoints are open/unauthenticated. Protected only by UUID. |
| **CORS Middleware** | ❌ Not Implemented | `app/main.py` | ⚠️ Blocker | Needs `CORSMiddleware` in FastAPI to allow browser requests. |
| **Agent / BRD Workflow ("Work")**| ❌ Not Implemented | `agents/brd/` (empty) | ❌ Future Agent Phase| UI switcher `Chat \| + Work` should show "Coming Soon". |

---

## C. Information Architecture

```
Application Root (/)
│
├── Projects Hub (/projects)
│   ├── Project Search & List
│   ├── Create Project Dialog
│   └── Filter Tabs (All / Created by you / Shared with you)
│
└── Project Workspace (/projects/:projectId)
    │
    ├── Project Header (Name, Share Action, Overflow Menu)
    │
    ├── Project Home (Default View)
    │   ├── Prominent New Chat Composer
    │   ├── Sub-Navigation Pills (Chats | Sources)
    │   │
    │   ├── Chats Tab (Active by default)
    │   │   ├── New Chat Shortcut
    │   │   └── Conversation History List (Title, Date, Message Count)
    │   │
    │   └── Sources Tab (Knowledge Base)
    │       ├── Add Sources Action (File Upload Modal / Drag-and-Drop)
    │       ├── Sort & Filter Controls (Newest / All)
    │       └── Source Document Cards (Name, Status Badge, Date, Delete Action)
    │
    └── Active Conversation View (/projects/:projectId/c/:conversationId)
        ├── Breadcrumb to Project Home
        ├── Conversation Header & Title Renaming
        ├── Chronological Message Stream (User & Assistant Turns)
        ├── Assistant Telemetry Inspector (Retrieved Chunks, Tokens, Latency)
        └── Bottom Chat Composer (Follow-up Turn Input)
```

---

## D. Routes & Navigation Specification

| Route Path | View / Component | Parent Context | Required Route Params | Query Params | Auth Required | Navigation From / To |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `/` | `RootRedirect` | None | None | None | None | Redirects to `/projects` if projects exist, or triggers First-Time Create Flow. |
| `/projects` | `ProjectsHubView` | App Layout | None | None | None | Entry point. Navigates to `/projects/:projectId` on row click. |
| `/projects/:projectId` | `ProjectHomeView` | Project Layout | `projectId` (UUID) | `tab`: `"chats"` \| `"sources"` (default: `"chats"`) | None | Loaded when selecting a project. Submitting composer navigates to conversation. |
| `/projects/:projectId/c/:conversationId` | `ConversationView` | Project Layout | `projectId`, `conversationId` | None | None | Active chat. Breadcrumb navigates back to `/projects/:projectId`. |

---

## E. Screen-by-Screen UI Specification

### 1. Initial / First-Time Experience
* **Purpose**: Provide a clean, frictionless onboarding state when no projects exist in the database.
* **Entry Point**: Direct access to `/` or `/projects` when `GET /projects` returns an empty array `[]`.
* **Layout Structure**: Minimalist, centered modal/card on deep black `#0A0A0A` canvas.
* **UI Elements**:
  * Brand Logo & Heading: *"Welcome to AI Agent Workspace"*.
  * Subheading: *"Create your first project workspace to begin chatting and building your knowledge base."*
  * Input: Project Name (autofocused, placeholder: `e.g. AI Agent, Market Research`).
  * Action Button: Primary white pill `Create Project`.
* **User Actions**: Submitting valid name triggers `POST /projects`, redirects to `/projects/{new_project_id}`.
* **Empty / Error States**: In-line validation error if field is empty or whitespace.

---

### 2. Projects Screen (Projects Hub)
*Reference: Image 2 (`input_file_1.png`)*
* **Purpose**: Central workspace selector to view, search, open, and create projects.
* **Route**: `/projects`
* **Layout Structure**:
  * **Header Row**:
    * Page Title: `Projects` (32px bold white).
    * Search Input: Rounded pill input with magnifying glass (`Search projects`).
    * Primary Action: White pill button `New` (+ Project).
  * **Filter Pills Row**:
    * Tabs: `All` (selected by default, dark pill with white text), `Created by you`, `Shared with you`.
    * *Implementation note*: Because user auth and project sharing do not exist in the backend, selecting `Created by you` or `Shared with you` displays an explanatory empty note or matches `All`.
  * **Table / List Header**:
    * Column Left: `Name`
    * Column Right: `Modified`
  * **Project Rows**:
    * Left icon: Rounded square containing folder icon `📁`.
    * Title: Project Name (e.g. `AI Agent`, `Personal Branding`, `synapse`).
    * Modified Date: Humanized timestamp (e.g. `Monday`, `Aug 23`, `Mar 31`).
    * Right action: Pin icon (stored in browser `localStorage` to pin favorites to the top).
* **API Dependencies**: `GET /projects?limit=100&offset=0`
* **Loading State**: Shimmer skeleton rows.
* **Empty State**: Friendly illustration: *"No projects found. Create a new project to get started."*
* **Error State**: Banner: *"Failed to load projects. [Retry]"*.

---

### 3. Project Creation Modal / Dialog
* **Purpose**: Create a new project container from anywhere in the app.
* **Trigger**: Clicking `New` on the Projects screen or `+` in future sidebar.
* **Form Fields**:
  * Project Name (`name`): Text input, required, max 255 chars.
  * Description (`description`): Optional multiline text area.
* **API Action**: `POST /projects` with body `{"name": "...", "description": "..."}`.
* **Success State**: Closes modal, immediately navigates to `/projects/{response.id}`.

---

### 4. Project Home (The Central Workspace)
*References: Image 1 (`input_file_0.png`) and Image 3 (`input_file_2.png`)*
* **Purpose**: Primary workspace dashboard scoped to the active project.
* **Route**: `/projects/:projectId`
* **Top Header**:
  * Center Mode Switcher: Pill with `Chat` (selected) and `+ Work` (unselected). Clicking `+ Work` reveals a tooltip: *"Agent workflows coming soon"*.
  * Left: Folder icon `📁` followed by Project Title (e.g. `📁 AI Agent`). Double clicking or clicking edit icon triggers `PATCH /projects/:projectId` to rename.
  * Right: `Share` button (placeholder modal) and `...` overflow menu (`Project Settings`, `Delete Project`).
* **Prominent New Chat Composer**:
  * Centered, elevated input container.
  * Placeholder: `+ New chat in {projectName}`
  * Right Toolbar within input:
    * `Think` pill toggle (visual indicator for deep reasoning mode).
    * Microphone icon (speech-to-text input via browser Web Speech API).
    * Audio/Waveform icon.
  * Action: Pressing Enter or clicking Send initiates the **New Chat → Conversation Transition Flow** (Section E.7).
* **Section Navigation Pills**:
  * Two pills: `[ Chats ]` and `[ Sources ]`.
  * URL synchronized via query parameter: `?tab=chats` vs `?tab=sources`.

---

### 5. Chats Tab View
*Reference: Image 3 (`input_file_2.png`)*
* **Purpose**: Display the conversation history scoped strictly to the current project.
* **Content Structure**:
  * Chronological list of conversations (`GET /projects/:projectId/conversations`).
  * Each conversation item displays:
    * Title (bold white, e.g. `Study Agent Harness`, `Configure Supabase Storage Privacy`).
    * Preview Snippet: Since the backend `ConversationResponse` does not provide `last_message_content` in the list endpoint, the frontend displays the title or fetches recent turns upon hover/hydration.
    * Date: Formatted date aligned to the right (e.g. `Sep 16`, `Sep 15`).
    * Hover Actions: Delete conversation icon (calls `DELETE .../conversations/{id}` with confirmation).
* **Click Action**: Clicking a row opens `/projects/:projectId/c/:conversationId`.
* **Empty State**: *"No conversations yet. Type a question above to start chatting."*

---

### 6. Sources Tab View (Knowledge Base)
*Reference: Image 1 (`input_file_0.png`)*
* **Purpose**: Manage the documents ingested into the project's RAG knowledge base.
* **Header Controls**:
  * Left: `+ Add sources` button (opens upload modal / file drawer).
  * Right: Filter and Sort dropdowns (`Newest ▾`, `All ▾`).
* **Source Cards List**:
  * File icon (blue document icon).
  * Document Name (e.g. `AI-Agent-Updated-Context-Final(1).md`, `RAG_ARCHITECTURE_CONTEXT.md`).
  * Subtitle: `File · {formatted_date}`.
  * Indexing Status Badge:
    * 🟡 `pending` / `indexing`: Animated spinner or pulsing badge (*"Processing..."*).
    * 🟢 `ready`: Subtle checkmark (*"Ready for RAG"*).
    * 🔴 `failed`: Warning badge with tooltip showing `latest_version.error_message`.
  * Actions Menu (`...`):
    * `Rename`: Opens inline title edit (`PATCH .../sources/{id}`).
    * `Upload New Version`: Opens version upload modal (`POST .../sources/{id}/versions`).
    * `Delete`: Deletes document (`DELETE .../sources/{id}`).
* **Empty State**: *"No knowledge base documents added. Add documents to ground your AI assistant in your project's data."*

---

### 7. New Chat → Conversation Interface Transition Flow
* **Critical Interaction**:
  1. User is on Project Home (`/projects/:projectId`).
  2. User focuses the composer: `+ New chat in {projectName}`.
  3. User types: *"What are the termination notice requirements?"* and presses Enter.
  4. **Frontend Action Sequence**:
     * Step A: Frontend generates a temporary conversation title from the first 40 characters of the prompt.
     * Step B: Frontend calls `POST /projects/:projectId/conversations` with payload `{"title": "..."}`.
     * Step C: Immediately on 201 Created response, frontend navigates router to `/projects/:projectId/c/:new_conversation_id`.
     * Step D: The `ConversationView` mounts. It renders the user turn optimistically into the message list.
     * Step E: It displays an animated *"AI is thinking / retrieving knowledge..."* assistant skeleton.
     * Step F: Frontend calls `POST /projects/:projectId/conversations/:new_conversation_id/messages?generate=true` with payload `{"content": "...", "role": "user"}`.
     * Step G: When the backend returns the 201 response containing the assistant turn, the assistant skeleton resolves into the markdown message with full telemetry metadata.

---

### 8. Conversation Interface (Active Chat Screen)
* **Purpose**: Multi-turn dialogue between user and project-scoped assistant.
* **Route**: `/projects/:projectId/c/:conversationId`
* **Header**:
  * Left: Breadcrumb link `← {projectName}` returning to `/projects/:projectId`.
  * Title: Editable conversation title (`PATCH .../conversations/:conversationId`).
  * Right: Share conversation / Delete thread actions.
* **Message Stream**:
  * Auto-scrolling, reversed container.
  * **User Turns**:
    * Right-aligned or distinctive subtle card.
    * Plain text content with timestamp.
  * **Assistant Turns**:
    * Left-aligned with AI avatar/glyph.
    * Full Markdown rendering (tables, headers, bold, bullet points).
    * Syntax-highlighted code blocks with copy-to-clipboard button.
    * **RAG Telemetry Accordion** (Collapsible footer on turn):
      * Shows: `Model: gpt-4o`, `Tokens: 1,420`, `Latency: 1.2s`, `Retrieved Chunks: 3`.
      * Expands to reveal the actual retrieval query and groundedness status.
* **Bottom Input Bar**:
  * Sticky input composer with multiline auto-expand.
  * Buttons for Send (Enter), `Shift+Enter` for newlines.
  * Disabled while assistant generation is in flight to prevent concurrent race conditions.

---

### 9. Source Upload Modal Flow
* **Purpose**: Upload `.txt`, `.md`, or `.docx` files to object storage and trigger indexing.
* **Trigger**: Clicking `+ Add sources` in the Sources tab.
* **Interface**:
  * Drag-and-drop dropzone supporting multiple file selection.
  * File restrictions note: *"Supported formats: Markdown (.md), Plain Text (.txt), Word (.docx). Max file size: 50MB."*
  * Warning note on PDF: *"PDF processing is coming in the next update. Please upload markdown or text files for now."*
  * Display Name input (optional, defaults to filename).
* **Execution**:
  * Dispatches `POST /projects/:projectId/sources` with `multipart/form-data`.
  * Displays upload progress bar (0% -> 100%).
  * On completion, polls or refetches source list. Since `AUTO_PROCESS_DOCUMENTS=True`, status will transition from `indexing` to `ready`.

---

## F. API Integration Specification

All endpoints are hosted at base URL `http://localhost:8000` (or `VITE_API_BASE_URL`).

### 1. Projects API

#### 1.1 Create Project
* **Endpoint**: `POST /projects`
* **Summary**: Create a new root tenant workspace.
* **Headers**: `Content-Type: application/json`
* **Request Body**:
```json
{
  "name": "AI Agent",
  "description": "Workspace for multi-agent RAG exploration"
}
```
* **Field Constraints**: `name` is required, length 1–255 characters, non-empty after trim. `description` is optional string or null.
* **Response (201 Created)**:
```json
{
  "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "AI Agent",
  "description": "Workspace for multi-agent RAG exploration",
  "created_at": "2026-09-16T10:15:30.123456Z",
  "updated_at": "2026-09-16T10:15:30.123456Z"
}
```
* **Error Handling**: `400 Bad Request` if name is empty or whitespace.
* **Frontend Component**: `CreateProjectModal`.

#### 1.2 List Projects
* **Endpoint**: `GET /projects`
* **Query Parameters**:
  * `limit` (optional int, 1–100, default: 100)
  * `offset` (optional int, >=0, default: 0)
* **Response (200 OK)**:
```json
[
  {
    "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "name": "AI Agent",
    "description": "Workspace for multi-agent RAG exploration",
    "created_at": "2026-09-16T10:15:30.123456Z",
    "updated_at": "2026-09-16T10:15:30.123456Z"
  }
]
```
* **Sorting**: Sorted by backend in descending order of creation.
* **Frontend Component**: `ProjectsHubView`.

#### 1.3 Get Project by ID
* **Endpoint**: `GET /projects/{project_id}`
* **Response (200 OK)**: `ProjectResponse` object.
* **Error**: `404 Not Found` (`{"detail": "Project '...' not found."}`).
* **Frontend Component**: `ProjectLayout` (validates project on mount).

#### 1.4 Update Project
* **Endpoint**: `PATCH /projects/{project_id}`
* **Request Body**: `{"name": "New Name", "description": "Optional updated description"}`
* **Response (200 OK)**: Updated `ProjectResponse`.
* **Frontend Component**: Inline title editor in Project Home header.

#### 1.5 Delete Project
* **Endpoint**: `DELETE /projects/{project_id}`
* **Response**: `204 No Content`
* **Behavior**: Cascades in PostgreSQL to all documents, versions, chunks, conversations, and messages, and cleans up files in Supabase Storage.
* **Frontend Action**: Prompt modal *"Are you sure? This will delete all chats and sources permanently."* On success, navigate to `/projects`.

---

### 2. Sources (Documents & Knowledge Base) API

#### 2.1 Upload Source Document
* **Endpoint**: `POST /projects/{project_id}/sources`
* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `file`: Binary file data (Required).
  * `name`: Optional string override for display name.
* **Response (201 Created)**:
```json
{
  "id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "AI-Agent-Updated-Context-Final(1).md",
  "created_at": "2026-09-16T10:20:00.000000Z",
  "updated_at": "2026-09-16T10:20:00.000000Z",
  "latest_version": {
    "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
    "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "version_number": 1,
    "storage_bucket": "documents",
    "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/AI-Agent-Updated-Context-Final(1).md",
    "original_filename": "AI-Agent-Updated-Context-Final(1).md",
    "content_type": "text/markdown",
    "size_bytes": 14250,
    "etag": "\"68b329da9893e34099c7d8ad5cb9c940\"",
    "status": "ready",
    "error_message": null,
    "indexed_at": "2026-09-16T10:20:05.120000Z",
    "created_at": "2026-09-16T10:20:00.000000Z",
    "updated_at": "2026-09-16T10:20:05.120000Z"
  },
  "versions_count": 1
}
```
* **Frontend Component**: `AddSourcesModal`.

#### 2.2 List Source Documents
* **Endpoint**: `GET /projects/{project_id}/sources`
* **Query Parameters**: `limit` (1–100, default 100), `offset` (>=0, default 0).
* **Response (200 OK)**: `list[DocumentResponse]`.
* **Frontend Component**: `SourcesTabView`.

#### 2.3 Get Source Document Detail
* **Endpoint**: `GET /projects/{project_id}/sources/{document_id}`
* **Response (200 OK)**: `DocumentDetailResponse` (includes `versions: list[DocumentVersionResponse]`).
* **Frontend Component**: Document Inspector drawer.

#### 2.4 Update Source Metadata
* **Endpoint**: `PATCH /projects/{project_id}/sources/{document_id}`
* **Request Body**: `{"name": "New Document Name"}`
* **Response (200 OK)**: Updated `DocumentResponse`.

#### 2.5 Delete Source Document
* **Endpoint**: `DELETE /projects/{project_id}/sources/{document_id}`
* **Response**: `204 No Content`.
* **Behavior**: Deletes PostgreSQL chunks, versions, document record, and physical files from Supabase.

#### 2.6 Upload New Source Version
* **Endpoint**: `POST /projects/{project_id}/sources/{document_id}/versions`
* **Content-Type**: `multipart/form-data` (`file: UploadFile`)
* **Response (201 Created)**: `DocumentVersionResponse`.

---

### 3. Conversations & Chat Generation API

#### 3.1 Create Conversation
* **Endpoint**: `POST /projects/{project_id}/conversations`
* **Request Body**:
```json
{
  "title": "Study Agent Harness"
}
```
* **Response (201 Created)**:
```json
{
  "id": "c1d2e3f4-5678-4901-abcd-ef0123456789",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "title": "Study Agent Harness",
  "created_at": "2026-09-16T10:30:00.000000Z",
  "updated_at": "2026-09-16T10:30:00.000000Z",
  "messages_count": 0
}
```

#### 3.2 List Conversations
* **Endpoint**: `GET /projects/{project_id}/conversations`
* **Query Parameters**: `limit` (1–100, default 100), `offset` (>=0, default 0).
* **Response (200 OK)**: `list[ConversationResponse]`.
* **Sorting**: Sorted by `updated_at DESC`.
* **Frontend Component**: `ChatsTabView` in Project Home.

#### 3.3 Get Conversation Details & History
* **Endpoint**: `GET /projects/{project_id}/conversations/{conversation_id}`
* **Response (200 OK)**:
```json
{
  "id": "c1d2e3f4-5678-4901-abcd-ef0123456789",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "title": "Study Agent Harness",
  "created_at": "2026-09-16T10:30:00.000000Z",
  "updated_at": "2026-09-16T10:30:15.000000Z",
  "messages_count": 2,
  "messages": [
    {
      "id": "m1111111-2222-3333-4444-555555555555",
      "conversation_id": "c1d2e3f4-5678-4901-abcd-ef0123456789",
      "role": "user",
      "content": "What is the agent harness architecture?",
      "metadata": {},
      "created_at": "2026-09-16T10:30:05.000000Z"
    },
    {
      "id": "m2222222-3333-4444-5555-666666666666",
      "conversation_id": "c1d2e3f4-5678-4901-abcd-ef0123456789",
      "role": "assistant",
      "content": "The agent harness is the runtime execution loop that coordinates tool calling...",
      "metadata": {
        "model": "gpt-4o",
        "finish_reason": "stop",
        "usage": {
          "prompt_tokens": 850,
          "completion_tokens": 120,
          "total_tokens": 970
        },
        "retrieval": {
          "chunks_count": 2,
          "duration_ms": 42.5,
          "attempts_count": 1,
          "retrieval_query": "agent harness architecture",
          "fallback_triggered": false
        },
        "evaluation_passed": true,
        "total_duration_ms": 1150.2
      },
      "created_at": "2026-09-16T10:30:07.000000Z"
    }
  ]
}
```
* **Frontend Component**: `ConversationView` (initial mount).

#### 3.4 Send Message & Generate AI Response
* **Endpoint**: `POST /projects/{project_id}/conversations/{conversation_id}/messages?generate=true`
* **Query Parameters**: `generate=true` (CRITICAL: activates RAG retrieval + LLM inference pipeline).
* **Request Body**:
```json
{
  "content": "Explain how the RAG pipeline works in this project.",
  "role": "user",
  "metadata": {}
}
```
* **Under-the-Hood Backend Lifecycle**:
  1. Validates project boundary (`project_id == conversation.project_id`).
  2. Persists the `user` message turn to the database immediately and commits.
  3. Executes hybrid RAG retrieval against Qdrant and PostgreSQL Content Store scoped strictly to `project_id`.
  4. Formats context and constructs prompt with system instructions.
  5. Calls the OpenAI-compatible LLM gateway.
  6. Passes response through groundedness & safety evaluation quality gate.
  7. Persists the `assistant` message turn to the database with generation telemetry in `metadata`.
  8. Returns HTTP `201 Created` containing the **assistant MessageResponse**!
* **Response (201 Created)**:
```json
{
  "id": "m9999999-8888-7777-6666-555555555555",
  "conversation_id": "c1d2e3f4-5678-4901-abcd-ef0123456789",
  "role": "assistant",
  "content": "The RAG pipeline operates in modular stages: document acquisition, parsing...",
  "metadata": {
    "user_message_id": "m1111111-2222-3333-4444-555555555555",
    "finish_reason": "stop",
    "usage": {
      "prompt_tokens": 1250,
      "completion_tokens": 340,
      "total_tokens": 1590
    },
    "model": "gpt-4o",
    "retrieval": {
      "chunks_count": 3,
      "duration_ms": 38.2,
      "attempts_count": 1,
      "retrieval_query": "RAG pipeline architecture",
      "fallback_triggered": false
    },
    "evaluation": {
      "grounded": true,
      "safe": true,
      "passed": true,
      "reason": "Directly supported by retrieved chunks"
    },
    "evaluation_passed": true,
    "generation_attempts": 1,
    "total_duration_ms": 1340.5
  },
  "created_at": "2026-09-16T10:31:02.000000Z"
}
```
* **Frontend Handling**:
  * The frontend displays the user turn optimistically in the chat stream before sending the request.
  * When the 201 response returns, the frontend appends the returned assistant turn.
  * If the request fails with `422 Unprocessable Entity` (`RegenerationExhaustedError`), the backend refused to answer because retrieved documents lacked sufficient evidence. The frontend should display a polite notice: *"I could not find sufficient grounded information in your uploaded project sources to answer this question."*

#### 3.5 Update Conversation Title
* **Endpoint**: `PATCH /projects/{project_id}/conversations/{conversation_id}`
* **Request Body**: `{"title": "Updated Topic Title"}`
* **Response (200 OK)**: `ConversationResponse`.

#### 3.6 Delete Conversation
* **Endpoint**: `DELETE /projects/{project_id}/conversations/{conversation_id}`
* **Response**: `204 No Content`.

---

### 4. Direct Knowledge Retrieval API (Optional / Diagnostics)
* **Endpoint**: `POST /projects/{project_id}/retrieval`
* **Request Body**:
```json
{
  "query": "termination notice requirements",
  "top_k": 3,
  "enable_transformation": true,
  "enable_sparse": true,
  "enable_reranking": true
}
```
* **Response (200 OK)**: Returns `RetrievalResponseSchema` including `chunks: list[RetrievedChunkSchema]`, `formatted_context`, and `execution_metadata`.
* **Frontend Usage**: Useful if you choose to build an advanced "Inspect Knowledge Base Search" or diagnostic modal.

---

## G. Data Models & Schemas

### 1. Project Entity
* **Database Model**: `ProjectModel` ([backend/src/models/project.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/models/project.py))
* **Pydantic Schema**: `ProjectResponse` ([backend/src/schemas/project.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/schemas/project.py))
* **Fields**:
  * `id`: `string` (UUIDv4 primary key)
  * `name`: `string` (Required, 1-255 characters)
  * `description`: `string | null` (Optional text description)
  * `created_at`: `string` (ISO 8601 UTC timestamp)
  * `updated_at`: `string` (ISO 8601 UTC timestamp)

### 2. Document (Source) Entity
* **Database Model**: `DocumentModel` ([backend/src/models/document.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/models/document.py))
* **Pydantic Schema**: `DocumentResponse` / `DocumentDetailResponse` ([backend/src/schemas/document.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/schemas/document.py))
* **Fields**:
  * `id`: `string` (UUIDv4 primary key)
  * `project_id`: `string` (Foreign key referencing `projects.id`)
  * `name`: `string` (Display name, defaults to uploaded filename)
  * `created_at`: `string` (ISO 8601 timestamp)
  * `updated_at`: `string` (ISO 8601 timestamp)
  * `latest_version`: `DocumentVersionResponse | null` (Summary of active physical file)
  * `versions_count`: `number` (Total revisions)

### 3. Document Version Entity
* **Database Model**: `DocumentVersionModel` ([backend/src/models/document.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/models/document.py))
* **Fields**:
  * `id`: `string` (UUIDv4)
  * `document_id`: `string` (Foreign key to `documents.id`)
  * `project_id`: `string` (Foreign key to `projects.id`)
  * `version_number`: `number` (Sequential integer, starts at 1)
  * `original_filename`: `string` (e.g. `contract.md`)
  * `content_type`: `string | null` (MIME type)
  * `size_bytes`: `number | null` (Integer bytes)
  * `status`: `string` (`"pending"` | `"indexing"` | `"ready"` | `"failed"`)
  * `error_message`: `string | null` (Populated if `status === "failed"`)
  * `indexed_at`: `string | null` (Timestamp when RAG indexing finished)

### 4. Conversation Entity
* **Database Model**: `ConversationModel` ([backend/src/models/conversation.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/models/conversation.py))
* **Pydantic Schema**: `ConversationResponse` / `ConversationDetailResponse` ([backend/src/schemas/conversation.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/schemas/conversation.py))
* **Fields**:
  * `id`: `string` (UUIDv4 primary key)
  * `project_id`: `string` (Foreign key referencing `projects.id`)
  * `title`: `string` (Default: `"New Conversation"`)
  * `created_at`: `string` (ISO 8601 timestamp)
  * `updated_at`: `string` (ISO 8601 timestamp)
  * `messages_count`: `number` (Count of message turns)
  * `messages`: `MessageResponse[]` (Only present in `ConversationDetailResponse`)

### 5. Message Entity
* **Database Model**: `MessageModel` ([backend/src/models/message.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/models/message.py))
* **Pydantic Schema**: `MessageResponse` ([backend/src/schemas/conversation.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/schemas/conversation.py))
* **Fields**:
  * `id`: `string` (UUIDv4 primary key)
  * `conversation_id`: `string` (Foreign key referencing `conversations.id`)
  * `role`: `string` (Strictly `"user"` or `"assistant"`)
  * `content`: `string` (Text body of turn)
  * `metadata`: `Record<string, any>` (Structured JSON containing token usage, model, retrieval counts, evaluation)
  * `created_at`: `string` (ISO 8601 timestamp)

---

## H. Entity Relationships & Scoping Rules

```
                       ┌────────────────────────┐
                       │      User (Future)     │
                       └───────────┬────────────┘
                                   │
                                   ▼ 1:N
                       ┌────────────────────────┐
                       │        Project         │
                       └─────┬────────────┬─────┘
                             │            │
                   1:N (Owns)│            │1:N (Owns)
                             ▼            ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐
        │        Document         │  │      Conversation       │
        │        (Source)         │  │         (Thread)        │
        └────────────┬────────────┘  └────────────┬────────────┘
                     │                            │
                 1:N │                        1:N │
                     ▼                            ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐
        │     DocumentVersion     │  │         Message         │
        └────────────┬────────────┘  └─────────────────────────┘
                     │
                 1:N │
                     ▼
        ┌─────────────────────────┐
        │       ChunkModel        │
        │ (Vectors + PostgreStore)│
        └─────────────────────────┘
```

### Critical Scoping Invariants:
1. **Strict Tenant Isolation**: `project_id` is the root container. Attempting to access a conversation or source using the wrong `project_id` in the URL results in an immediate `404 Not Found` (`ProjectConversationMismatchError` or `ProjectDocumentMismatchError`).
2. **Preserve IDs in Route Params**: The frontend must keep `projectId` in active router context (e.g. React Router `:projectId` parameter or Zustand workspace store). Every conversation and source API call requires `projectId` as the first path parameter.
3. **No Cross-Project Data Leakage**: Sources in Project A are NEVER retrieved when chatting in Project B. The RAG retrieval pipeline applies hard SQL filters (`WHERE project_id = :project_id`) and Qdrant payload filters on every query.

---

## I. Form & Validation Requirements

| Form | Field | Type | Required | Constraints / Validation | Frontend Validation | Backend Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Create Project** | `name` | text | Yes | 1–255 chars, cannot be pure whitespace. | Trim string; show error if `< 1` char. | Raises `ValueError` / `InvalidProjectDataError` (HTTP 400). |
| **Create Project** | `description` | textarea | No | Max 1000 chars suggested. | Optional trim. | Stored as nullable Text. |
| **Update Project** | `name` | text | No | 1–255 chars if provided. | Prevent submitting empty string. | Validates non-empty if provided. |
| **Upload Source** | `file` | file binary | Yes | Only `.md`, `.txt`, `.docx`. Max 50MB. | Check file extension and `file.size <= 50MB`. | Checks for non-empty bytes; sanitizes filename. |
| **Upload Source** | `name` | text | No | Optional display name (max 255 chars). | Default to uploaded filename if left blank. | Uses clean filename if `name` is null. |
| **New Chat Composer**| `content` | textarea | Yes | Non-empty text string. | Disable submit button while input is whitespace. | Raises `ValueError` / `InvalidMessageDataError` (HTTP 400). |
| **Rename Conversation**| `title`| text | Yes | 1–255 chars. | Prevent whitespace-only submission. | Falls back to "New Conversation" if blank. |

---

## J. UI State Model

For every data-driven view, implement this standard UI state machine:

```
[ INITIAL / IDLE ]
        │
        ▼ (Mount / Fetch)
  [ LOADING ] ──── (Network Fail) ────► [ NETWORK ERROR ]
        │                                      │ (Retry)
        ├──────────────────────────────────────┘
        ▼ (200 OK)
   [ IS_EMPTY? ] ─── YES ───► [ EMPTY STATE (Actionable illustration) ]
        │ NO
        ▼
   [ LOADED / READY ]
        │
        ▼ (User Submits Form)
  [ SUBMITTING ]
        │
        ├── (Success 200/201/204) ──► [ SUCCESS TOAST / REDIRECT ]
        │
        └── (Failure 4xx/5xx) ─────► [ INLINE ERROR / RETRY BUTTON ]
```

### Specific State Responses:
* **`unauthorized (401)` / `forbidden (403)`**: Display toast *"Session expired or access unauthorized"*. (Ready for future auth integration).
* **`not_found (404)`**: When opening an invalid `projectId` or `conversationId`, render a dedicated `404 Workspace Not Found` screen with a button: *"Return to Projects"*.
* **`unprocessable_entity (422)`**: Emitted when LLM generation fails the groundedness quality gate. Show an assistant warning callout: *"The assistant could not formulate an answer grounded in your project documents."*
* **`too_many_requests (429)`**: Rate limit reached. Show toast: *"AI service is currently busy. Please retry in a few seconds."*
* **`gateway_timeout (504)`**: AI generation timed out. Show retry button on the turn: *"Request timed out. [Retry generation]"*.

---

## K. Error Handling Architecture

The backend standardizes error payloads on FastAPI's `{ "detail": "..." }` format. Exception handlers in [backend/src/app/main.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/app/main.py) map domain exceptions directly to standard HTTP status codes:

```typescript
// Proposed Frontend Error Normalizer
export interface ApiError {
  status: number;
  message: string;
  fieldErrors?: Record<string, string>;
}

export function parseApiError(error: any): ApiError {
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;
    
    // FastAPI validation errors
    if (status === 422 && Array.isArray(data.detail)) {
      const fieldErrors: Record<string, string> = {};
      data.detail.forEach((err: any) => {
        const field = err.loc[err.loc.length - 1];
        fieldErrors[field] = err.msg;
      });
      return { status, message: "Validation failed", fieldErrors };
    }
    
    return {
      status,
      message: typeof data.detail === "string" ? data.detail : "An error occurred",
    };
  }
  return { status: 0, message: error.message || "Network error. Please check your connection." };
}
```

---

## L. Authentication & Authorization

### Current Backend State
* **Status**: **Unauthenticated / Open Access**.
* The existing backend endpoints do not require an `Authorization` header, session cookie, or API key.
* Tenant isolation is enforced exclusively through the client-provided `project_id` UUID path parameter.

### Frontend Engineering Action:
1. **Prepare Auth Interceptor**: Build the frontend API client (e.g. Axios or Fetch wrapper) with a request interceptor that checks for an `accessToken` in memory/cookies:
   ```typescript
   client.interceptors.request.use((config) => {
     const token = getAuthToken(); // returns null for now
     if (token) {
       config.headers.Authorization = `Bearer ${token}`;
     }
     return config;
   });
   ```
2. **Handle 401 Gracefully**: If the backend introduces auth in the future, the interceptor will automatically catch `401 Unauthorized` and redirect to a login screen.
3. **Mock User State**: Maintain a mock user state (e.g. `{ name: "Current User", email: "user@example.com" }`) to populate the user avatar in the navigation bar until user management endpoints are built.

---

## M. Source Management & RAG Integration

### Document Ingestion Lifecycle
When a user uploads a source via `POST /projects/{project_id}/sources`:
1. The binary file is stored in Supabase Object Storage under `{project_id}/{document_id}/v1/{filename}`.
2. A `DocumentModel` and `DocumentVersionModel` are created in PostgreSQL with `status = "pending"`.
3. Because `AUTO_PROCESS_DOCUMENTS = True` in [backend/src/core/config.py](file:///c:/DEXMIQ_PROJECTS/agent-mvp/backend/src/core/config.py), the backend immediately invokes `DocumentProcessingService.trigger_processing()`.
4. The processing pipeline runs:
   * **Acquisition & Parsing**: Reads file from storage and parses via `MarkdownParser`, `TXTParser`, or `DOCXParser`.
   * **Cleaning & Normalization**: Strips excessive whitespace and artifacts.
   * **Semantic Chunking**: Splits text into chunks respecting headings and structural boundaries (max size: 1,000 characters).
   * **Embedding Generation**: Computes dense vector embeddings using **Voyage AI** (`voyage-4`, 1024 dimensions) and technical hash sparse vectors.
   * **PostgreSQL Content Store**: Saves verbatim chunks and structural coordinates in the `chunks` table.
   * **Vector Store Indexing**: Upserts vectors and metadata into Qdrant collection `document_chunks`.
   * **Status Transition**: Updates `DocumentVersionModel.status = "ready"` and sets `indexed_at`. If an error occurs, it sets `status = "failed"` and writes `error_message`.

### Frontend Polling / Status Refresh
* After upload, if `latest_version.status` is `"pending"` or `"indexing"`, the frontend should poll `GET /projects/{project_id}/sources` every 2 seconds until status changes to `"ready"` or `"failed"`.
* Once status is `"ready"`, the document is actively searchable by the RAG generation pipeline.

---

## N. Chat & Agent Integration

### Current Synchronous Execution Model
* Chat is currently a **synchronous REST request**:
  `POST /projects/{project_id}/conversations/{conversation_id}/messages?generate=true`
* The request takes between 1.0 to 3.5 seconds depending on document retrieval and LLM latency.
* **Frontend Requirement**: The UI must display an animated "thinking" state on the assistant bubble while awaiting the HTTP response.

### Future Agent Integration Points (Marked: PLANNED / NOT CURRENTLY IMPLEMENTED)
The incoming frontend architecture should anticipate these future capabilities without attempting to call endpoints that do not exist today:

1. **Token Streaming via Server-Sent Events (SSE)**:
   * *Status*: Planned.
   * *Future Pattern*: `GET /projects/:projectId/conversations/:conversationId/stream?message=...` using `EventSource` or `fetch-event-source` for incremental typing animation.
2. **Agent Tool Execution Telemetry**:
   * *Status*: Planned (backend `src/agents/brd/` directory is currently empty).
   * *Future Pattern*: Events emitting intermediate tool executions (`searching_web`, `reading_file`, `calculating`) rendered as expandable step items in the assistant message.
3. **Human-in-the-Loop (HITL) Interruptions**:
   * *Status*: Planned.
   * *Future Pattern*: Agent pauses execution and asks the user for confirmation (e.g., approval to execute an external API mutation), requiring an interactive action card in the chat view.

---

## O. Frontend vs. Backend Responsibility Boundary

To avoid architectural duplication, strictly adhere to these boundaries:

| Concern | Frontend Responsibility | Backend Responsibility |
| :--- | :--- | :--- |
| **Authentication Enforcement** | Stores token, attaches header, redirects on 401. | Authoritative token verification and user scoping. |
| **Tenant Isolation** | Passes `projectId` in all API paths. | Authoritative SQL & vector query isolation. Rejects mismatches. |
| **Document Processing** | File selection, extension validation, size check, progress bar. | Parsing, chunking, embedding, vector upsert, and storage. |
| **RAG Retrieval** | Displays resulting chunk telemetry returned in metadata. | Hybrid search, query rewrite, fusion, reranking, and scoring. |
| **AI Generation** | Submits prompt; renders markdown, syntax highlighting, and code. | Prompt assembly, LLM inference, and groundedness evaluation. |
| **Conversation State** | Optimistic turns, active input state, local scroll position. | Persistent chronological sequence, message turn IDs, timestamps. |
| **Sort / Search (Current)** | Client-side filter on project names and source dates. | Authoritative database order (`updated_at DESC`). |

---

## P. Backend Gaps & Required Future Work

The following features shown in the UX screenshots or required for production do not yet have corresponding backend support. They must be handled via frontend fallbacks until backend tickets are implemented:

### 1. Missing CORS Middleware (URGENT BLOCKER)
* **Problem**: `backend/src/app/main.py` does not currently add `CORSMiddleware`.
* **Impact**: Browser security will block requests from `http://localhost:5173` or `3000`.
* **Required Backend Fix**:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  application.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:5173", "http://localhost:3000"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
* **Frontend Workaround**: During local dev before this fix is merged, run Vite with a reverse proxy in `vite.config.ts` (`server.proxy: {'/api': 'http://localhost:8000'}`).

### 2. Project Search Query Parameter
* **Problem**: `GET /projects` only accepts `limit` and `offset`. It has no `?search=` or `?query=` parameter.
* **Frontend Solution**: Filter the loaded projects array client-side by comparing `project.name.toLowerCase().includes(searchTerm.toLowerCase())`.

### 3. Project Pinning & Filter Tabs (`Created by you` / `Shared with you`)
* **Problem**: `ProjectModel` has no `is_pinned`, `owner_id`, or `collaborators` fields.
* **Frontend Solution**:
  * Store pinned project IDs in `localStorage` under key `dexmiq_pinned_projects`.
  * The tabs `Created by you` and `Shared with you` can display all projects or a tooltip indicating collaboration is coming soon.

### 4. Conversation Preview Snippet
* **Problem**: `ConversationResponse` returns `title` and `messages_count`, but lacks `last_message_content`. The screenshot shows a preview snippet under the title.
* **Frontend Solution**: Display the conversation `title` prominently and use the date on the right. If a preview is desired, the frontend can cache the first message sent or the backend can add `last_message_preview` in a future migration.

### 5. PDF Parser Registration
* **Problem**: `ParserRegistry` only registers parsers for `.txt`, `.md`, and `.docx`. Uploading `.pdf` will fail during ingestion.
* **Frontend Solution**: Restrict file upload picker to accept `.txt, .md, .docx`. Display an informational tag explaining PDF support is in progress.

### 6. Manual Re-index / Retry Endpoint
* **Problem**: If a document fails processing, there is no endpoint to retry indexing.
* **Frontend Solution**: Delete the failed source (`DELETE .../sources/{id}`) and prompt the user to re-upload.

---

## Q. Frontend Implementation Guidance

### Recommended Tech Stack
* **Framework**: React 18 / 19 with Vite (or Next.js App Router).
* **Language**: TypeScript (Strict mode enabled).
* **Styling**: Vanilla CSS or TailwindCSS with custom design tokens (Dark mode first).
* **Icons**: `lucide-react` (Folder, Sparkles, Send, Mic, AudioWaveform, Pin, Share2, MoreVertical, FileText, CheckCircle2, AlertCircle).
* **Markdown Rendering**: `react-markdown` with `remark-gfm` and `rehype-highlight` (or `prismjs`) for syntax-highlighted code blocks.
* **State Management**: Zustand or TanStack Query (React Query) for server-state caching and polling.

### Suggested Project Folder Structure (`c:\DEXMIQ_PROJECTS\agent-mvp\frontend`)
```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── api/
│   │   ├── client.ts              # Axios/Fetch client with base URL & error interceptor
│   │   ├── projects.ts            # Project API calls
│   │   ├── sources.ts             # Source & Version upload calls
│   │   └── conversations.ts       # Conversations, messages & generation calls
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx         # App brand & user profile
│   │   │   └── ProjectHeader.tsx  # Folder icon, project title, Share, ...
│   │   ├── projects/
│   │   │   ├── ProjectCard.tsx    # Row item with pin and relative date
│   │   │   └── CreateProjectModal.tsx
│   │   ├── sources/
│   │   │   ├── SourceCard.tsx     # File row with status badge
│   │   │   └── AddSourcesModal.tsx# Drag-and-drop file uploader
│   │   ├── chat/
│   │   │   ├── Composer.tsx       # Elevated new-chat input with Think/Mic icons
│   │   │   ├── MessageTurn.tsx    # Markdown bubble + copy code button
│   │   │   └── TelemetryView.tsx  # Collapsible RAG retrieval metrics
│   │   └── ui/                    # Reusable Button, Input, Modal, Badge, Dropdown
│   ├── pages/
│   │   ├── ProjectsPage.tsx       # Projects Hub (Image 2)
│   │   ├── ProjectHomePage.tsx    # Project Home with Chats/Sources tabs (Image 1 & 3)
│   │   └── ConversationPage.tsx   # Active chat view
│   ├── types/
│   │   ├── project.ts             # TypeScript interfaces matching backend schemas
│   │   ├── source.ts
│   │   └── conversation.ts
│   ├── App.tsx                    # Route definitions
│   └── main.tsx
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Visual Styling Palette (Dark Mode Aesthetic)
* **Canvas Background**: `#0A0A0A`
* **Card / Elevated Surface**: `#141414`
* **Border / Divider**: `#222222` (hover: `#333333`)
* **Primary Text**: `#FFFFFF`
* **Secondary / Muted Text**: `#8E8E93`
* **Accent / Button White**: `#F5F5F7` (Text: `#000000`)
* **Status Badges**:
  * Success / Ready: Green `#30D158` (Background: `rgba(48, 209, 88, 0.15)`)
  * Warning / Indexing: Orange `#FF9F0A` (Background: `rgba(255, 159, 10, 0.15)`)
  * Error / Failed: Red `#FF453A` (Background: `rgba(255, 69, 58, 0.15)`)

---

## R. Acceptance Criteria for Initial Implementation

The initial frontend delivery will be verified against the following concrete criteria:

1. **Workspace Creation**:
   * A user on an empty database is prompted to create a project.
   * Creating a project (`POST /projects`) navigates directly into the new Project Home.
2. **Projects Hub**:
   * Lists all existing projects sorted by last created (`GET /projects`).
   * Searching in the search bar filters project names immediately in the table.
   * Clicking a project opens `/projects/:projectId`.
3. **Project Home Navigation**:
   * Displays the project title with a folder icon and active tabs `[ Chats ]` and `[ Sources ]`.
   * Toggling between `Chats` and `Sources` updates the active view and synchronizes URL query params.
4. **Knowledge Base Management**:
   * In the `Sources` tab, clicking `+ Add sources` allows uploading `.txt`, `.md`, and `.docx` files (`POST .../sources`).
   * Uploaded sources show their filename, file size, upload date, and indexing status (`pending` -> `ready`).
   * Clicking delete removes the source (`DELETE .../sources/:id`) and updates the list.
5. **New Chat → Conversation Transition**:
   * Typing a prompt into the Project Home composer and submitting:
     * Immediately creates a conversation in the backend (`POST .../conversations`).
     * Navigates to `/projects/:projectId/c/:conversationId`.
     * Renders the user turn optimistically.
     * Displays a thinking state.
     * Dispatches `POST .../messages?generate=true`.
     * Renders the generated assistant response turn upon completion.
6. **Thread Continuity**:
   * Submitting subsequent messages inside the conversation view maintains conversational context and appends turns.
   * Clicking the back breadcrumb returns to Project Home, where the conversation now appears in the `Chats` list.
7. **Error Resilience**:
   * Form validation prevents blank project names and blank messages.
   * If RAG generation quality gate fails (HTTP 422), an informative explanation is shown rather than a generic crash.
   * 404 errors on invalid IDs render a friendly "Workspace Not Found" screen with a safe return link.
