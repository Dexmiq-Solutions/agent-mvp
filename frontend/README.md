# Frontend Architecture & Technical Specification

Please refer to the authoritative specification document at [FRONTEND_SPECIFICATION.md](../FRONTEND_SPECIFICATION.md).

## Dexmiq AI Agent Frontend (Demo UI)

A lightweight, minimal frontend designed to manually demonstrate end-to-end integration between the FastAPI backend, RAG pipeline, Conversations, Sources, and the BRD Agent.

### Features
- **Project Hub**: List existing projects, create new projects, and select a project workspace.
- **Project Workspace**: Scoped by authoritative `project_id` to guarantee project boundary isolation.
- **Conversations**: Create new conversation threads, switch between threads, and view complete message histories.
- **Sources**: Upload project knowledge base documents (PDF, Word `.docx`, Markdown `.md`, Text `.txt`), view indexing status (`ready`, `indexing`, `pending`, `failed`), and auto-refresh after upload.
- **BRD Agent Chat**: Live streaming Server-Sent Events (SSE) chat interface connected to the backend BRD Agent.

### Quick Start

#### 1. Install Dependencies
```bash
npm install
```

#### 2. Start Dev Server
```bash
npm run dev
```
The Vite development server runs on `http://localhost:5173` and proxies API requests to `http://127.0.0.1:8000`.

#### 3. Run Tests
```bash
npm test
```

#### 4. Build for Production
```bash
npm run build
```
