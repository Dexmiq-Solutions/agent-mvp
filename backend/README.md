# Backend - AI Agent / RAG Platform

FastAPI backend service for the AI Agent and RAG platform.

## Object Storage (Supabase Storage)

### Responsibility & Scope

In this architecture, **Supabase Storage** is responsible exclusively for storing and retrieving **original source documents** (e.g., raw PDFs, Markdown, text files, and docx files).

```
Source Documents  ──►  Supabase Storage  ──►  Object Storage Module  ──►  Future Ingestion Pipeline
```

> [!NOTE]
> Document parsing, text extraction, chunking, metadata relations in PostgreSQL, vector embeddings via Voyage, and vector storage in Qdrant are separate architectural components handled by dedicated downstream modules. Supabase Storage does not perform parsing or vector processing.

### Communication & SDK

The backend communicates with Supabase Storage using the official asynchronous Python SDK (`supabase-py` / `storage3`). Direct vendor SDK calls are isolated inside the `storage.object` infrastructure layer behind the `BaseObjectStorage` interface, keeping the rest of the application decoupled from low-level storage APIs.

### Configuration & Environment Variables

Configure the following environment variables in `backend/.env`:

| Variable | Description | Example |
| :--- | :--- | :--- |
| `SUPABASE_URL` | Base URL of the Supabase project instance | `https://xyzproject.supabase.co` |
| `SUPABASE_KEY` | Supabase API key (anon / client key) | `eyJhbGciOi...` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role secret key (preferred for backend access) | `eyJhbGciOi...` |
| `SUPABASE_STORAGE_BUCKET` | Target bucket name for source documents | `documents` |

### Supabase Bucket Setup

1. **Automatic Initialization (Idempotent)**: The backend can automatically verify and create the target private bucket if it does not already exist via `ensure_bucket_exists()`.
2. **Manual Setup (Supabase Dashboard)**:
   - Go to **Storage** > **New Bucket**.
   - Set the bucket name (e.g., `documents`).
   - Keep the bucket **Private** (recommended) so objects are only accessible via authenticated server calls or temporary pre-signed URLs.

---

## Development

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

### Installation & Setup
```bash
cp .env.example .env
uv sync
```

### Running the Application
```bash
uv run uvicorn app.main:app --reload
```

### Running Tests
```bash
uv run pytest
```
