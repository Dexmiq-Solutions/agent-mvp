# API Usage

This document serves as the complete, authoritative guide to all currently available public HTTP APIs in the application. Every endpoint documented here reflects the active backend routing, Pydantic schemas, database models, and service implementations.

---

## Table of Contents

- [Overview & Global Conventions](#overview--global-conventions)
- [System & Health](#system--health)
  - [Root Welcome Status](#root-welcome-status)
  - [Application Health Check](#application-health-check)
- [Authentication](#authentication)
- [Projects](#projects)
  - [Create Project](#create-project)
  - [List Projects](#list-projects)
  - [Get Project by ID](#get-project-by-id)
  - [Update Project](#update-project)
  - [Delete Project](#delete-project)
- [Sources / Documents](#sources--documents)
  - [Upload Source Document](#upload-source-document)
  - [List Source Documents](#list-source-documents)
  - [Get Source Document Details](#get-source-document-details)
  - [Update Source Document Metadata](#update-source-document-metadata)
  - [Delete Source Document](#delete-source-document)
  - [Upload New Document Version](#upload-new-document-version)
- [Conversations](#conversations)
  - [Create Conversation](#create-conversation)
  - [List Conversations](#list-conversations)
  - [Get Conversation Details & History](#get-conversation-details--history)
  - [Update Conversation Title](#update-conversation-title)
  - [Delete Conversation](#delete-conversation)
- [Messages & Generation](#messages--generation)
  - [Send Message / Generate AI Response](#send-message--generate-ai-response)
  - [List Chronological Messages](#list-chronological-messages)
- [Knowledge Retrieval](#knowledge-retrieval)
  - [Execute Project Knowledge Retrieval](#execute-project-knowledge-retrieval)
- [Missing, Broken, or Inconsistent APIs](#missing-broken-or-inconsistent-apis)

---

## Overview & Global Conventions

### Base URL
- **Local Development**: `http://localhost:8000`
- Configured via `HOST` (`0.0.0.0`) and `PORT` (`8000`) in application settings (`core/config.py`).

### Project Isolation & Tenant Boundary
The system enforces strict multi-tenancy anchored around the **Project** entity (`project_id`). All child resources (documents, document versions, conversations, messages, and vector/sparse embeddings) are scoped to a specific `project_id`. Attempting to access or link a resource with an mismatched `project_id` triggers an HTTP `404` or `400` boundary rejection.

### Standard Response Formats
- Successful responses return JSON with standard HTTP status codes (`200 OK`, `201 Created`, `204 No Content`).
- Error responses adhere to FastAPI's standard detail convention:
  ```json
  {
    "detail": "Descriptive error message"
  }
  ```

### Common HTTP Status Codes
| Status Code | Meaning | Typical Trigger |
| :--- | :--- | :--- |
| `200 OK` | Success | Successful GET, PATCH, or Retrieval request |
| `201 Created` | Created | Successful POST entity creation |
| `204 No Content` | No Content | Successful DELETE deletion |
| `400 Bad Request` | Validation Failure | Empty names, invalid roles, malformed queries |
| `404 Not Found` | Resource Not Found | Project, Document, Conversation, or Message ID not found |
| `409 Conflict` | State Conflict | Invalid document processing state transition |
| `422 Unprocessable`| Quality Gate Rejection | AI generation failed groundedness/safety checks |
| `429 Too Many Requests`| Provider Rate Limit | Upstream LLM provider rate limit exceeded |
| `502 Bad Gateway` | Upstream Failure | Storage (Supabase) or LLM provider authentication error |
| `504 Gateway Timeout`| Upstream Timeout | LLM inference or reranker provider timeout |

### Authentication Architecture
- **Current State**: All endpoints are currently **open and unauthenticated** at the HTTP routing layer. There are no API tokens, cookies, or authorization headers required to make calls.
- **Future Bearer Token Usage**: If authentication is activated, requests will require the standard HTTP Authorization header:
  ```http
  Authorization: Bearer <your_jwt_access_token>
  ```
  In Postman, select the **Authorization** tab, choose **Bearer Token**, and supply your token variable `{{accessToken}}`.

---

## System & Health

### Root Welcome Status

- **Purpose**: Verifies that the FastAPI server is reachable and reports the runtime environment.
- **HTTP Method**: `GET`
- **Endpoint**: `/`
- **Required Path/Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: None

#### Response Structure (200 OK)
```json
{
  "message": "Welcome to Agent MVP",
  "environment": "development"
}
```

#### cURL
```bash
curl -X GET "http://localhost:8000/"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/`
- **Headers**: None
- **Body**: None
- **Authentication**: Inherit auth from parent / No Auth

---

### Application Health Check

- **Purpose**: Liveness and readiness probe for container orchestrators and monitoring tools.
- **HTTP Method**: `GET`
- **Endpoint**: `/health`
- **Required Path/Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: None

#### Response Structure (200 OK)
```json
{
  "status": "ok",
  "app_name": "Agent MVP",
  "version": "0.1.0",
  "environment": "development"
}
```

#### cURL
```bash
curl -X GET "http://localhost:8000/health"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/health`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

## Authentication

> **Notice: No Authentication Endpoints Currently Implemented**  
> There are currently **no public authentication endpoints** (`/auth/login`, `/auth/signup`, `/auth/logout`, or `/auth/refresh`) implemented in the backend application.  
> 
> All API operations are currently public and unauthenticated. User tenant isolation is maintained logically by specifying the target `project_id` in path parameters. See [Missing, Broken, or Inconsistent APIs](#missing-broken-or-inconsistent-apis) for further details.

---

## Projects

Projects are the top-level tenant boundary. All documents, conversations, retrieval queries, and LLM responses must belong to an existing project.

---

### Create Project

- **Purpose**: Creates a new project container. The returned `id` is required for all project-scoped operations.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: None

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | string | **Yes** | Project name (1 to 255 characters; cannot be whitespace only) |
| `description` | string | No | Optional human-readable project description |

```json
{
  "name": "Legal Intelligence Hub",
  "description": "Enterprise compliance analysis and contractual RAG pipeline"
}
```

#### Response Structure (201 Created)
```json
{
  "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Legal Intelligence Hub",
  "description": "Enterprise compliance analysis and contractual RAG pipeline",
  "created_at": "2026-09-14T12:00:00.000000Z",
  "updated_at": "2026-09-14T12:00:00.000000Z"
}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Legal Intelligence Hub",
    "description": "Enterprise compliance analysis and contractual RAG pipeline"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "name": "Legal Intelligence Hub",
    "description": "Enterprise compliance analysis and contractual RAG pipeline"
  }
  ```
- **Authentication**: No Auth

---

### List Projects

- **Purpose**: Retrieves all projects ordered by creation date descending.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects`
- **Required Path Parameters**: None
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `limit` | integer | `100` | Maximum number of projects to return (`1` to `100`) |
  | `offset` | integer | `0` | Number of projects to skip (`>= 0`) |
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: None

#### Response Structure (200 OK)
```json
[
  {
    "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "name": "Legal Intelligence Hub",
    "description": "Enterprise compliance analysis and contractual RAG pipeline",
    "created_at": "2026-09-14T12:00:00.000000Z",
    "updated_at": "2026-09-14T12:00:00.000000Z"
  }
]
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects?limit=20&offset=0"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects`
- **Params**:
  - `limit`: `20`
  - `offset`: `0`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Get Project by ID

- **Purpose**: Retrieves details for a specific project by its unique ID.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Unique project identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project) or [List Projects](#list-projects).

#### Response Structure (200 OK)
```json
{
  "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Legal Intelligence Hub",
  "description": "Enterprise compliance analysis and contractual RAG pipeline",
  "created_at": "2026-09-14T12:00:00.000000Z",
  "updated_at": "2026-09-14T12:00:00.000000Z"
}
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Update Project

- **Purpose**: Updates the display name or description of an existing project.
- **HTTP Method**: `PATCH`
- **Endpoint**: `/projects/{project_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Unique project identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project) or [List Projects](#list-projects).

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | string | No | Updated project name (1 to 255 characters; non-empty) |
| `description` | string | No | Updated project description |

```json
{
  "name": "Legal & Compliance AI Hub",
  "description": "Updated enterprise compliance and legal retrieval knowledge base"
}
```

#### Response Structure (200 OK)
```json
{
  "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Legal & Compliance AI Hub",
  "description": "Updated enterprise compliance and legal retrieval knowledge base",
  "created_at": "2026-09-14T12:00:00.000000Z",
  "updated_at": "2026-09-14T12:30:00.000000Z"
}
```

#### cURL
```bash
curl -X PATCH "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Legal & Compliance AI Hub",
    "description": "Updated enterprise compliance and legal retrieval knowledge base"
  }'
```

#### Postman
- **Method**: `PATCH`
- **URL**: `http://localhost:8000/projects/:project_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "name": "Legal & Compliance AI Hub",
    "description": "Updated enterprise compliance and legal retrieval knowledge base"
  }
  ```
- **Authentication**: No Auth

---

### Delete Project

- **Purpose**: Permanently deletes a project, cascades deletion to all associated relational records (documents, versions, conversations, messages), and deletes all physical files in Supabase Object Storage.
- **HTTP Method**: `DELETE`
- **Endpoint**: `/projects/{project_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Unique project identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project).

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

## Sources / Documents

The Sources API manages uploaded knowledge base files (PDF, Markdown, TXT, etc.). A **Document** represents the logical entity, while a **DocumentVersion** represents a physical file revision stored in Supabase Object Storage.

---

### Upload Source Document

- **Purpose**: Uploads a new source document file to a project, stores the binary in Supabase Object Storage (`{project_id}/{document_id}/v1/{filename}`), creates the document record, and initializes Version 1 with status `pending`. (If `AUTO_PROCESS_DOCUMENTS=true` in backend configuration, synchronously triggers document parsing, chunking, and embedding).
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/sources`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Target project identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: multipart/form-data`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project).

#### Request Body Structure (Multipart Form-Data)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | File (Binary) | **Yes** | Binary file content to upload |
| `name` | string (Text) | No | Custom display name (defaults to uploaded filename if omitted) |

#### Response Structure (201 Created)
```json
{
  "id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Master Services Agreement 2026",
  "created_at": "2026-09-14T12:05:00.000000Z",
  "updated_at": "2026-09-14T12:05:00.000000Z",
  "latest_version": {
    "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
    "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "version_number": 1,
    "storage_bucket": "documents",
    "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/contract.pdf",
    "original_filename": "contract.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "etag": "\"a1b2c3d4e5f6\"",
    "status": "pending",
    "error_message": null,
    "indexed_at": null,
    "created_at": "2026-09-14T12:05:00.000000Z",
    "updated_at": "2026-09-14T12:05:00.000000Z"
  },
  "versions_count": 1
}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources" \
  -F "file=@/path/to/contract.pdf" \
  -F "name=Master Services Agreement 2026"
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/sources`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**: Leave blank or let Postman automatically manage `multipart/form-data` with boundaries.
- **Body** (`form-data`):
  | Key | Type | Value |
  | :--- | :--- | :--- |
  | `file` | File | Select local file `contract.pdf` |
  | `name` | Text | `Master Services Agreement 2026` |
- **Authentication**: No Auth

---

### List Source Documents

- **Purpose**: Lists all logical documents belonging to a project, including summary details of the latest physical version.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}/sources`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `limit` | integer | `100` | Maximum number of documents to return (`1` to `100`) |
  | `offset` | integer | `0` | Number of documents to skip (`>= 0`) |
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project).

#### Response Structure (200 OK)
```json
[
  {
    "id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "name": "Master Services Agreement 2026",
    "created_at": "2026-09-14T12:05:00.000000Z",
    "updated_at": "2026-09-14T12:05:00.000000Z",
    "latest_version": {
      "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
      "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
      "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
      "version_number": 1,
      "storage_bucket": "documents",
      "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/contract.pdf",
      "original_filename": "contract.pdf",
      "content_type": "application/pdf",
      "size_bytes": 1048576,
      "etag": "\"a1b2c3d4e5f6\"",
      "status": "ready",
      "error_message": null,
      "indexed_at": "2026-09-14T12:05:30.000000Z",
      "created_at": "2026-09-14T12:05:00.000000Z",
      "updated_at": "2026-09-14T12:05:30.000000Z"
    },
    "versions_count": 1
  }
]
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/05e40750-acbb-436a-8169-9cd7022f7a91/sources?limit=25&offset=0"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/sources`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `limit`: `25`
  - `offset`: `0`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Get Source Document Details

- **Purpose**: Retrieves a document along with its complete historical list of physical versions.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}/sources/{document_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `document_id` | string (UUID) | Logical document identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document) or [List Source Documents](#list-source-documents).

#### Response Structure (200 OK)
```json
{
  "id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Master Services Agreement 2026",
  "created_at": "2026-09-14T12:05:00.000000Z",
  "updated_at": "2026-09-14T12:05:00.000000Z",
  "latest_version": {
    "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
    "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "version_number": 1,
    "storage_bucket": "documents",
    "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/contract.pdf",
    "original_filename": "contract.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "etag": "\"a1b2c3d4e5f6\"",
    "status": "ready",
    "error_message": null,
    "indexed_at": "2026-09-14T12:05:30.000000Z",
    "created_at": "2026-09-14T12:05:00.000000Z",
    "updated_at": "2026-09-14T12:05:30.000000Z"
  },
  "versions_count": 1,
  "versions": [
    {
      "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
      "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
      "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
      "version_number": 1,
      "storage_bucket": "documents",
      "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/contract.pdf",
      "original_filename": "contract.pdf",
      "content_type": "application/pdf",
      "size_bytes": 1048576,
      "etag": "\"a1b2c3d4e5f6\"",
      "status": "ready",
      "error_message": null,
      "indexed_at": "2026-09-14T12:05:30.000000Z",
      "created_at": "2026-09-14T12:05:00.000000Z",
      "updated_at": "2026-09-14T12:05:30.000000Z"
    }
  ]
}
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Update Source Document Metadata

- **Purpose**: Updates the mutable display name of a source document under project isolation.
- **HTTP Method**: `PATCH`
- **Endpoint**: `/projects/{project_id}/sources/{document_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `document_id` | string (UUID) | Logical document identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document).

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | string | No | Updated display name (1 to 255 characters; non-empty) |

```json
{
  "name": "Master Services Agreement 2026 (Executed)"
}
```

#### Response Structure (200 OK)
```json
{
  "id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "name": "Master Services Agreement 2026 (Executed)",
  "created_at": "2026-09-14T12:05:00.000000Z",
  "updated_at": "2026-09-14T12:15:00.000000Z",
  "latest_version": {
    "id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
    "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "version_number": 1,
    "storage_bucket": "documents",
    "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v1/contract.pdf",
    "original_filename": "contract.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "etag": "\"a1b2c3d4e5f6\"",
    "status": "ready",
    "error_message": null,
    "indexed_at": "2026-09-14T12:05:30.000000Z",
    "created_at": "2026-09-14T12:05:00.000000Z",
    "updated_at": "2026-09-14T12:05:30.000000Z"
  },
  "versions_count": 1
}
```

#### cURL
```bash
curl -X PATCH "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Master Services Agreement 2026 (Executed)"
  }'
```

#### Postman
- **Method**: `PATCH`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "name": "Master Services Agreement 2026 (Executed)"
  }
  ```
- **Authentication**: No Auth

---

### Delete Source Document

- **Purpose**: Deletes a document, cascades deletion to all its versions and database chunk records, and deletes all associated physical files across all versions from Supabase Object Storage.
- **HTTP Method**: `DELETE`
- **Endpoint**: `/projects/{project_id}/sources/{document_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `document_id` | string (UUID) | Logical document identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `document_id`.

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Upload New Document Version

- **Purpose**: Uploads a new sequential file version (v2, v3, ...) for an existing document. Stores the file at `{project_id}/{document_id}/v{next_version}/{filename}` in Supabase Object Storage and initializes the version status to `pending`.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/sources/{document_id}/versions`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `document_id` | string (UUID) | Logical document identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: multipart/form-data`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document).

#### Request Body Structure (Multipart Form-Data)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | File (Binary) | **Yes** | Updated binary file content |

#### Response Structure (201 Created)
```json
{
  "id": "76c4e3d0-9012-4ef0-a345-678901cdefab",
  "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "version_number": 2,
  "storage_bucket": "documents",
  "storage_path": "b7e6c5a1-4321-4def-9876-543210abcdef/e4a2c1b8-7890-4cde-8123-456789abcdef/v2/contract_revised.pdf",
  "original_filename": "contract_revised.pdf",
  "content_type": "application/pdf",
  "size_bytes": 1052600,
  "etag": "\"f6e5d4c3b2a1\"",
  "status": "pending",
  "error_message": null,
  "indexed_at": null,
  "created_at": "2026-09-14T12:20:00.000000Z",
  "updated_at": "2026-09-14T12:20:00.000000Z"
}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef/versions" \
  -F "file=@/path/to/contract_revised.pdf"
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id/versions`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**: Leave blank or let Postman handle boundary automatically.
- **Body** (`form-data`):
  | Key | Type | Value |
  | :--- | :--- | :--- |
  | `file` | File | Select local revised file `contract_revised.pdf` |
- **Authentication**: No Auth

---

## Conversations

The Conversations API manages dialogue sessions within a project. Each conversation owns an append-only chronological history of message turns.

---

### Create Conversation

- **Purpose**: Creates a new conversation thread strictly scoped to the project boundary.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/conversations`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project).

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `title` | string | No | Conversation display title (max 255 chars; defaults to `"New Conversation"`) |

```json
{
  "title": "Vendor Liability & Termination Analysis"
}
```
*(Note: Passing an empty JSON `{}` is also valid and assigns the default title `"New Conversation"`).*

#### Response Structure (201 Created)
```json
{
  "id": "c3d2e1f0-1234-5678-9abc-def012345678",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "title": "Vendor Liability & Termination Analysis",
  "created_at": "2026-09-14T12:10:00.000000Z",
  "updated_at": "2026-09-14T12:10:00.000000Z",
  "messages_count": 0
}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Vendor Liability & Termination Analysis"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/conversations`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "title": "Vendor Liability & Termination Analysis"
  }
  ```
- **Authentication**: No Auth

---

### List Conversations

- **Purpose**: Lists all conversations belonging to the project boundary, ordered by creation date descending.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}/conversations`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `limit` | integer | `100` | Maximum number of conversations to return (`1` to `100`) |
  | `offset` | integer | `0` | Number of conversations to skip (`>= 0`) |
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project).

#### Response Structure (200 OK)
```json
[
  {
    "id": "c3d2e1f0-1234-5678-9abc-def012345678",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "title": "Vendor Liability & Termination Analysis",
    "created_at": "2026-09-14T12:10:00.000000Z",
    "updated_at": "2026-09-14T12:10:00.000000Z",
    "messages_count": 2
  }
]
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations?limit=50&offset=0"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/conversations`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `limit`: `50`
  - `offset`: `0`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Get Conversation Details & History

- **Purpose**: Retrieves conversation metadata along with its complete, chronological list of message turns.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `conversation_id` | string (UUID) | Conversation identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `conversation_id`.

#### Response Structure (200 OK)
```json
{
  "id": "c3d2e1f0-1234-5678-9abc-def012345678",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "title": "Vendor Liability & Termination Analysis",
  "created_at": "2026-09-14T12:10:00.000000Z",
  "updated_at": "2026-09-14T12:10:00.000000Z",
  "messages_count": 2,
  "messages": [
    {
      "id": "11111111-2222-3333-4444-555555555555",
      "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
      "role": "user",
      "content": "What notice period is required to terminate the contract without cause?",
      "metadata": {},
      "created_at": "2026-09-14T12:11:00.000000Z"
    },
    {
      "id": "66666666-7777-8888-9999-000000000000",
      "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
      "role": "assistant",
      "content": "According to Section 9.2, either party may terminate without cause by providing at least 60 days written notice.",
      "metadata": {
        "user_message_id": "11111111-2222-3333-4444-555555555555",
        "finish_reason": "stop",
        "model": "gpt-4o",
        "retrieval": {
          "chunks_count": 3,
          "duration_ms": 42.5
        }
      },
      "created_at": "2026-09-14T12:11:05.000000Z"
    }
  ]
}
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

### Update Conversation Title

- **Purpose**: Updates the display title of an existing conversation thread.
- **HTTP Method**: `PATCH`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `conversation_id` | string (UUID) | Conversation identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `conversation_id`.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `title` | string | **Yes** | Updated display title (1 to 255 characters; non-empty) |

```json
{
  "title": "Vendor Liability & Notice Analysis (Reviewed)"
}
```

#### Response Structure (200 OK)
```json
{
  "id": "c3d2e1f0-1234-5678-9abc-def012345678",
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "title": "Vendor Liability & Notice Analysis (Reviewed)",
  "created_at": "2026-09-14T12:10:00.000000Z",
  "updated_at": "2026-09-14T12:15:00.000000Z",
  "messages_count": 2
}
```

#### cURL
```bash
curl -X PATCH "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Vendor Liability & Notice Analysis (Reviewed)"
  }'
```

#### Postman
- **Method**: `PATCH`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "title": "Vendor Liability & Notice Analysis (Reviewed)"
  }
  ```
- **Authentication**: No Auth

---

### Delete Conversation

- **Purpose**: Deletes a conversation and cascades deletion to all messages contained within it.
- **HTTP Method**: `DELETE`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `conversation_id` | string (UUID) | Conversation identifier |
- **Query Parameters**: None
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `conversation_id`.

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

## Messages API (Multi-turn History)

The Messages API powers multi-turn dialogue within a conversation by persisting and listing message turns (`user` or `assistant`).

> **Architectural Boundary Notice (Phase 1 — RAG Generation Layer Retired)**:
> In earlier versions, `POST .../messages` supported a query parameter `generate=true` that ran an end-to-end RAG retrieval + LLM synthesis + groundedness evaluation pipeline within RAG. In Phase 1 of the BRD Agent architecture, this RAG-owned generation layer has been retired. RAG is strictly responsible for document processing, indexing, and retrieval (`POST /projects/{project_id}/retrieval`), returning a clean `RetrievalResult`. Multi-turn dialogue generation, evaluation, reasoning, and document synthesis will be owned by the future BRD Agent.

---

### Create Message Turn

- **Purpose**: Appends and persists a single message turn (`user` or `assistant`) into the conversation history. Returns the created `MessageResponse`.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}/messages`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `conversation_id` | string (UUID) | Target conversation identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `conversation_id`.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `content` | string | **Yes** | Message text (non-empty) |
| `role` | string | No | Message sender role: `"user"` or `"assistant"` (default: `"user"`) |
| `metadata` | object | No | Optional arbitrary key-value JSON metadata |

```json
{
  "content": "What is the maximum liability cap specified in the agreement?",
  "role": "user",
  "metadata": {
    "client_session": "web-client-v1"
  }
}
```

#### Response Structure (201 Created)
```json
{
  "id": "66666666-5555-4444-3333-222222222222",
  "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
  "role": "user",
  "content": "What is the maximum liability cap specified in the agreement?",
  "metadata": {
    "client_session": "web-client-v1"
  },
  "created_at": "2026-09-14T12:12:00.000000Z"
}
```

#### cURL (Message Persistence)
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What is the maximum liability cap specified in the agreement?",
    "role": "user"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id/messages`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "content": "What is the maximum liability cap specified in the agreement?",
    "role": "user"
  }
  ```
- **Authentication**: No Auth

---

### List Chronological Messages

- **Purpose**: Retrieves all message turns within a conversation in ascending chronological order.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}/messages`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier |
  | `conversation_id` | string (UUID) | Conversation identifier |
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `limit` | integer | `100` | Maximum number of messages to return (`1` to `100`) |
  | `offset` | integer | `0` | Number of messages to skip (`>= 0`) |
- **Required Headers**: None
- **Request Body**: None
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` and `conversation_id`.

#### Response Structure (200 OK)
```json
[
  {
    "id": "66666666-5555-4444-3333-222222222222",
    "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
    "role": "user",
    "content": "What is the maximum liability cap specified in the agreement?",
    "metadata": {},
    "created_at": "2026-09-14T12:11:58.000000Z"
  },
  {
    "id": "77777777-8888-9999-aaaa-bbbbbbbbbbbb",
    "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
    "role": "assistant",
    "content": "Under Section 11.1 of the agreement, the aggregate liability of either party is capped at twelve (12) months of fees paid immediately preceding the claim.",
    "metadata": {
      "user_message_id": "66666666-5555-4444-3333-222222222222",
      "finish_reason": "stop",
      "model": "gpt-4o"
    },
    "created_at": "2026-09-14T12:12:00.000000Z"
  }
]
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678/messages?limit=50&offset=0"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id/messages`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
  - `limit`: `50`
  - `offset`: `0`
- **Headers**: None
- **Body**: None
- **Authentication**: No Auth

---

## Knowledge Retrieval

The Knowledge Retrieval API executes the complete multi-stage RAG retrieval pipeline without invoking the LLM synthesis or conversation persistence layers. It is ideal for semantic search, search inspection, and debugging relevance ranking.

---

### Execute Project Knowledge Retrieval

- **Purpose**: Executes query preprocessing, query transformation (expansion/rewriting), dense vector search via Qdrant, sparse keyword search via Qdrant, Reciprocal Rank Fusion (RRF), metadata filtering, cross-encoder reranking, PostgreSQL chunk hydration, context assembly, and text formatting strictly within the project boundary.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/retrieval`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Target project identifier |
- **Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Ensure source documents have been indexed into the project.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | string | **Yes** | Search query string (non-empty) |
| `top_k` | integer | No | Maximum number of hydrated chunks to return (gt: `0`) |
| `dense_top_k` | integer | No | Vector search candidate limit (gt: `0`) |
| `sparse_top_k` | integer | No | Keyword search candidate limit (gt: `0`) |
| `enable_transformation` | boolean | No | Toggle query transformation (rewriting/expansion) stage |
| `enable_sparse` | boolean | No | Toggle sparse keyword retrieval branch |
| `enable_reranking` | boolean | No | Toggle cross-encoder reranking stage |
| `enable_relevance_check` | boolean | No | Toggle post-retrieval relevance evaluation gate |
| `metadata_filters` | object | No | Optional key-value constraints to filter candidates |

```json
{
  "query": "termination notice period and breach conditions",
  "top_k": 3,
  "dense_top_k": 10,
  "sparse_top_k": 10,
  "enable_transformation": true,
  "enable_sparse": true,
  "enable_reranking": true,
  "enable_relevance_check": false,
  "metadata_filters": {
    "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef"
  }
}
```

#### Response Structure (200 OK)
```json
{
  "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "original_query": "termination notice period and breach conditions",
  "retrieval_query": "What are the contractual terms regarding notice period for termination and material breach?",
  "chunk_count": 2,
  "chunks": [
    {
      "chunk_id": "chunk-9988-7766",
      "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
      "document_version_id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
      "content": "Section 9.2 Termination for Convenience. Either party may terminate this Agreement without cause upon providing at least sixty (60) days prior written notice to the other party.",
      "rank": 1,
      "score": 0.945,
      "heading": "Section 9.2 Termination for Convenience",
      "section_path": [
        "Article 9: Term and Termination",
        "Section 9.2 Termination for Convenience"
      ],
      "contextual_content": null,
      "metadata": {
        "page_number": 14,
        "source": "contract.pdf"
      }
    },
    {
      "chunk_id": "chunk-9988-7767",
      "document_id": "e4a2c1b8-7890-4cde-8123-456789abcdef",
      "document_version_id": "f5b3d2c9-8901-4def-9234-567890bcdefa",
      "content": "Section 9.3 Termination for Material Breach. If either party materially breaches any provision of this Agreement, the non-breaching party may terminate immediately upon thirty (30) days written notice if the breach remains uncured.",
      "rank": 2,
      "score": 0.912,
      "heading": "Section 9.3 Termination for Material Breach",
      "section_path": [
        "Article 9: Term and Termination",
        "Section 9.3 Termination for Material Breach"
      ],
      "contextual_content": null,
      "metadata": {
        "page_number": 14,
        "source": "contract.pdf"
      }
    }
  ],
  "formatted_context": "RETRIEVED CONTEXT\n\n[Context 1]\nSection 9.2 Termination for Convenience. Either party may terminate this Agreement without cause upon providing at least sixty (60) days prior written notice to the other party.\n\n[Context 2]\nSection 9.3 Termination for Material Breach. If either party materially breaches any provision of this Agreement, the non-breaching party may terminate immediately upon thirty (30) days written notice if the breach remains uncured.\n\nEND RETRIEVED CONTEXT",
  "execution_metadata": {
    "total_duration_ms": 45.8,
    "attempts_count": 1,
    "fallback_triggered": false,
    "attempts": [
      {
        "attempt": 1,
        "query": "What are the contractual terms regarding notice period for termination and material breach?",
        "is_transformed": true,
        "transformed_query": "What are the contractual terms regarding notice period for termination and material breach?",
        "strategy_used": "llm_rewrite",
        "dense_candidates_count": 10,
        "sparse_candidates_count": 10,
        "fused_candidates_count": 15,
        "filtered_candidates_count": 15,
        "reranked_candidates_count": 3,
        "hydrated_candidates_count": 2,
        "is_relevant": true,
        "relevance_reason": null,
        "latency_ms": 45.8
      }
    ],
    "stage_latencies_ms": {
      "query_transformation_ms": 12.0,
      "dense_search_ms": 8.5,
      "sparse_search_ms": 4.1,
      "fusion_ms": 1.2,
      "rerank_ms": 14.3,
      "hydration_ms": 5.7
    }
  }
}
```

#### cURL
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "termination notice period and breach conditions",
    "top_k": 3,
    "enable_transformation": true,
    "enable_sparse": true,
    "enable_reranking": true
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/retrieval`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "query": "termination notice period and breach conditions",
    "top_k": 3,
    "enable_transformation": true,
    "enable_sparse": true,
    "enable_reranking": true
  }
  ```
- **Authentication**: No Auth

---

## Missing, Broken, or Inconsistent APIs

The following observations, gaps, and design inconsistencies were identified during this comprehensive architectural inspection:

### 1. Authentication Endpoints are Completely Missing
- **Current Situation**: The application lacks user management, signup, login, session validation, or API key generation endpoints.
- **Impact**: All endpoints are public and unprotected at the network level. Security and tenant isolation rely entirely on client-supplied `project_id` path parameters.

### 2. Lack of Manual Document Processing / Indexing Trigger Endpoint
- **Current Situation**: `DocumentProcessingService` implements parsing, cleaning, chunking, embedding generation, and vector indexing. However, there is **no public API endpoint** (e.g. `POST /projects/{project_id}/sources/{document_id}/process` or `POST .../versions/{version_id}/index`) to trigger or retry indexing.
- **Impact**: In the default configuration (`AUTO_PROCESS_DOCUMENTS=false`), uploaded files remain in `"status": "pending"` indefinitely unless an external worker or custom test script invokes `DocumentProcessingService.trigger_processing()`.

### 3. Retired RAG Generation Layer (Phase 1)
- **Current Situation**: In Phase 1 of the BRD Agent transition, the RAG-owned generation layer (`GenerationService`, `schemas/generation.py`, `rag/generation/`, and the `generate=true` parameter on the messages endpoint) has been formally retired.
- **Architectural Shift**: RAG strictly retrieves knowledge evidence (`RetrievalResult`), while reasoning, LLM generation, validation, and multi-turn response synthesis will be owned by the future BRD Agent in subsequent phases.

### 4. Unimplemented Individual Message Modification & Deletion
- **Current Situation**: FastAPI exception handlers in `app/main.py` explicitly handle `MessageNotFoundError` and `ConversationMessageMismatchError`. However, `api/conversations.py` only defines endpoints to create and list messages. There are no endpoints to update (`PATCH .../messages/{message_id}`) or delete (`DELETE .../messages/{message_id}`) an individual message.

### 5. No Standalone Endpoint for Single Document Version Status
- **Current Situation**: To inspect the indexing status of an uploaded document version (e.g., checking if it changed from `pending` to `ready`), clients cannot query a single version endpoint (e.g., `GET /projects/{project_id}/sources/{document_id}/versions/{version_id}`).
- **Workaround**: Clients must query the parent document via `GET /projects/{project_id}/sources/{document_id}` and inspect the `latest_version` or the `versions` array.
