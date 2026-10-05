# API Usage

This document serves as the complete, authoritative guide to all currently available public HTTP APIs in the application. Every endpoint documented here reflects the active backend routing, Pydantic schemas, database models, and service implementations.

---

## Table of Contents

- [Overview & Global Conventions](#overview--global-conventions)
- [System & Health](#system--health)
  - [Root Welcome Status](#root-welcome-status)
  - [Application Health Check](#application-health-check)
- [Authentication](#authentication)
  - [User Signup](#user-signup)
  - [User Login](#user-login)
  - [Refresh Access Token](#refresh-access-token)
  - [User Logout](#user-logout)
  - [Get Current Authenticated User](#get-current-authenticated-user)
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

### User Ownership & Project Tenancy Boundary
The system enforces strict multi-tenancy anchored around the **User** (`user_id`) and **Project** (`project_id`) entities:
- **User → Projects**: Every project is owned by the user who created it (`user_id`). Users can only list, view, update, and delete their own projects.
- **Projects → Child Resources**: All child resources (documents, document versions, conversations, messages, chunks, and vector/sparse embeddings) inherit tenancy from their parent project boundary.
- **Strict Boundary Rejection**: Attempting to access, modify, or query a project or child resource that belongs to another user (or mismatched project) triggers an HTTP `404 Not Found` response. Returning `404` rather than `403` ensures that the existence of other tenants' resources is not leaked.

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
| `201 Created` | Created | Successful POST entity creation (User, Project, Document, Conversation, Message) |
| `204 No Content` | No Content | Successful DELETE deletion |
| `400 Bad Request` | Validation Failure | Empty names, invalid roles, malformed queries |
| `401 Unauthorized` | Authentication Failure | Missing, invalid, or expired Bearer JWT access token, or invalid login credentials |
| `404 Not Found` | Resource Not Found | Project, Document, Conversation, or Message ID not found, or resource belongs to another user |
| `409 Conflict` | Conflict / Duplicate | Email already registered during signup, or invalid document state transition |
| `422 Unprocessable`| Validation / Quality Rejection | Request body schema failure or AI generation failed groundedness checks |
| `429 Too Many Requests`| Rate Limit Exceeded | Login rate limit exceeded (5 attempts / 5 min window) or upstream LLM provider limit |
| `502 Bad Gateway` | Upstream Failure | Storage (Supabase) or LLM provider authentication error |
| `504 Gateway Timeout`| Upstream Timeout | LLM inference or reranker provider timeout |

### Authentication Architecture
- **Bearer Token Authentication**: All protected application endpoints (`/projects`, `/projects/{project_id}/*`, `/auth/me`) require a valid Bearer JWT access token passed in the HTTP `Authorization` header:
  ```http
  Authorization: Bearer <your_jwt_access_token>
  ```
  In Postman, select the **Authorization** tab, choose **Bearer Token**, and supply your token variable `{{accessToken}}`.
- **JWT Access Tokens**: Short-lived (15-minute default) tokens containing standard claims (`sub` = user ID, `email`, `jti`, `exp`, `iat`) signed with HMAC-SHA256. Access tokens are decoded and validated statelessly on every protected request.
- **Opaque Refresh Tokens**: Cryptographically random 32-byte URL-safe strings used to obtain new access tokens via `POST /auth/refresh`. Refresh tokens are persisted strictly as SHA-256 digests in the database (`refresh_tokens` table) with expiration and revocation tracking.
- **Single-Use Rotation & Reuse Detection**: Each refresh token can only be used once. Upon refresh, the previous token is revoked and a new refresh token is issued. If an already-revoked refresh token is presented, the system detects potential token theft and immediately revokes **all** active refresh tokens for that user.
- **Login Rate Limiting**: An in-memory sliding window rate limiter protects `POST /auth/login`. By default, 5 login attempts are allowed per 5-minute sliding window per client IP + email. Exceeding this threshold returns HTTP `429 Too Many Requests` with a standard `Retry-After` header.

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

The authentication system provides secure user registration, credential verification, stateless JWT access token validation, and rotating refresh tokens with built-in reuse detection.

- **Password Security**: Passwords are verified and hashed using **Argon2id** (`argon2-cffi`).
- **Access Tokens**: Short-lived (15-minute default) HMAC-SHA256 JWTs providing stateless authorization on protected endpoints.
- **Refresh Tokens**: Opaque 32-byte URL-safe strings stored strictly as SHA-256 hashes in the database.
- **Single-Use Rotation**: Refreshing an access token automatically revokes the submitted refresh token and returns a newly generated pair.
- **Reuse Detection**: Presenting a previously revoked refresh token triggers emergency mitigation: all active refresh tokens for that user account are revoked immediately.
- **Rate Limiting**: `POST /auth/login` is guarded by an in-memory sliding window rate limiter (default: 5 attempts per 5 minutes per IP + email). Exceeding this returns HTTP `429 Too Many Requests` with a `Retry-After` header.

---

### User Signup

- **Purpose**: Creates a new user account with an email address and strong password.
- **HTTP Method**: `POST`
- **Endpoint**: `/auth/signup`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Prerequisites**: Email must not already exist in the system.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `email` | string | **Yes** | Valid email address |
| `password` | string | **Yes** | Account password (minimum 8 characters) |

```json
{
  "email": "analyst@enterprise.com",
  "password": "CorrectHorseBatteryStaple123!"
}
```

#### Response Structure (201 Created)
```json
{
  "id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
  "email": "analyst@enterprise.com",
  "is_active": true,
  "created_at": "2026-10-05T06:00:00.000000Z",
  "updated_at": "2026-10-05T06:00:00.000000Z"
}
```

#### Error Responses
- `400 Bad Request` / `422 Unprocessable Entity`: Password is less than 8 characters, or email is malformed.
- `409 Conflict`: Email address is already registered:
  ```json
  {
    "detail": "User with email 'analyst@enterprise.com' already exists."
  }
  ```

#### cURL
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "analyst@enterprise.com",
    "password": "CorrectHorseBatteryStaple123!"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/auth/signup`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "email": "analyst@enterprise.com",
    "password": "CorrectHorseBatteryStaple123!"
  }
  ```
- **Authentication**: No Auth

---

### User Login

- **Purpose**: Authenticates user credentials and issues a JWT access token and opaque rotating refresh token.
- **HTTP Method**: `POST`
- **Endpoint**: `/auth/login`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None
- **Rate Limit**: 5 attempts per 5-minute sliding window per client IP + email.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `email` | string | **Yes** | Registered account email |
| `password` | string | **Yes** | Account password |

```json
{
  "email": "analyst@enterprise.com",
  "password": "CorrectHorseBatteryStaple123!"
}
```

#### Response Structure (200 OK)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1NGYzZTJkMS05ODc2LTRhYmMtZGVmMC0xMjM0NTY3ODkwYWIiLCJlbWFpbCI6ImFuYWx5c3RAZW50ZXJwcmlzZS5jb20iLCJqdGkiOiJmNGJhMzg1Ny1iMWUzLTRjMDMtYTk4MC05ODBiZDQ2MTQ4N2MiLCJleHAiOjE3NTk2NDQ5MDAsImlhdCI6MTc1OTY0NDAwMH0.sample_signature_jwt",
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",
  "token_type": "bearer",
  "expires_in": 900
}
```

#### Error Responses
- `401 Unauthorized`: Invalid email, incorrect password, or deactivated user account:
  ```json
  {
    "detail": "Invalid email or password."
  }
  ```
- `429 Too Many Requests`: Login rate limit exceeded. The response includes a `Retry-After` header indicating the number of seconds to wait:
  ```json
  {
    "detail": "Too many failed login attempts. Please try again in 294 seconds."
  }
  ```

#### cURL
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "analyst@enterprise.com",
    "password": "CorrectHorseBatteryStaple123!"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/auth/login`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "email": "analyst@enterprise.com",
    "password": "CorrectHorseBatteryStaple123!"
  }
  ```
- **Authentication**: No Auth
- **Postman Tests Script** (Save tokens automatically):
  ```javascript
  if (pm.response.code === 200) {
    const data = pm.response.json();
    pm.environment.set("accessToken", data.access_token);
    pm.environment.set("refreshToken", data.refresh_token);
  }
  ```

---

### Refresh Access Token

- **Purpose**: Obtains a fresh JWT access token and rotated refresh token by submitting an active refresh token.
- **HTTP Method**: `POST`
- **Endpoint**: `/auth/refresh`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None (token provided in body)
- **Rotation Behavior**: The submitted refresh token is permanently revoked upon success, and a newly generated refresh token is returned in the response.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `refresh_token` | string | **Yes** | Opaque refresh token received from login or previous refresh |

```json
{
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2"
}
```

#### Response Structure (200 OK)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1NGYzZTJkMS05ODc2LTRhYmMtZGVmMC0xMjM0NTY3ODkwYWIiLCJlbWFpbCI6ImFuYWx5c3RAZW50ZXJwcmlzZS5jb20iLCJqdGkiOiJlMWMyYjNhNC1kNWU2LTRmNzgtYTk4MC05ODBiZDQ2MTQ4N2MiLCJleHAiOjE3NTk2NDU4MDAsImlhdCI6MTc1OTY0NDkwMH0.sample_new_signature_jwt",
  "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3",
  "token_type": "bearer",
  "expires_in": 900
}
```

#### Error Responses
- `401 Unauthorized`: Token is malformed, expired, already revoked, or user is inactive.
  ```json
  {
    "detail": "Invalid, expired, or revoked refresh token."
  }
  ```
  > **Note on Token Theft Protection**: If a previously revoked token is reused, all active refresh tokens for the associated user account are immediately revoked.

#### cURL
```bash
curl -X POST "http://localhost:8000/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/auth/refresh`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "refresh_token": "{{refreshToken}}"
  }
  ```
- **Authentication**: No Auth
- **Postman Tests Script**:
  ```javascript
  if (pm.response.code === 200) {
    const data = pm.response.json();
    pm.environment.set("accessToken", data.access_token);
    pm.environment.set("refreshToken", data.refresh_token);
  }
  ```

---

### User Logout

- **Purpose**: Explicitly revokes an active refresh token, preventing further token renewals.
- **HTTP Method**: `POST`
- **Endpoint**: `/auth/logout`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Content-Type: application/json`
- **Authentication**: None (token provided in body)

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `refresh_token` | string | **Yes** | Active refresh token to be revoked |

```json
{
  "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3"
}
```

#### Response Structure (200 OK)
```json
{
  "message": "Logged out successfully."
}
```

#### Error Responses
- `400 Bad Request` / `422 Unprocessable Entity`: Missing `refresh_token` in request body.

#### cURL
```bash
curl -X POST "http://localhost:8000/auth/logout" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3"
  }'
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/auth/logout`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "refresh_token": "{{refreshToken}}"
  }
  ```
- **Authentication**: No Auth

---

### Get Current Authenticated User

- **Purpose**: Returns the profile and identity of the authenticated user resolved from the Bearer JWT access token.
- **HTTP Method**: `GET`
- **Endpoint**: `/auth/me`
- **Required Path/Query Parameters**: None
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)

#### Response Structure (200 OK)
```json
{
  "id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
  "email": "analyst@enterprise.com",
  "is_active": true,
  "created_at": "2026-10-05T06:00:00.000000Z",
  "updated_at": "2026-10-05T06:00:00.000000Z"
}
```

#### Error Responses
- `401 Unauthorized`: Token is missing, expired, invalid signature, or user is disabled:
  ```json
  {
    "detail": "Could not validate credentials."
  }
  ```

#### cURL
```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/auth/me`
- **Headers**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Projects

Projects are the top-level tenant boundary. Every project is strictly owned by the authenticated **User** who created it (`user_id`). All documents, conversations, retrieval queries, and LLM responses must belong to an existing project owned by the caller. Attempting to query, modify, or delete a project belonging to another user results in an HTTP `404 Not Found`.

---

### Create Project

- **Purpose**: Creates a new project container owned by the authenticated user. The returned `id` is required for all project-scoped operations.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects`
- **Required Path/Query Parameters**: None
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Valid access token from [User Login](#user-login) or [User Signup](#user-signup).

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
  "user_id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

### List Projects

- **Purpose**: Retrieves all projects owned by the currently authenticated user ordered by creation date descending.
- **HTTP Method**: `GET`
- **Endpoint**: `/projects`
- **Required Path Parameters**: None
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `limit` | integer | `100` | Maximum number of projects to return (`1` to `100`) |
  | `offset` | integer | `0` | Number of projects to skip (`>= 0`) |
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Valid access token.

#### Response Structure (200 OK)
```json
[
  {
    "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "user_id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
    "name": "Legal Intelligence Hub",
    "description": "Enterprise compliance analysis and contractual RAG pipeline",
    "created_at": "2026-09-14T12:00:00.000000Z",
    "updated_at": "2026-09-14T12:00:00.000000Z"
  }
]
```

#### cURL
```bash
curl -X GET "http://localhost:8000/projects?limit=20&offset=0" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects`
- **Params**:
  - `limit`: `20`
  - `offset`: `0`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project) or [List Projects](#list-projects). Project must belong to caller.

#### Response Structure (200 OK)
```json
{
  "id": "b7e6c5a1-4321-4def-9876-543210abcdef",
  "user_id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
  "name": "Legal Intelligence Hub",
  "description": "Enterprise compliance analysis and contractual RAG pipeline",
  "created_at": "2026-09-14T12:00:00.000000Z",
  "updated_at": "2026-09-14T12:00:00.000000Z"
}
```

#### Error Responses
- `401 Unauthorized`: Missing or invalid Bearer access token.
- `404 Not Found`: Project ID does not exist or belongs to another user.

#### cURL
```bash
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

### Update Project

- **Purpose**: Updates the display name or description of an existing project owned by the authenticated user.
- **HTTP Method**: `PATCH`
- **Endpoint**: `/projects/{project_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Unique project identifier |
- **Query Parameters**: None
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Project must belong to the authenticated caller.

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
  "user_id": "u4f3e2d1-9876-4abc-def0-1234567890ab",
  "name": "Legal & Compliance AI Hub",
  "description": "Updated enterprise compliance and legal retrieval knowledge base",
  "created_at": "2026-09-14T12:00:00.000000Z",
  "updated_at": "2026-09-14T12:30:00.000000Z"
}
```

#### Error Responses
- `401 Unauthorized`: Missing or invalid Bearer access token.
- `404 Not Found`: Project ID does not exist or belongs to another user.

#### cURL
```bash
curl -X PATCH "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

### Delete Project

- **Purpose**: Permanently deletes a project, cascades deletion to all associated relational records (documents, versions, conversations, messages), and deletes all physical files in Supabase Object Storage and vector points in Qdrant.
- **HTTP Method**: `DELETE`
- **Endpoint**: `/projects/{project_id}`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Unique project identifier |
- **Query Parameters**: None
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Project must belong to the authenticated caller.

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### Error Responses
- `401 Unauthorized`: Missing or invalid Bearer access token.
- `404 Not Found`: Project ID does not exist or belongs to another user.

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Sources / Documents

The Sources API manages uploaded knowledge base files (PDF, Markdown, TXT, etc.). A **Document** represents the logical entity, while a **DocumentVersion** represents a physical file revision stored in Supabase Object Storage.

All source operations require Bearer token authentication and verify that the target `project_id` belongs to the authenticated user. Operations on projects belonging to other users return HTTP `404 Not Found`.

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
- **Required Headers**:
  - `Content-Type: multipart/form-data`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Target project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Project must belong to caller.

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
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources?limit=25&offset=0" \
  -H "Authorization: Bearer <your_access_token>"
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document) or [List Source Documents](#list-source-documents). Project must belong to caller.

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
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document). Project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `document_id`. Project must belong to caller.

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/sources/e4a2c1b8-7890-4cde-8123-456789abcdef" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id/sources/:document_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `document_id`: `e4a2c1b8-7890-4cde-8123-456789abcdef`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**:
  - `Content-Type: multipart/form-data`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `document_id` from [Upload Source Document](#upload-source-document). Project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Conversations

The Conversations API manages dialogue sessions within a project. Each conversation owns an append-only chronological history of message turns.

All conversation operations require Bearer token authentication and verify that the target `project_id` belongs to the authenticated user. Operations on projects belonging to other users return HTTP `404 Not Found`.

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
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Project must belong to caller.

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
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations?limit=50&offset=0" \
  -H "Authorization: Bearer <your_access_token>"
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `conversation_id`. Project must belong to caller.

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
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `GET`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `conversation_id`. Project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `conversation_id`. Project must belong to caller.

#### Response Structure (204 No Content)
- **Status**: `204 No Content`
- **Body**: Empty

#### cURL
```bash
curl -X DELETE "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678" \
  -H "Authorization: Bearer <your_access_token>"
```

#### Postman
- **Method**: `DELETE`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
- **Headers**: None
- **Body**: None
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Messages API (Multi-turn History & Agent Streaming)

The Messages API powers multi-turn dialogue within a conversation. Sending a `user` message invokes the **BRD Lead Agent** runtime under strict project boundary isolation, streams the agent response in real time via Server-Sent Events (SSE), and automatically persists both user and assistant message turns.

All message operations require Bearer token authentication and verify that the target `project_id` belongs to the authenticated user. In addition, the authenticated user identity (`user_id`) is automatically injected into the agent's context (`AgentContext.user_id`). Operations on projects belonging to other users return HTTP `404 Not Found`.

> **Agent-Driven Dialogue Architecture**:
> `POST /projects/{project_id}/conversations/{conversation_id}/messages` is the single application boundary for chat.
> - The application establishes and enforces `project_id` isolation before invoking the agent runtime; the agent never selects the project.
> - The existing `conversation_id` is mapped directly to the agent's execution thread identity (`thread_id=conversation_id`), preserving multi-turn memory and execution state across turns.
> - Responses are streamed incrementally over HTTP SSE (`text/event-stream`), filtering out internal tool calls while delivering real-time tokens to the frontend.

---

### Create Message Turn / Invoke Agent

- **Purpose**: Persists a message turn and, for user messages, invokes the `BRDLeadAgent` to stream the assistant's response.
- **HTTP Method**: `POST`
- **Endpoint**: `/projects/{project_id}/conversations/{conversation_id}/messages`
- **Required Path Parameters**:
  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `project_id` | string (UUID) | Owning project identifier (tenant boundary) |
  | `conversation_id` | string (UUID) | Target conversation identifier (agent `thread_id`) |
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  | :--- | :--- | :--- | :--- |
  | `stream` | boolean | `true` | When `true` (default) and `role="user"`, returns a `text/event-stream` SSE response streaming the agent's output. When `false` and `role="user"`, executes the full 9-phase BRD workflow synchronously and returns the generated assistant message as HTTP `201 Created` (`MessageResponse`). |
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project) and `conversation_id` from [Create Conversation](#create-conversation). Target project must belong to caller.

#### Request Body Structure
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `content` | string | **Yes** | Message text (non-empty) |
| `role` | string | No | Message sender role: `"user"` or `"assistant"` (default: `"user"`) |
| `stream` | boolean | No | Optional override for streaming behavior (takes precedence over query param) |
| `metadata` | object | No | Optional arbitrary key-value JSON metadata |

```json
{
  "content": "Draft the Executive Summary section for the BRD based on the uploaded RFP.",
  "role": "user",
  "metadata": {
    "client_session": "web-client-v1"
  }
}
```

#### Streaming Response Behavior (`stream=true`, `role="user"`)
Returns HTTP `200 OK` with `Content-Type: text/event-stream; charset=utf-8` and headers `Cache-Control: no-cache`, `X-Accel-Buffering: no`.

The stream emits standard SSE events:

1. **Content Chunks (`event: message`)**:
   Emitted incrementally as the agent generates tokens.
   ```text
   event: message
   data: {"type": "content", "content": "The Executive Summary "}

   event: message
   data: {"type": "content", "content": "outlines the key business objectives..."}
   ```

2. **Completion Event (`event: done`)**:
   Emitted when the agent finishes execution and the final assistant response has been persisted to the database. The `data` payload contains the persisted `MessageResponse`.
   ```text
   event: done
   data: {"id": "77777777-8888-9999-aaaa-bbbbbbbbbbbb", "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678", "role": "assistant", "content": "The Executive Summary outlines the key business objectives...", "metadata": {"agent_run_id": "88888888-7777-6666-5555-444444444444", "user_message_id": "66666666-5555-4444-3333-222222222222", "workflow_state": {...}}, "created_at": "2026-09-14T12:12:05.000000Z"}
   ```

3. **Error Event (`event: error`)**:
   Emitted if an execution, LLM provider, or persistence error occurs during streaming.
   ```text
   event: error
   data: {"error": "LLM provider timeout during inference"}
   ```

#### Non-Streaming Agent Execution (`stream=false`, `role="user"`)
When `stream=false` (via query parameter or body payload) and `role="user"`, the endpoint executes the exact same 9-phase BRD workflow synchronously via `BRDLeadAgent.run_workflow_async`.
It persists both the incoming user message and the generated assistant response to the conversation database, returning HTTP `201 Created` with the **assistant** `MessageResponse` JSON including durable `workflow_state` metadata:

```json
{
  "id": "77777777-8888-9999-aaaa-bbbbbbbbbbbb",
  "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
  "role": "assistant",
  "content": "## 1.0 Executive Summary\n\nThis Business Requirements Document defines the core payment gateway architecture...",
  "metadata": {
    "user_message_id": "66666666-5555-4444-3333-222222222222",
    "agent_run_id": "88888888-7777-6666-5555-444444444444",
    "conversation_id": "c3d2e1f0-1234-5678-9abc-def012345678",
    "project_id": "b7e6c5a1-4321-4def-9876-543210abcdef",
    "duration_seconds": 3.42,
    "workflow_state": {
      "objective": "Produce an evidence-grounded Business Requirements Document",
      "section_progress": {
        "1.0 Executive Summary": "Completed"
      },
      "waiting_for_user": false,
      "pending_clarification": null
    }
  },
  "created_at": "2026-09-14T12:12:05.000000Z"
}
```

#### Direct Message Seeding (`role="assistant"`)
When `role="assistant"` is provided (e.g. for historical seeding or manual transcription), the endpoint persists the assistant message directly to the conversation history and returns HTTP `201 Created` without invoking the BRD agent.

#### cURL (Non-Streaming Execution)
```bash
curl -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678/messages?stream=false" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_access_token>" \
  -d '{
    "content": "Draft the Executive Summary section for the BRD based on the uploaded RFP.",
    "role": "user"
  }'
```

#### cURL (Streaming)
```bash
curl -N -X POST "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678/messages?stream=true" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_access_token>" \
  -d '{
    "content": "What are the key functional requirements?",
    "role": "user"
  }'
```

#### Example Frontend / Client Usage (JavaScript `fetch` + `ReadableStream`)
```javascript
async function sendMessageToAgent(projectId, conversationId, userText, onChunk, onComplete, onError, accessToken) {
  const response = await fetch(
    `/projects/${projectId}/conversations/${conversationId}/messages?stream=true`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify({ content: userText, role: 'user' })
    }
  );

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep partial line

    let currentEvent = 'message';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.slice(6).trim();
        if (!dataStr) continue;
        const parsed = JSON.parse(dataStr);

        if (currentEvent === 'message' && parsed.content) {
          onChunk(parsed.content);
        } else if (currentEvent === 'done') {
          onComplete(parsed); // Persisted MessageResponse
        } else if (currentEvent === 'error') {
          onError(parsed.error);
        }
      }
    }
  }
}
```

#### Postman
- **Method**: `POST`
- **URL**: `http://localhost:8000/projects/:project_id/conversations/:conversation_id/messages?stream=true`
- **Params**:
  - `project_id`: `b7e6c5a1-4321-4def-9876-543210abcdef`
  - `conversation_id`: `c3d2e1f0-1234-5678-9abc-def012345678`
  - `stream`: `true`
- **Headers**:
  - `Content-Type`: `application/json`
- **Body** (`raw` - `JSON`):
  ```json
  {
    "content": "Draft the Executive Summary section for the BRD based on the uploaded RFP.",
    "role": "user"
  }
  ```
- **Authentication**: Bearer Token (`{{accessToken}}`)
- *Note: Postman will stream the SSE chunks in real time under the Response view.*

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
- **Required Headers**: `Authorization: Bearer <access_token>`
- **Request Body**: None
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` and `conversation_id`. Project must belong to caller.

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
curl -X GET "http://localhost:8000/projects/b7e6c5a1-4321-4def-9876-543210abcdef/conversations/c3d2e1f0-1234-5678-9abc-def012345678/messages?limit=50&offset=0" \
  -H "Authorization: Bearer <your_access_token>"
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Knowledge Retrieval

The Knowledge Retrieval API executes the complete multi-stage RAG retrieval pipeline without invoking the LLM synthesis or conversation persistence layers. It is ideal for semantic search, search inspection, and debugging relevance ranking.

Knowledge retrieval requires Bearer token authentication and verifies that the target `project_id` belongs to the authenticated user. Operations on projects belonging to other users return HTTP `404 Not Found`.

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
- **Required Headers**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <access_token>`
- **Authentication**: Bearer Token (Required)
- **Prerequisites**: Obtain `project_id` from [Create Project](#create-project). Ensure source documents have been indexed into the project. Project must belong to caller.

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
  -H "Authorization: Bearer <your_access_token>" \
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
- **Authentication**: Bearer Token (`{{accessToken}}`)

---

## Missing, Broken, or Inconsistent APIs

The following observations, gaps, and design inconsistencies were identified during this comprehensive architectural inspection:

### 1. [RESOLVED] Authentication MVP & Project Tenancy Implemented
- **Resolution**: Implemented comprehensive email + password authentication (`/auth/signup`, `/auth/login`, `/auth/refresh`, `/auth/logout`, `/auth/me`), Argon2id password hashing, 15-minute JWT access tokens, 32-byte rotating opaque refresh tokens with reuse detection, and sliding window login rate limiting (5 attempts / 5 mins).
- **Ownership Enforcement**: Projects are strictly owned by Users (`user_id`). All project and sub-resource operations (`/projects`, `/sources`, `/conversations`, `/retrieval`) require a valid Bearer token and enforce tenancy isolation.

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
