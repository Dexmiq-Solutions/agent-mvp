# LLM Generation Integration Architecture

## 1. Overview & Purpose

The **LLM Generation Integration** layer provides the application-level coordination bridging Dexmiq's conversational boundary, project-isolated RAG retrieval pipeline, and LLM inference engine.

It connects the end-to-end flow:
```
Conversation
    ↓
User Message Turn
    ↓
RAG Service (Dense + Sparse + RRF + Reranking + Hydration + Assembly)
    ↓
Retrieved Context
    ↓
Context Formatting (Structured, deterministic markdown context)
    ↓
Prompt Construction (System instructions + context + original user query)
    ↓
LLM Interface (Provider-independent async inference with timeouts & retries)
    ↓
Model Response
    ↓
Post-processing (Whitespace & line ending normalization, finish reasons)
    ↓
Groundedness & Safety Evaluation Quality Gate (Bounded feedback regeneration)
    ↓
Assistant Message Persistence
    ↓
Conversation Turn Result
```

This layer establishes a reliable, non-agentic RAG-backed response generation pipeline ensuring that every generated response is strictly grounded in project-scoped knowledge, meets all safety criteria, and is persisted chronologically within tenant conversations.

---

## 2. Non-Negotiable Architectural Rule: Single Source of Truth

In strict accordance with system architecture rules, the LLM Generation Integration layer introduces **zero duplicate abstractions or configurations**:

| Responsibility | Authoritative Implementation Reused | Architectural Rationale |
| :--- | :--- | :--- |
| **LLM Configuration** | `app.core.config.Settings` & `rag.generation.llm.config.LLMConfig` | All model names, timeouts, retry limits, temperatures, and credentials derive strictly from application `Settings`. No parallel `GenerationConfig` or `ChatConfig`. |
| **LLM Inference** | `rag.generation.llm.service.LLMService` & `BaseLLMInterface` | OpenAI-compatible adapter encapsulates provider calls, payload mapping, and vendor-specific error translation. |
| **Infrastructure Retries** | `OpenAICompatibleLLMAdapter` backoff loop | Network timeouts and transient 5xx/429 errors are retried at the adapter layer. The generation layer never wraps this with a duplicate HTTP retry loop. |
| **RAG Retrieval** | `services.rag_service.RAGService` (`rag/retrieval/service.py`) | Orchestrates multi-stage retrieval across Qdrant, technical sparse encoding, and PostgreSQL hydration. |
| **Context Formatting** | `rag.generation.formatting.service.ContextFormattingService` | Pre-formats retrieved candidates into deterministic, model-readable context items. |
| **Prompt Construction** | `rag.generation.prompt.service.PromptConstructionService` | Composes system instruction, verbatim context, and original query with clean logical boundaries and regeneration feedback support. |
| **Post-processing** | `rag.generation.postprocessing.service.PostProcessingService` | Deterministic normalization of whitespace, line endings, finish reasons, and JSON parsing. |
| **Evaluation Quality Gate** | `rag.generation.evaluation.service.EvaluationService` | Impartial evaluator checking factual groundedness against context and safety compliance. |
| **Conversation & Messages** | `services.conversation_service.ConversationService` | Authoritative persistence for both user and assistant message models in PostgreSQL/SQLite. |
| **Project Boundary Isolation** | `ConversationService.get_conversation`, `ProjectService.get_project`, `RAGService.retrieve` | Asserts tenant ownership across conversation lookups, vector filtering, and chunk hydration. |

---

## 3. End-to-End Generation Workflow

```
[User Request / Client API]
           │
           ▼
[POST /projects/{project_id}/conversations/{conversation_id}/messages?generate=true]
           │
           ▼
[ConversationService.get_conversation(project_id, conversation_id)]  ──► Validates Project Ownership
           │
           ▼
[ConversationService.create_message(role='user')]  ────────────────────► Persists User Message
           │
     (Session Commit)  ───► Short Transaction Window (Locks released before external calls)
           │
           ▼
[RAGService.retrieve(project_id, query)]  ────────────────────────────► Hybrid Retrieval + Hydration
           │
     (RetrievalResult)
           │
           ▼
  ┌──► [PromptConstructionService.construct()]  ──────────────────────► Composes System + Context + Query
  │        │
  │        ▼
  │    [LLMService.generate()]  ──────────────────────────────────────► Bounded Timeout & HTTP Retries
  │        │
  │        ▼
  │    [PostProcessingService.process()]  ────────────────────────────► Deterministic Normalization
  │        │
  │        ▼
  │    [EvaluationService.evaluate_async()]  ─────────────────────────► Impartial Quality Gate
  │        │
  │        ├─► [passed == False & attempts < max] ────────────────────► Propagate Feedback to Regeneration Loop
  │        │
  │        ├─► [passed == False & attempts >= max] ───────────────────► Raise RegenerationExhaustedError
  │        │                                                            (User msg preserved; NO fake assistant turn)
  │        ▼
  └─────── [passed == True]
           │
           ▼
[ConversationService.create_message(role='assistant', metadata=...)] ─► Persists Accepted Turn
           │
     (Session Commit)
           │
           ▼
[Return GenerationResult / MessageResponse]
```

---

## 4. State & Database Transaction Boundaries

To ensure database scalability and prevent transaction lock exhaustion:

1. **Short Write Window 1 (User Message)**:
   The user message is inserted into PostgreSQL and committed immediately. If downstream retrieval or LLM inference fails, the user turn remains safely recorded in conversation history.
2. **Transaction-Free Long-Running Window**:
   Database write locks are **never** held open during:
   - Vector similarity search (Qdrant)
   - Cross-encoder reranking
   - LLM generation (OpenAI/compatible)
   - Quality gate evaluation calls
3. **Short Write Window 2 (Assistant Message)**:
   Upon quality gate acceptance, the assistant message turn is persisted with rich execution metadata and committed in a short transaction window.
4. **No Fake Assistant Turns**:
   If generation fails (e.g. LLM timeout, quality gate exhaustion), the user message is retained, and no placeholder or hallucinated assistant message is persisted.

---

## 5. Bounded Regeneration vs. Infrastructure Retry

The system strictly decouples **infrastructure retries** from **semantic regeneration**:

- **Infrastructure Retries (Provider Layer)**:
  Handled by `OpenAICompatibleLLMAdapter` using exponential backoff (`LLM_MAX_RETRIES`, `LLM_RETRY_DELAY`, `LLM_RETRY_BACKOFF`) for transient network timeouts, HTTP 502/503/504 errors, and HTTP 429 rate limits.
- **Semantic Quality Gate Regeneration (Generation Layer)**:
  If the generated response contains ungrounded factual claims (`grounded=False`) or safety violations (`safe=False`), `EvaluationService` returns a structured critique (`reason`). The generation loop passes this critique into `PromptConstructionService` as `evaluation_feedback`, directing the model to regenerate its response adhering strictly to context.
  - Bounded by `Settings.EVALUATION_MAX_REGENERATION_ATTEMPTS` (default: 2 regeneration attempts).
  - If attempts are exhausted without passing, `RegenerationExhaustedError` is raised.

---

## 6. API Boundary & Backwards Compatibility

The generation flow integrates directly into the existing conversation API:

```http
POST /projects/{project_id}/conversations/{conversation_id}/messages
```

### Request Parameters:
- `project_id`: Mandatory tenant project identifier.
- `conversation_id`: Mandatory conversation identifier.
- `payload`: Standard `MessageCreate` schema (`role`, `content`, `metadata`).
- `generate`: Optional boolean query parameter (`default=False`).

### Behavior:
1. **`generate=False` (Default)**:
   Directly persists the message turn and returns `MessageResponse`. Preserves 100% backwards compatibility for existing CRUD tests and offline sync.
2. **`generate=True` & `role='user'`**:
   Executes the full RAG retrieval + LLM generation + Quality Gate pipeline, persists both the user turn and accepted assistant turn, and returns the assistant `MessageResponse` with metadata:
   ```json
   {
     "id": "msg_asst_uuid",
     "conversation_id": "conv_uuid",
     "role": "assistant",
     "content": "The return policy allows returns within 30 days with receipt.",
     "metadata": {
       "user_message_id": "msg_user_uuid",
       "finish_reason": "stop",
       "usage": { "input_tokens": 150, "output_tokens": 35, "total_tokens": 185 },
       "model": "gpt-4o",
       "retrieval": {
         "chunks_count": 3,
         "duration_ms": 42.5,
         "attempts_count": 1,
         "retrieval_query": "What is the return policy?"
       },
       "evaluation_passed": true,
       "generation_attempts": 1,
       "total_duration_ms": 1250.4
     },
     "created_at": "2026-09-14T10:00:00Z"
   }
   ```

---

## 7. Observability & Privacy Protection

All generation lifecycle events use the centralized logger (`app.core.logging.get_logger`):

- **Trace Identifiers**: Every log event includes `project_id`, `conversation_id`, and `user_message_id`.
- **Lifecycle Milestones**:
  - `Initiating RAG generation`: Logs query character length.
  - `RAG retrieval completed`: Logs chunk count and retrieval duration.
  - `Starting bounded generation regeneration attempt`: Logs attempt index and reason.
  - `Generation passed quality gate`: Logs grounded/safe status and duration.
  - `Generation workflow completed successfully`: Logs assistant message ID and total duration.
- **Privacy Enforcement**: Full user queries, raw prompt dumps, complete retrieved context, and full model completions are **never** logged to stdout/telemetry; only character lengths, chunk counts, and structured metadata are recorded.

---

## 8. Strict Exclusions & Deferred Work

The following functionalities are intentionally excluded from this layer:
- **Agent Orchestration**: No LangGraph, DeepAgents, autonomous planning, or reflection agents.
- **Tools & MCP**: No tool calling, Model Context Protocol, or external function execution.
- **Multi-Agent Coordination**: Single-turn project RAG response generation only.
- **Conversation Summarization**: Token-budgeted history compression is deferred.
- **Frontend / UI**: Backend application service and API boundary only.
