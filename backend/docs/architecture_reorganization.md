# Backend Architectural Reorganization: LLM, Core & Observability

## 1. Executive Summary & Purpose

This document details the architectural boundaries established during the repository-wide reorganization of the backend prior to frontend integration.

Misplaced responsibilities and inverted dependency couplings have been resolved:
- **`src/llm/`** is established as a first-class root-level capability, decoupling LLM provider abstraction and model inference from the RAG generation pipeline.
- **`src/core/`** serves as the application foundation, owning centralized configuration (`Settings`, `get_settings`) and shared primitives.
- **`src/observability/`** serves as the cross-cutting diagnostics layer, owning application logging setup (`setup_logging`, `get_logger`).
- **`src/app/`** is strictly the delivery and deployment layer (FastAPI), eliminating lower-layer dependencies on application internals.

---

## 2. Root-Level Architecture

```
backend/src/
│
├── core/                         # Shared foundational primitives & configuration
│   ├── __init__.py               # Exports: Settings, get_settings
│   └── config.py                 # Single source of truth for app/env settings
│
├── observability/                # Telemetry, logging & monitoring
│   ├── __init__.py               # Exports: setup_logging, get_logger, LOG_FORMAT, DATE_FORMAT
│   └── logging.py                # Idempotent logging configuration & logger provider
│
├── llm/                          # Platform-wide LLM capability
│   ├── __init__.py               # Canonical LLM interface exports
│   ├── interface.py              # BaseLLMInterface abstract contract
│   ├── models.py                 # Normalized domain models: LLMResult, LLMStreamEvent, LLMUsage
│   ├── config.py                 # Typed LLMConfig (derives from core.config.Settings)
│   ├── service.py                # LLMService, generate_async, generate_stream_async
│   └── adapters/
│       ├── __init__.py           # Exports: OpenAICompatibleLLMAdapter
│       └── openai.py             # AsyncOpenAI SDK adapter with bounded backoff
│
├── rag/                          # Project-isolated RAG subsystem
│   ├── acquisition/              # Raw document acquisition
│   ├── ingestion/                # Ingestion validation & metadata assignment
│   ├── parsing/                  # Docx, Markdown, TXT parsers
│   ├── cleaning/                 # Text normalization & rule-based cleaning
│   ├── normalization/            # Unicode & line ending normalization
│   ├── chunking/                 # Content-aware deterministic chunk splitting
│   ├── metadata_enrichment/      # Metadata enrichment rules
│   ├── contextual_enrichment/    # Structured context generation
│   ├── embeddings/               # Voyage embedding client & Redis cache
│   ├── indexing/                 # Dual vector + sparse indexing orchestrator
│   ├── retrieval/                # Multi-stage retrieval (Dense, Sparse, RRF, Rerank, Hydration, Assembly)
│   └── generation/               # RAG response generation workflow
│       ├── formatting/           # Markdown context formatting
│       ├── prompt/               # System instruction & prompt construction
│       ├── postprocessing/       # Normalization & JSON extraction
│       └── evaluation/           # Quality gate groundedness & safety evaluator
│
├── db/                           # PostgreSQL engine & session management
├── models/                       # SQLAlchemy declarative entity models
├── schemas/                      # Pydantic request/response validation schemas
├── services/                     # Application & business orchestration services
├── storage/                      # Supabase object storage & Qdrant vector storage
├── exceptions/                   # Domain exception hierarchy (including exceptions/llm.py)
│
└── app/                          # FastAPI delivery layer
    ├── __init__.py
    └── main.py                   # App factory, lifespan, global exception handlers, health routes
```

---

## 3. Subsystem Ownership & Boundaries

### 3.1 LLM Subsystem (`src/llm/`)
- **Why it is root-level**: LLM inference is not a RAG-only capability. Autonomous Agents, Tools, evaluation routines, and future multi-agent workflows require direct access to language models without depending on retrieval context or prompt construction.
- **What it owns**:
  - `BaseLLMInterface`: Abstract asynchronous contract for model inference.
  - `OpenAICompatibleLLMAdapter`: Provider implementation utilizing `AsyncOpenAI` client with bounded retries and exponential backoff.
  - `LLMResult`, `LLMStreamEvent`, `LLMUsage`: Normalized application-level outputs with token usage and timing metadata.
  - `LLMConfig`: Domain configuration resolving default parameters, credentials, timeouts, and overrides from `core.config.Settings`.
  - `LLMService`: Singleton orchestrator with convenience entry points `generate_async` and `generate_stream_async`.
- **What it MUST NOT own**: RAG retrieval, chunk fetching, prompt formatting, context assembly, groundedness evaluation.
- **Invariants**: `llm` does NOT import `rag`, does NOT import `app`, and does NOT depend on `fastapi`.

### 3.2 RAG Generation (`src/rag/generation/`)
- **What it owns**:
  - `formatting/`: Formatting retrieved chunks into deterministic context blocks.
  - `prompt/`: Combining system instructions, formatted context, and user queries into structured prompt representations (`ConstructedPrompt`).
  - `postprocessing/`: Cleansing raw model completions, whitespace/line ending normalization, and structured output parsing.
  - `evaluation/`: Groundedness and safety evaluation quality gate verifying factual consistency against retrieved chunks.
- **Relationship to LLM**:
  ```
  RAG Generation (evaluation/service.py, etc.)
            │
            ▼ (consumes)
           LLM (BaseLLMInterface, LLMService)
  ```
  RAG owns the *workflow*. LLM owns the *model execution capability*.

### 3.3 Core Subsystem (`src/core/`)
- **What it owns**: Foundational application-wide configuration (`Settings`, `get_settings`).
- **Single Source of Truth**: All environment variables and `.env` properties are read and cached once via `get_settings()`. Domain configurations (e.g. `LLMConfig`, `RetrievalConfig`) derive their typed settings from this central instance.
- **Invariants**: `core` does NOT import domain layers (`rag`, `llm`, `storage`, `db`, `services`, `app`).

### 3.4 Observability Subsystem (`src/observability/`)
- **What it owns**: Centralized logging configuration (`setup_logging`, `get_logger`, `LOG_FORMAT`, `DATE_FORMAT`). Ensures idempotency by replacing root logger handlers to avoid duplicate log entries.
- **Invariants**: `observability` depends only on Python standard libraries and `core.config.get_settings` for `LOG_LEVEL`. It does NOT import `rag`, `llm`, or `app`.

---

## 4. Dependency Direction

The clean dependency graph strictly prevents inverted couplings:

```
                      ┌───────────────┐
                      │  src/core     │ (Foundational Config)
                      └───────┬───────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
     │src/observabi- │ │ src/storage  │ │    src/db    │
     │     lity      │ └──────┬───────┘ └──────┬───────┘
     └───────┬───────┘        │                │
             │                ▼                ▼
             │         ┌──────────────┐ ┌──────────────┐
             │         │   src/llm    │ │  src/models  │
             │         └──────┬───────┘ └──────┬───────┘
             │                │                │
             │                ▼                ▼
             │         ┌──────────────┐ ┌──────────────┐
             │         │   src/rag    │ │ src/schemas  │
             │         └──────┬───────┘ └──────┬───────┘
             │                │                │
             ▼                ▼                ▼
      ┌────────────────────────────────────────────────┐
      │                  src/services                  │
      └───────────────────────┬────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │  src/api  │
                        └─────┬─────┘
                              │
                              ▼
                        ┌───────────┐
                        │  src/app  │ (Delivery Layer)
                        └───────────┘
```

Lower-level reusable components (`rag`, `storage`, `db`, `services`) no longer depend on the delivery layer (`app`) to obtain settings or loggers.

---

## 5. Future Agent Compatibility

The promotion of `src/llm/` to root level directly unblocks future Agent architectures:

```
                 ┌──────────────┐
                 │   src/llm    │
                 └──────▲───────┘
                        │
             ┌──────────┴──────────┐
             │                     │
      ┌──────┴──────┐       ┌──────┴──────┐
      │   src/rag   │       │ src/agents  │ (Future Agent workflows)
      └─────────────┘       └─────────────┘
```

Future autonomous agents will import `BaseLLMInterface` and `LLMService` directly from `llm` without dragging in RAG retrieval dependencies.

---

## 6. Architectural Decision Records & Open Decisions

### Finalized
1. **LLM as Root Subsystem**: Canonical location established at `src/llm/`. Old path `src/rag/generation/llm/` completely removed.
2. **Core Foundation**: `Settings` and `get_settings` established at `src/core/config.py`. Old path `src/app/core/` completely removed.
3. **Observability Ownership**: Logging setup and utilities established at `src/observability/logging.py`.
4. **Exception Hierarchy**: Standalone `src/exceptions/llm.py` established.

### Open Decisions & Future Possibilities
1. **Query Transformation LLM Client Unification**:
   - `src/rag/retrieval/transformation/provider.py` contains a lightweight `BaseLLMClient` and `OpenAICompatibleLLMClient` specifically tailored for query rewriting.
   - *Current Decision*: Left intact during this reorganization to prevent regressions in query transformation contracts.
   - *Future Possibility*: Evaluate refactoring query transformation to optionally consume `src/llm/` as an adapter provider during retrieval optimization.
2. **Distributed Tracing & Metrics**:
   - `src/observability/` currently implements centralized standard library logging with correlation/formatting support.
   - *Future Possibility*: When OpenTelemetry or distributed tracing is formally specified, it will live under `src/observability/` without altering existing logging contracts.
