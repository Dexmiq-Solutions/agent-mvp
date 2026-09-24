# RAG & Agent Architectural Boundary (Phase 1 — Generation Layer Retired)

## 1. Architectural Mission & Overview

This document defines the architectural boundary established in **Phase 1 of the BRD Agent transition**.

The Dexmiq platform previously coupled document retrieval with an internal LLM generation layer:
```text
OLD STANDALONE RAG GENERATION (RETIRED)

User Query
    ↓
RAG Retrieval (Dense + Sparse + RRF + Reranking + Hydration + Assembly)
    ↓
Context Formatting
    ↓
Prompt Construction (System instructions + context + user query)
    ↓
LLM Inference
    ↓
Post-processing & Normalization
    ↓
Groundedness & Safety Evaluation Quality Gate
    ↓
Regeneration / Quality Feedback Loop
    ↓
Final Response
```

In the target architecture, the **BRD Agent** becomes the intelligent workflow layer. RAG is no longer a standalone question-answering system that generates answers or evaluates them.

The new architecture establishes a clean, decoupled boundary:

```text
TARGET ARCHITECTURE (PHASE 1 ESTABLISHED)

Query / Retrieval Request
    ↓
RAG Retrieval Pipeline (Document ingestion, indexing, search, hydration, context assembly, telemetry)
    ↓
RetrievalResult (Evidence Chunks + Metadata + Scores + Formatted Context + Execution Telemetry)
    ↓
Agent Boundary (Future Agent Tool)
    ↓
BRD Agent (Reasoning, Evidence Evaluation, BRD Synthesis, Validation, Clarification, Feedback Loops)
```

> **Core Architectural Principle**:
> **RAG provides knowledge. The Agent decides what to do with that knowledge.**

---

## 2. Target Responsibility Boundary

| Responsibility | Owning Subsystem | Status in Phase 1 |
| :--- | :--- | :--- |
| **Document Ingestion & Parsing** | RAG | Active (`DocumentProcessingService`) |
| **Cleaning & Normalization** | RAG | Active (`DocumentProcessingService`) |
| **Chunking & Metadata Enrichment** | RAG | Active (`DocumentProcessingService`) |
| **Dense & Sparse Indexing** | RAG | Active (Qdrant + BM25/Sparse) |
| **Query Preprocessing & Transformation** | RAG | Active (`QueryPreprocessor`, `QueryTransformer`) |
| **Dense & Sparse Vector Retrieval** | RAG | Active (`VectorSearcher`, `SparseSearcher`) |
| **Reciprocal Rank Fusion (RRF)** | RAG | Active (`ReciprocalRankFusion`) |
| **Cross-Encoder Reranking** | RAG | Active (`Reranker`) |
| **Database Chunk Hydration** | RAG | Active (`ChunkHydrator`) |
| **Context Assembly** | RAG | Active (`ContextAssembler`) |
| **Context Formatting** | RAG | Active (`rag.retrieval.formatting`, Stage 11) |
| **Retrieval Telemetry & Timing** | RAG | Active (`RetrievalResult.execution_metadata`) |
| **Evidence Reasoning & Evaluation** | **Agent** | Deferred to Phase 2+ |
| **Gap Detection & Contradiction Handling** | **Agent** | Deferred to Phase 2+ |
| **Multi-turn Dialogue & Response Synthesis** | **Agent** | Deferred to Phase 2+ |
| **BRD Section & Artifact Validation** | **Agent** | Deferred to Phase 2+ |
| **Regeneration & Quality Rework Loops** | **Agent** | Deferred to Phase 2+ |
| **User Clarification (HITL)** | **Agent** | Deferred to Phase 2+ |

---

## 3. The Retrieval Boundary Contract (`RetrievalResult`)

The primary contract between RAG and the consuming application (and future Agent) is `RetrievalResult`, exposed by `RAGService.retrieve(...)`:

```python
result: RetrievalResult = await rag_service.retrieve(
    project_id=project_id,
    query=query,
    config=retrieval_config,
    session=db_session,
)
```

### RetrievalResult Structure
* `chunks: list[HydratedChunk]`: Full chunk payload with persistent text content, section headers, document metadata, source document references, and character/token spans.
* `scores: list[float]`: Combined fusion / reranker relevance scores.
* `ranks: list[int]`: 1-based ordinal relevance rankings.
* `execution_metadata: RetrievalExecutionMetadata`: Granular stage-by-stage execution latency (dense, sparse, fusion, rerank, hydration, assembly, formatting), query transformation details, attempts, and fallback flags.
* `formatted_context: FormattedContext | None`: Structured, deterministic markdown representation assembled during Stage 11 of retrieval (`rag.retrieval.formatting`).
* `formatted_text: str`: Model-ready formatted string (or extracted directly from `formatted_context`) for consumption by downstream reasoning agents or LLM prompts.

---

## 4. Summary of Phase 1 Retirements & Refactoring

### A. Retired Components (Category A)
1. **`services.generation_service.GenerationService`**: The RAG-owned generation orchestrator that executed prompt construction, LLM inference, post-processing, groundedness evaluation, and regeneration loops.
2. **`rag.generation.prompt.service.PromptConstructionService`**: Answer prompt construction templates and formatting.
3. **`rag.generation.postprocessing.service.PostProcessingService`**: Response whitespace normalization, finish reason filtering, and JSON extraction.
4. **`rag.generation.evaluation.service.EvaluationService`**: RAG-owned groundedness and safety evaluation quality gates with regeneration triggers.
5. **`schemas/generation.py`**: Generation request/response schemas (`GenerationRequestSchema`, `GenerationResponseSchema`).
6. **`exceptions/generation.py`**: Generation exceptions (`GenerationError`, `PromptConstructionError`, `RegenerationExhaustedError`, etc.).
7. **`generate=true` API branch on `POST .../messages`**: The endpoint now purely persists conversation message turns.

### B. Relocated Components (Category C)
1. **Context Formatting (`rag.retrieval.formatting`)**:
   - Relocated from `rag/generation/formatting/` to `rag/retrieval/formatting/`.
   - Context formatting is Stage 11 of the RAG retrieval pipeline and feeds `RetrievalResult.formatted_context` / `RetrievalResult.formatted_text`.
   - New exceptions `ContextFormattingError` and `ContextFormattingValidationError` inherit from `RetrievalError` in `src/exceptions/retrieval.py`.

### C. Retained Shared Infrastructure (Category B)
1. **Generic LLM Abstraction (`src/llm/`)**:
   - `LLMService`, `BaseLLMInterface`, `OpenAICompatibleLLMAdapter`, and `LLMConfig` are retained.
   - Used by RAG for query transformation / expansion.
   - Serves as the generic OpenAI/OpenRouter foundation for the future Agent.
2. **Conversation & Message Persistence (`ConversationService`)**:
   - Unaffected message turn persistence in PostgreSQL/SQLite for multi-turn history.
3. **Application Configuration (`src/core/config.py`)**:
   - Generic `LLM_*` settings are retained; generation-specific (`PROMPT_CONSTRUCTION_*`, `POST_PROCESSING_*`, `EVALUATION_*`) settings are removed.

---

## 5. Next Steps (Phase 2 Roadmap)

1. **DeepAgents & OpenRouter Integration**: Establish the agent framework using the OpenAI-compatible adapter.
2. **RAG Agent Tool**: Expose `RAGService.retrieve(...)` as a first-class tool callable by the Agent.
3. **BRD Lead Agent & State Management**: Implement reasoning, evidence evaluation, and section generation within the Agent domain.
4. **Agent Validation & Feedback**: Port quality evaluation and regeneration loops into the Agent validation workflow.
