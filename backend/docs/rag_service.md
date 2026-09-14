# RAG Service / Retrieval Orchestration Architecture

## 1. Overview & Architectural Role

The **RAG Service / Retrieval Orchestration Layer** connects the modular, low-level retrieval components into a single, cohesive, application-facing retrieval workflow.

Operating as the boundary between the data layer and future generative workflows, the service receives a user query scoped to a specific project and returns a rich, structured `RetrievalResult` representing the most relevant, hydrated, and formatted knowledge from indexed project documents.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Application Layer                               │
│         (FastAPI: POST /projects/{project_id}/retrieval or Service Call)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RAG Service / Retrieval Orchestrator                   │
│                                                                             │
│   1. Query Preprocessing       ──▶ Clean & validate query                   │
│   2. Query Transformation      ──▶ Optional LLM rewrite / expand            │
│   3. Query Embedding           ──▶ Voyage AI dense embedding (voyage-4)     │
│   4. Dual Retrieval (Parallel) ──▶ Dense (Qdrant) + Sparse (BM25/Hash)      │
│   5. Hybrid Fusion (RRF)       ──▶ Reciprocal Rank Fusion                   │
│   6. Metadata Filtering        ──▶ Tenant boundary & metadata spec filters  │
│   7. Cross-Encoder Reranking   ──▶ Batch text resolve + Voyage reranker     │
│   8. Chunk Hydration           ──▶ Batched PostgreSQL ChunkRepository fetch │
│   9. Context Assembly          ──▶ Token budget truncation & enrichment     │
│  10. Relevance Check/Fallback  ──▶ Optional quality gate with retry         │
│  11. Context Formatting        ──▶ Markdown/XML formatted context block     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Authoritative Retrieval Result                       │
│    (Structured RetrievedChunks, FormattedContext, Latency & Audit Metadata)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Invariants

### 2.1 Strict Project & Tenant Isolation
- **Mandatory Tenant Scoping**: Every retrieval operation is strictly isolated by `project_id`.
- **Zero Cross-Tenant Leakage**:
  - Dense search applies a Qdrant filter constraint `project_id == target_project_id`.
  - Sparse search enforces the SQL clause `project_id = target_project_id`.
  - Metadata filtering re-verifies candidate `project_id`.
  - Chunk hydration queries PostgreSQL with `WHERE project_id = :project_id AND chunk_id IN (...)`.
  - Context assembly discards any cross-tenant chunk with a warning.
- **Fail-Fast Boundary Validation**: Queries containing empty, malformed, or mismatched project identifiers raise `ProjectBoundaryViolationError` immediately before performing any external or database operations.

### 2.2 Short Transactions & DB Session Isolation
- PostgreSQL transactions are isolated strictly to read-only chunk hydration queries.
- Database connections are never held open during external HTTP calls (Voyage AI embedding, Qdrant vector search, cross-encoder reranking, or LLM relevance checking).

### 2.3 Dual Storage Separation
- **Vector Index Store (Qdrant)**: Stores dense vector embeddings, sparse representations, and minimal routing payloads (`chunk_id`, `document_id`, `version_id`, `project_id`, `chunk_index`).
- **Relational Content Store (PostgreSQL `chunks` table)**: Authoritative store for verbatim chunk content, contextual summaries, section hierarchy paths, and structural metadata.

### 2.4 Document Versioning Awareness
- Only chunks belonging to document versions in `READY` status indexed into PostgreSQL and Qdrant are retrievable.
- Historical or superseded document version points are pruned upon newer version indexing.
- Retrieved chunks preserve `document_version_id`, ensuring full traceability to physical storage artifacts.

---

## 3. Retrieval Pipeline Sequence & Stage Details

### Stage 1: Query Preprocessing (`rag.retrieval.preprocessing`)
- **Action**: Deterministically normalizes whitespace, trims boundary controls, validates maximum character length (default: 2,000 chars), and enforces Unicode normalization (NFC).
- **Error Handling**: Raises `InvalidQueryError` (or subclasses `EmptyQueryError`, `QueryTooLongError`) on invalid input.

### Stage 2: Query Transformation (`rag.retrieval.transformation`)
- **Action**: When enabled, applies adaptive query transformations (e.g., query rewriting, expansion, sub-query decomposition) to align conversational questions with indexing terminology while preserving exact identifiers and domain acronyms.
- **Failure Resilience**: If the query transformation provider fails or times out, the service falls back gracefully to the preprocessed original query without interrupting the retrieval workflow.

### Stage 3: Query Embedding (`rag.embedding`)
- **Action**: Converts the effective retrieval query into a high-dimensional dense vector using Voyage AI (`voyage-4`, 1024 dimensions) with `input_type="query"`.

### Stage 4: Dual Branch Retrieval (`rag.retrieval.search`)
- **Concurrent Execution**: Dense vector search (via Qdrant) and sparse keyword search (via PostgreSQL full-text/BM25) run concurrently using `asyncio.gather`.
- **Branch Independence**: If sparse search is disabled via configuration, the pipeline seamlessly transitions to dense-only retrieval without overhead.

### Stage 5: Hybrid Fusion (`rag.retrieval.fusion`)
- **Action**: Combines dense and sparse ranked candidate lists using Reciprocal Rank Fusion (RRF, $k=60$).
- **Output**: Produces a deduplicated, unified ranking of `FusedCandidate` instances with combined scores.

### Stage 6: Metadata Filtering (`rag.retrieval.filtering`)
- **Action**: Applies tenant isolation predicates and user-defined operational filter specifications (e.g., document ID sets, file types, source tags, version boundaries) in memory against candidate metadata.

### Stage 7: Cross-Encoder Reranking (`rag.retrieval.reranking`)
- **Candidate Bounding**: Limits candidates to `rerank_candidate_limit` (default: 50).
- **Batch Text Resolution**: If candidates lack raw text (e.g. results directly from vector payloads), the service performs a single batched fetch from `ChunkRepository` (`WHERE chunk_id IN (...) AND project_id = :project_id`) before calling the cross-encoder.
- **Reranker Invocation**: Reranks documents against the query via the Voyage cross-encoder API and retains the top `rerank_result_limit` candidates.
- **Bypass Mode**: When reranking is disabled, candidates are sliced directly to `min(top_k, rerank_result_limit)`.

### Stage 8: Chunk Fetching / Hydration (`rag.retrieval.hydration`)
- **Action**: Fetches complete `ChunkModel` database entities for all selected candidates in a single batched SQL query (`SELECT ... WHERE project_id = :project_id AND chunk_id IN (...)`).
- **N+1 Prevention**: Zero single-row queries. If a candidate was already hydrated during Stage 7 pre-fetch, cached content is leveraged.
- **Output**: Produces ordered `HydratedCandidate` objects preserving verbatim text, contextual content, section path, and document version ID.

### Stage 9: Context Assembly (`rag.retrieval.assembly`)
- **Action**: Orders hydrated chunks, enforces maximum token budgets, resolves contextual enrichment vs. raw chunk content based on configuration, and computes context metadata.
- **Output**: Produces an `AssembledContext` container.

### Stage 10: Optional Relevance Check & Bounded Fallback (`rag.retrieval.relevance`)
- **Action**: When `enable_relevance_check=True`, evaluates whether the assembled context sufficiently addresses the user query using either heuristic checks or LLM evaluation.
- **Bounded Fallback**: If deemed non-relevant and `attempt < max_attempts`:
  1. Records attempt metadata and failure diagnostics.
  2. Modifies query strategy or falls back from transformed to original query.
  3. Re-executes the retrieval pipeline up to `max_attempts` (default: 2).
  4. Returns the best available assembled context if maximum attempts are exhausted.

### Stage 11: Context Formatting (`rag.retrieval.formatting`)
- **Action**: Formats the assembled context into a clean, LLM-ready context block (Markdown, XML, or Plaintext format) with citation anchors and document provenance markers.

---

## 4. Public Service Contract

### 4.1 Python Interface (`RAGService`)

```python
from rag.retrieval import get_rag_service, RetrievalConfig, RetrievalResult

rag_service = get_rag_service()

result: RetrievalResult = await rag_service.retrieve(
    project_id="proj-123",
    query="How does the distributed consensus mechanism handle network partitions?",
    config=RetrievalConfig(
        top_k=5,
        enable_sparse=True,
        enable_reranking=True,
        enable_relevance_check=False,
    ),
    session=db_session,  # Optional existing AsyncSession
)
```

### 4.2 Data Models (`rag.retrieval.models`)

- **`RetrievedChunk`**: Immutable presentation chunk containing `chunk_id`, `document_id`, `project_id`, `content`, `rank`, `score`, `document_version_id`, `heading`, `section_path`, and `metadata`.
- **`RetrievalExecutionMetadata`**: Execution audit containing total duration, attempt count, fallback trigger boolean, per-stage latencies, and detailed attempt records.
- **`RetrievalResult`**: The primary output envelope containing `project_id`, `original_query`, `retrieval_query`, `chunks`, `assembled_context`, `formatted_context`, and `execution_metadata`.

### 4.3 REST API (`POST /projects/{project_id}/retrieval`)

- **Path**: `/projects/{project_id}/retrieval`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "What is the deployment topology?",
    "top_k": 5,
    "dense_top_k": 10,
    "sparse_top_k": 10,
    "enable_transformation": true,
    "enable_sparse": true,
    "enable_reranking": true,
    "enable_relevance_check": false,
    "metadata_filters": {
      "file_type": "pdf"
    }
  }
  ```
- **Response**: `RetrievalResponseSchema` (HTTP 200 OK) containing structured chunks, formatted context, and execution audit metadata.

---

## 5. Scope Boundaries

To preserve clean modularity, the following capabilities are **explicitly excluded** from this retrieval orchestration layer and belong to downstream services:

| Component | Responsibility | Boundary |
|---|---|---|
| **Conversations & Messages** | Persisting user/assistant chat history | Chat / Session Service |
| **Prompt Construction** | Assembling system prompts, chat history, and context into LLM templates | Prompt Engineering Service |
| **LLM Answer Generation** | Streaming LLM completions from OpenAI / Anthropic / local models | Generation / Completion Service |
| **Groundedness Evaluation** | Verifying factual consistency between generated answer and retrieved context | Groundedness Checker |
| **Frontend Integration** | UI widgets, chat interfaces, streaming responses | Web Client Application |
| **Agent Tool Calling** | Tool calling loops, multi-step planning, autonomous execution | Agent Orchestration Engine |
