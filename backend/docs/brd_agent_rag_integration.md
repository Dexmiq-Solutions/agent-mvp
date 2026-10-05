# Phase 4.4 — BRD Agent RAG Capability Integration

## 1. Executive Summary

This document details the architectural integration connecting the **BRD Lead Agent** to the platform's existing **RAG retrieval capability** through the **DeepAgents runtime harness**.

The implementation preserves the single source of truth for knowledge retrieval:
- **No duplicate RAG implementations**: Reuses `search_project_knowledge` and `RAGService`.
- **Preserved architectural boundaries**: The BRD Lead Agent performs domain reasoning and decides *when* project knowledge is required; the RAG subsystem handles retrieval; DeepAgents provides the execution harness.
- **Strict project isolation**: The application runtime controls the `project_id` via `AgentContext`. The Agent cannot independently select, override, or alter project boundaries.

---

## 2. Architecture & Data Flow

```text
                                  User Request
                                       │
                                       ▼
                              BRD Lead Agent
                                       │
                                       ▼
                              DeepAgents Harness
                                       │
                                       ▼
                            search_project_knowledge
                                       │
                                       ▼
                              Existing RAG Service
                                       │
                                       ▼
                          Existing Retrieval Pipeline
                         (Dense + Sparse + RRF + Rerank)
                                       │
                                       ▼
                                RetrievalResult
                                       │
                                       ▼
                              DeepAgents Harness
                                       │
                                       ▼
                              BRD Lead Agent
```

### Component Responsibilities

| Component | Responsibility | Boundary Rule |
| :--- | :--- | :--- |
| **BRD Lead Agent** | Domain reasoning, BRD objective progression, deciding when project knowledge is needed. | Does not directly query vector stores, databases, or retrieval internals. |
| **DeepAgents Harness** | Graph-based state machine and tool orchestration. | Executes bound tools via LangChain/DeepAgents runtime. |
| **search_project_knowledge** | Canonical Agent/RAG boundary tool. | Resolves `project_id` strictly from application runtime context (`AgentContext`). Model supplies only `query`. |
| **RAGService** | Orchestrates query preprocessing, dense vector search, sparse BM25 search, reciprocal rank fusion (RRF), cross-encoder reranking, and context hydration. | Single source of truth for knowledge retrieval across all agents. |
| **RetrievalResult** | Authoritative structured retrieval outcome containing chunks, scores, and formatted context text. | Returned to agent without modification or custom encapsulation. |

---

## 3. RAG Decision Framework

The BRD Lead Agent is guided by authoritative behavioral instructions in `system_instruction.md`:

```text
BRD Lead Agent:
"What information do I need for this objective?"
        │
        ├─► Already in context / general question ──► Proceed directly without RAG
        │
        └─► Project-specific information needed ──► search_project_knowledge(query)
                                                             │
                                                             ▼
                                                    RAG: "Here is the project evidence."
                                                             │
                                                             ▼
                                                    BRD Lead Agent: Continue reasoning
```

### Decision Criteria:
1. **When to Retrieve**:
   - The current objective requires project-specific facts, architectural details, constraints, stakeholder requirements, or domain rules that are not already present in context or conversation history.
2. **When to Proceed Without Retrieval**:
   - The required information is already available in the prompt or conversation history.
   - The request asks for general business analysis methodology, terminology, or document structuring.
   - Avoid rigid "every turn triggers RAG" patterns.
3. **Query Formulation**:
   - Focused, natural language queries representing the specific information need (e.g., `"What authentication mechanism is used?"`).
   - The Agent does not supply technical retrieval parameters, filters, or database queries.

---

## 4. Strict Project Isolation

Project isolation is inviolable:
- `AgentContext(project_id=...)` is supplied by the application session.
- Context-local storage (`contextvars`) maintains `_current_agent_context`.
- `search_project_knowledge` inspects `get_current_agent_context()`.
- If `project_id` is missing or blank, retrieval is immediately rejected with:
  `[RETRIEVAL_ERROR] Project boundary violation: No active project context.`
- The tool's input schema (`SearchProjectKnowledgeInput`) exposes only `query: str`, preventing model prompt-injection of project identifiers.
- Project A can never access or retrieve knowledge belonging to Project B.

---

## 5. Verification Matrix

The integration is verified by 16 targeted tests in `tests/test_brd_agent_rag_integration.py` plus the complete suite in `tests/test_brd_lead_agent.py`:

| Verification Requirement | Test Name | Status |
| :--- | :--- | :--- |
| **1. Access to existing RAG tool** | `test_brd_lead_agent_has_rag_capability_by_default` | Passed |
| **2. DeepAgents tool execution (sync)** | `test_brd_lead_agent_executes_rag_through_deepagents_sync` | Passed |
| **3. DeepAgents tool execution (async)** | `test_brd_lead_agent_executes_rag_through_deepagents_async` | Passed |
| **4. Strict project isolation** | `test_project_isolation_tenant_separation` | Passed |
| **5. Missing context rejection** | `test_missing_project_context_rejects_retrieval` | Passed |
| **6. Model cannot modify project ID** | `test_model_cannot_select_or_modify_project_identifier` | Passed |
| **7. Successful retrieval with chunks** | `test_brd_lead_agent_executes_rag_through_deepagents_sync` | Passed |
| **8. Empty retrieval handling** | `test_brd_lead_agent_handles_no_evidence_outcome` | Passed |
| **9. RAG service error handling** | `test_brd_lead_agent_handles_retrieval_service_error` | Passed |
| **10. Selective retrieval (no RAG when sufficient)** | `test_brd_lead_agent_proceeds_without_rag_when_context_sufficient` | Passed |
| **11. Injected RAG service support** | `test_brd_lead_agent_with_injected_rag_service` | Passed |
| **12. Factory function support** | `test_create_brd_lead_agent_factory_equips_rag` | Passed |
| **13. Explicit RAG disable toggle** | `test_brd_lead_agent_rag_can_be_explicitly_disabled` | Passed |
| **14. Runtime Agent RAG unaffected** | `test_existing_runtime_agent_rag_behavior_unaffected` | Passed |
| **15. No direct infrastructure access** | `test_brd_agent_has_no_direct_infrastructure_dependencies` | Passed |
| **16. System instruction RAG guidance** | `test_system_instruction_contains_knowledge_retrieval_guidance` | Passed |
