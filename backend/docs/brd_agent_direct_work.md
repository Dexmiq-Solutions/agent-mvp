# Phase 4.5 — BRD Agent Direct Work Capability

## 1. Executive Summary

This document describes the design and integration of the **Direct Work capability** for the **BRD Lead Agent** in Phase 4.5.

The BRD Lead Agent now possesses two operational actions:
1. **Direct Work**: When the information and capabilities required to accomplish a task are already available in the current context, the BRD Lead Agent performs the reasoning directly itself and returns the result to continue the workflow.
2. **Knowledge Retrieval (RAG)**: When external project knowledge (facts, architectural rules, constraints) is required and not present in context, the Agent invokes `search_project_knowledge`.

```text
                                Current Task / Objective
                                           │
                                           ▼
                                    BRD Lead Agent
                                           │
                                Determine Required Action
                                           │
                              Is the required information
                                  already available?
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                       Yes                                    No
                        │                                     │
                 Can Lead Agent                         Invoke RAG
               handle task itself?              (search_project_knowledge)
                        │                                     │
                        ▼                                     ▼
                   Direct Work                        Retrieve Evidence
                        │                                     │
               Agent Performs Work                    Synthesize Evidence
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           ▼
                                   Continue Workflow
```

### Core Tenets of Direct Work
- **Not a Tool**: Direct Work is not an external tool (`direct_work()`, `execute_task()`, etc.). The Agent itself is the cognitive executor.
- **Not Another Agent**: Direct Work does not spawn child agents or sub-agents.
- **Not Defined Only by Simplicity**: Direct Work is appropriate for tasks requiring substantial analytical reasoning (e.g. deduplicating requirements, identifying contradictions, trade-off analysis) as long as the necessary facts/inputs are available in context.
- **DeepAgents Remains Execution Harness**: Execution continues through the unified DeepAgents state graph runnable.

---

## 2. Architecture & Execution Flow

```text
                  User Request / Task
                           │
                           ▼
                  BRD Lead Agent (execute / execute_async)
                           │
                  Normalize Request & State (AgentRunRequest, AgentContext)
                           │
                           ▼
                  DeepAgents Execution Harness
                           │
                  Agent Reasoning & Action Determination
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        [Direct Work]                  [RAG]
     No Tool Calls Made         Tool Call Issued
             │                  search_project_knowledge
             │                           │
             │                  RAGService.retrieve(project_id, query)
             │                           │
             │                  RetrievalResult Returned
             │                           │
             │                  Model Synthesizes Evidence
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  AIMessage Produced
                           │
                  AgentRunResponse (is_direct_work, state, output_text)
                           │
                  Continue BRD Workflow
```

### Component Responsibilities

| Component | Responsibility in Direct Work | Boundary Rule |
| :--- | :--- | :--- |
| **BRD Lead Agent** | Receives task, maintains working state (`BRDAgentState`), adopts task context (`current_task`), and yields normalized response. | Does not spawn sub-agents or create duplicate execution engines. |
| **DeepAgents Harness** | Executes the agent state graph with bound model and tools. | Returns directly when model produces `AIMessage` without tool calls. |
| **System Instruction** | Defines authoritative behavioral principles, action determination criteria, and evidence grounding. | Authoritative markdown artifact (`system_instruction.md`). |
| **Agent State (`BRDAgentState`)** | Maintains working context, active section progress, collected evidence, unresolved gaps, and `current_task`. | Remains single source of truth for runtime progress. |
| **RAG Tool (`search_project_knowledge`)** | Remains available for knowledge retrieval when external facts are missing. | Bypassed when information is already present. |

---

## 3. Direct Work vs RAG vs (Future) Delegation

| Dimension | Direct Work (Phase 4.5) | RAG Capability (Phase 4.4) | Future Delegation (Out of Scope) |
| :--- | :--- | :--- | :--- |
| **Trigger** | Required information is already available in context. | Required project-specific knowledge is missing from context. | Task requires independent, bounded, isolated execution. |
| **Executor** | BRD Lead Agent itself. | RAG Subsystem via `search_project_knowledge`. | Temporary, scoped sub-agents. |
| **Mechanism** | Internal model reasoning and generation. | Tool invocation -> Vector/Keyword search -> Evidence synthesis. | Agent decomposition and worker execution. |
| **Tool Calls** | None (`tool_calls == []`). | One or more (`search_project_knowledge`). | Delegation tools / subagent invocations. |
| **Complexity Range** | Simple rewrites to complex analytical deductions over context. | Knowledge lookups, constraint verification, architectural queries. | Large-scale multi-part document authoring. |

---

## 4. Examples of Direct Work

### Example 1: Requirement Rewrite
- **Input**: `"Rewrite this requirement clearly: Users must be able to reset their password."`
- **Context Evaluation**: All required information is provided in the prompt.
- **Action**: Direct Work. No RAG tool invocation.
- **Result**: Self-contained, unambiguous functional requirement statement formatted per BRD conventions.

### Example 2: Summarizing Existing Context
- **Input**: `"Summarize the requirements listed above: 1. User login with MFA. 2. Role-based access control. 3. Audit trail for administrative actions."`
- **Context Evaluation**: The three requirements are explicitly present in the input.
- **Action**: Direct Work. No RAG tool invocation.
- **Result**: Concise executive summary synthesizing the security requirements.

### Example 3: Substantial Analytical Reasoning Over Provided Context
- **Input**: `"Identify duplicate and conflicting requirements in the following list: REQ-1 (8-char password), REQ-2 (15-min timeout), REQ-3 (8-char minimum password), REQ-4 (infinite session duration)."`
- **Context Evaluation**: The task involves complex semantic comparison, contradiction detection, and deduplication. All information is present in the prompt.
- **Action**: Direct Work. High complexity does not automatically trigger delegation or retrieval.
- **Result**: Clear analytical breakdown identifying the duplicate pair (REQ-1/REQ-3) and conflicting constraint (REQ-2/REQ-4).

---

## 5. State Lifecycle & Workflow Continuity

Direct Work integrates seamlessly into the established state lifecycle:

1. **State Preservation**: The Agent's working context (`BRDAgentState`), including `objective`, `current_section`, `section_progress`, `evidence`, and `unresolved_information`, remains intact during Direct Work.
2. **Current Task Tracking**: Execution accepts an optional `current_task` parameter, storing the immediate focus in `state.current_task` and returning it in `response.state`.
3. **Multi-Turn Continuity**: Direct Work can follow RAG retrieval (e.g., Turn 1 retrieves architecture; Turn 2 performs Direct Work to draft a requirement from that architecture) or precede further actions without state loss.

---

## 6. Verification Matrix

The Direct Work capability is verified by **12 dedicated tests** in `tests/test_brd_agent_direct_work.py`, along with the existing test suites in `tests/test_brd_lead_agent.py` and `tests/test_brd_agent_rag_integration.py` (total **46 tests passed**):

| Verification Requirement | Test Name | Result |
| :--- | :--- | :--- |
| **1. Direct Work: Requirement rewrite** | `test_direct_work_rewrite_requirement_example_1` | Passed |
| **2. Direct Work: Context summarization** | `test_direct_work_summarize_existing_context_example_2` | Passed |
| **3. Direct Work: Substantial reasoning / deduplication** | `test_direct_work_substantial_reasoning_deduplicate_and_conflict_example_3` | Passed |
| **4. No separate direct work tool** | `test_direct_work_does_not_create_or_require_separate_tool` | Passed |
| **5. Selective branch choice: Direct Work vs RAG** | `test_agent_selects_direct_work_when_info_present_vs_rag_when_info_missing` | Passed |
| **6. State preservation & current_task flow** | `test_direct_work_preserves_agent_state_and_updates_current_task` | Passed |
| **7. State set_current_task method** | `test_brd_agent_state_set_current_task_method` | Passed |
| **8. Structured AgentRunRequest support** | `test_direct_work_with_agent_run_request_payload` | Passed |
| **9. DeepAgents harness execution** | `test_direct_work_executes_through_deepagents_harness` | Passed |
| **10. Async execution support** | `test_direct_work_async_execution` | Passed |
| **11. Absence of delegation primitives** | `test_no_delegation_mechanisms_introduced` | Passed |
| **12. System instruction Direct Work guidance** | `test_system_instruction_contains_action_determination_and_direct_work` | Passed |
| **13. Existing Lead Agent tests (18 tests)** | `tests/test_brd_lead_agent.py` | All Passed |
| **14. Existing RAG integration tests (16 tests)** | `tests/test_brd_agent_rag_integration.py` | All Passed |
