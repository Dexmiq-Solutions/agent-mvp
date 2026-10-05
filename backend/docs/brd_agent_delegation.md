# Phase 4.6 — BRD Agent Delegation & Temporary Sub-Agent Execution

## 1. Executive Summary

This document describes the design, integration, and verification of the **Delegation capability** for the **BRD Lead Agent** in Phase 4.6.

The BRD Lead Agent now possesses three operational execution capabilities:
1. **Direct Work**: When required information and analytical capabilities are present in context, the Lead Agent performs reasoning directly.
2. **Knowledge Retrieval (RAG)**: When external project knowledge (facts, rules, architecture) is missing from context, the Agent invokes `search_project_knowledge`.
3. **Delegation**: When an objective is complex, composite, or benefits from bounded decomposition, the Lead Agent decomposes the objective into dynamic bounded tasks, executes them via temporary task-scoped sub-agents, collects individual task results, and produces a unified Delegation Action Result.

All three execution paths converge onto a single conceptual boundary: the common **Action Result**.

```text
BRD Lead Agent
  ├── RAG
  ├── Direct Work
  └── Delegation
        ↓
   Task Decomposition
        ↓
   Task A → Temporary Sub-Agent
   Task B → Temporary Sub-Agent
   Task C → Temporary Sub-Agent
        ↓
   Collect Task Results
        ↓
   Delegation Action Result
```

Result Convergence:
```text
RAG
   ↓
Action Result

Direct Work
   ↓
Action Result

Delegation
   ↓
Task Results
   ↓
Delegation Action Result
   ↓
Action Result
```

Downstream workflow capabilities (such as future Evaluation) consume this uniform `ActionResult` boundary without needing branch-specific handling.

---

## 2. Core Architectural Tenets

- **Lead Agent Retains Workflow Ownership**: The BRD Lead Agent owns the overarching BRD objective, decomposes tasks, scopes sub-agent context, consumes task results, and determines next steps. Sub-agents are never workflow owners or permanent coordinators.
- **Temporary Sub-Agents**: Sub-agents are created on demand for the duration of a single bounded task and cease execution upon return. There are **no permanent specialist agents** (e.g. no persistent `ResearchAgent`, `AnalysisAgent`, `WriterAgent`, `ValidationAgent`).
- **DeepAgents Remains Execution Harness**: Execution continues through the DeepAgents state graph runnable (`create_runtime_agent` / `create_deep_agent`).
- **Bounded Task Definition**: Each delegated task is bounded with Objective, Relevant Input / Context, Scope, Constraints, and Expected Output.
- **Dynamic Task Count**: Supports dynamic decomposition (2, 3, 5, or more tasks) tailored to the objective rather than a rigid fixed count.
- **Strict Context & Project Isolation**: Sub-agents receive only the minimum context required for their task (not the entire agent state or history). Tenant `project_id` is inherited and inviolable.
- **Depth Limit & Tool Scoping**: Sub-agents receive only minimal required tools; recursive delegation is prohibited (maximum delegation depth = 1: Lead Agent → Sub-Agent).
- **Single Source of Truth**: All progress, tasks, results, and action results are maintained in `BRDAgentState`.

---

## 3. Component Architecture & Responsibilities

| Component | Responsibility in Delegation | Boundary Rule |
| :--- | :--- | :--- |
| **BRD Lead Agent** | Owns the overall BRD objective, triggers decomposition, delegates tasks, and consumes collected results. | Does not relinquish workflow control or transfer ownership to sub-agents. |
| **Task Decomposition (`decompose_objective`)** | Analyzes objective and produces dynamic list of bounded `DelegatedTask` objects. | Does not force a fixed number of tasks. |
| **Delegated Task (`DelegatedTask`)** | Bounded unit of work defining Objective, Relevant Context, Scope, Constraints, Expected Output. | Contains only task-specific context. |
| **Temporary Sub-Agent (`execute_subagent_task`)** | Executes a single `DelegatedTask` via a temporary DeepAgents state graph runnable. | Cannot delegate; max depth = 1; cannot choose or override `project_id`. |
| **Task Result (`TaskResult`)** | Attributable result recording `task_id`, `objective`, `content`, `execution_info`, and `success`. | In-memory execution artifact; no redundant DB persistence. |
| **Result Collection (`collect_task_results`)** | Consolidates individual task results into a coherent `DelegationResult` preserving task attribution. | Preserves failure details if tasks fail without corrupting state. |
| **Common Action Result (`ActionResult`)** | Unified result boundary converging Direct Work, RAG, and Delegation (`source=ActionSource.DELEGATION`). | Common boundary consumed by future downstream evaluation. |
| **Agent State (`BRDAgentState`)** | Maintains `delegated_tasks`, `task_results`, `delegation_result`, and `latest_action_result`. | Preserves state continuity across Lead Agent execution cycles. |

---

## 4. Execution Flow

```text
                  Lead Agent Objective
                            │
                            ▼
             Lead Agent (delegate / execute)
                            │
             Dynamic Task Decomposition
             (decompose_objective)
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
       [Task 1]          [Task 2]          [Task 3]
       Objective         Objective         Objective
      Scoped Context    Scoped Context    Scoped Context
          │                 │                 │
          ▼                 ▼                 ▼
   Temporary Sub-Agent Temporary Sub-Agent Temporary Sub-Agent
   (DeepAgents Graph) (DeepAgents Graph) (DeepAgents Graph)
          │                 │                 │
          ▼                 ▼                 ▼
    [Task 1 Result]   [Task 2 Result]   [Task 3 Result]
      Attribution       Attribution       Attribution
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                Collect Task Results
                 (DelegationResult)
                            │
                            ▼
                   Common Action Result
                   (ActionResult, source=delegation)
                            │
                            ▼
                Stored in BRDAgentState
                (latest_action_result)
                            │
                            ▼
                   Continue BRD Workflow
```

---

## 5. Data Models

### 5.1 `DelegatedTask`
```python
@dataclass
class DelegatedTask:
    task_id: str
    objective: str
    relevant_context: str = ""
    scope: str = ""
    constraints: str = ""
    expected_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.2 `TaskResult`
```python
@dataclass
class TaskResult:
    task_id: str
    objective: str
    content: str
    success: bool = True
    execution_info: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.3 `DelegationResult`
```python
@dataclass
class DelegationResult:
    content: str
    task_results: list[TaskResult] = field(default_factory=list)
    success: bool = True
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.4 `ActionResult` & `ActionSource`
```python
class ActionSource(str, Enum):
    DIRECT_WORK = "direct_work"
    RAG = "rag"
    DELEGATION = "delegation"

@dataclass
class ActionResult:
    source: ActionSource | str
    content: str
    context: AgentContext = field(default_factory=AgentContext)
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## 6. Verification Matrix

The Delegation capability is verified by **17 dedicated tests** in `tests/test_brd_agent_delegation.py`, alongside existing suites in `tests/test_brd_agent_direct_work.py`, `tests/test_brd_agent_rag_integration.py`, and `tests/test_brd_lead_agent.py` (**total 63 tests passing**):

| # | Verification Area | Test Name | Result |
| :--- | :--- | :--- | :--- |
| **1** | Dynamic task decomposition & bounded tasks | `test_dynamic_task_decomposition_produces_bounded_tasks` | Passed |
| **2** | Dynamic task counts (2, 3, 5 tasks) | `test_dynamic_task_decomposition_supports_various_task_counts` | Passed |
| **3** | Cognitive LLM task decomposition | `test_llm_based_dynamic_task_decomposition` | Passed |
| **4** | Sub-agent scoped context delivery | `test_temporary_subagent_receives_only_scoped_context` | Passed |
| **5** | Sub-agent tenant project isolation | `test_temporary_subagent_enforces_project_isolation` | Passed |
| **6** | No recursive delegation (max depth = 1) | `test_subagent_cannot_recursively_delegate` | Passed |
| **7** | Attributable task results & serialization | `test_task_results_are_attributable` | Passed |
| **8** | Result collection & synthesis | `test_result_collection_synthesizes_multiple_task_results` | Passed |
| **9** | Delegation convergence to ActionResult | `test_delegation_converges_to_common_action_result` | Passed |
| **10** | Three-capability ActionResult convergence | `test_three_capabilities_converge_on_action_result_boundary` | Passed |
| **11** | Lead Agent delegation & state continuity | `test_lead_agent_delegates_and_stores_state` | Passed |
| **12** | String prompt decomposition delegation | `test_lead_agent_executes_delegation_via_string_prompt` | Passed |
| **13** | Execute with delegate flag routing | `test_lead_agent_execute_with_delegate_flag` | Passed |
| **14** | Sub-agent execution error preservation | `test_subagent_execution_failure_preserves_error_details` | Passed |
| **15** | Partial failure handling in collection | `test_delegation_collection_with_partial_failure` | Passed |
| **16** | Asynchronous delegation execution | `test_lead_agent_async_delegation_execution` | Passed |
| **17** | Absence of out-of-scope capabilities | `test_no_evaluation_or_permanent_specialist_agents` | Passed |
| **18** | Direct Work capability suite (12 tests) | `tests/test_brd_agent_direct_work.py` | 12 Passed |
| **19** | RAG integration capability suite (16 tests) | `tests/test_brd_agent_rag_integration.py` | 16 Passed |
| **20** | BRD Lead Agent baseline suite (18 tests) | `tests/test_brd_lead_agent.py` | 18 Passed |
