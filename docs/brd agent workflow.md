# BRD Agent Workflow Instructions

This document provides a comprehensive, node-by-node operational guide to the **Business Requirements Document (BRD) Agent Workflow**. It is structured so that you can follow along directly with the execution workflow diagram, tracing how user inputs transform into a validated, evidence-grounded Business Requirements Document.

---

## 1. Executive Summary & Architecture Overview

The BRD Agent subsystem is an **application-orchestrated multi-agent pipeline** built with Python, LangChain, and DeepAgents. Unlike unstructured conversational agents that attempt to write entire documents in a single unconstrained prompt, this system enforces a deterministic, 9-phase lifecycle governed by strict quality gates, bounded rework loops, and specialized sub-agents.

```
+----------------------------------------------------------------------------------------------------+
|                                    APPLICATION ORCHESTRATION LAYER                                  |
|                                                                                                    |
|  [User Turn] ──> [API / Tenant Isolation] ──> [State Restoration] ──> [Evidence Ingestion]          |
|                                                                              │                     |
|  ┌───────────────────────────────────────────────────────────────────────────┴──────────────────┐  |
|  │ 1. Context & Objective Init  ──>  2. Action Decision (Lead Agent)                             │  |
|  │                                            │                                                  │  |
|  │ 4. Evidence Evaluation (Sub-Agent) <── 3. Action Execution (RAG / Delegation / Direct Work)    │  |
|  │         │                                                                                     │  |
|  │ 5. Sequential Section Loop (RAG ──> Generate ──> Validate ──> Bounded Rework [Max 2])         │  |
|  │         │                                                                                     │  |
|  │ 6. Document Assembly Gate (Deterministic Stitching)                                           │  |
|  │         │                                                                                     │  |
|  │ 7. Final Validation Pass 1 (Sub-Agent) ──> [If Needs Rework] ──> 8. BRD Rewriter Correction   │  |
|  │                                                                         │                     │  |
|  │ 7. Final Validation Pass 2 (Sub-Agent) <────────────────────────────────┘                     │  |
|  │         │                                                                                     │  |
|  │ 9. Document Finalization & Consolidated Clarification Gate                                    │  |
|  └───────────────────────────────────────────┬──────────────────────────────────────────────────┘  |
|                                              ▼                                                     |
|                           [Persisted BRD Sections (PostgreSQL) & Stream Delivery]                  |
+----------------------------------------------------------------------------------------------------+
```

### Core Architecture Principles:
1. **Tenant Project Isolation**: Every query and state snapshot is strictly bounded to a verified `project_id`.
2. **Authoritative Evidence Grounding**: The agent writes requirements based exclusively on indexed project knowledge (`RAGService`) and verified user conversation, refusing to fabricate technical rules or business constraints.
3. **Separation of Concerns**: The Lead Agent manages high-level reasoning and workflow decisions; specialized sub-agents handle bounded generation, evaluation, validation, and minimal text rewriting.
4. **Resumable State**: The entire operational state (`BRDAgentState`) is serialized and persisted with each assistant turn, allowing workflows to pause for clarification and resume seamlessly across sessions.
5. **Persistent, Versioned BRD Storage**: PostgreSQL is the single source of truth for BRD structured sections and their statuses. Conversation message metadata stores foreign-key references (`brd_id`, `brd_version_id`) but never duplicates section content. The working draft is updated in-place on every turn; immutable point-in-time snapshots are created only on explicit request.

---

## 2. Node-by-Node Workflow Walkthrough

Follow each stage below in chronological execution order.

---

### Node 1: API Boundary & Tenant Isolation
* **What it does**: Receives the incoming HTTP/SSE request, verifies authentication, enforces project boundaries, and persists the user message.
* **How it works**: The FastAPI endpoint (`POST /projects/{project_id}/conversations/{id}/messages`) verifies that the user has access to `project_id`. It saves the user turn into the database, closes the pre-agent database transaction to prevent connection starvation during long LLM runs, and initializes an execution context (`AgentContext`).
* **Tools and agents used**: Internal API middleware and `ConversationService`. (No LLMs or tools used).
* **Input**: User prompt string, `project_id`, `conversation_id`, user authentication token, streaming query flag.
* **Output**: Persisted user `MessageModel` and initialized `AgentContext`.
* **Connection**: Proceeds immediately to **Node 2: State Restoration & Evidence Ingestion**.

---

### Node 2: State Restoration & Conversation Evidence Ingestion
* **What it does**: Restores durable workflow progress from prior message metadata, hydrates canonical BRD section state from PostgreSQL, and extracts verified user input as authoritative evidence.
* **How it works**:
  1. The system inspects prior assistant turns for serialized `BRDAgentState`. If found, the turn-specific execution state (`section_progress`, `current_section`, `unresolved_information`, etc.) is restored.
  2. **PostgreSQL Hydration**: `BRDService.get_or_create_canonical_brd()` is called to ensure the canonical `BRDModel` and its active working draft (`BRDVersionModel`, `is_current=True`) exist for the project. If this is the first turn, it initializes all 16 template sections seeded from `brd_template.md`. `BRDService.hydrate_state_from_db()` then re-reads section content and statuses from `brd_sections` rows into `BRDAgentState`, making PostgreSQL the authoritative source for section data rather than message metadata.
  3. User turns are ingested into state evidence (`source="user_conversation"`, `is_authoritative=True`), deliberately ignoring previous assistant turns to prevent recursive model-echo loops.
* **Tools and agents used**: `BRDAgentState.from_dict()`, `BRDService.get_or_create_canonical_brd()`, `BRDService.hydrate_state_from_db()`.
* **Input**: Prior message history, raw conversation metadata, active database session.
* **Output**: Hydrated `BRDAgentState` with canonical section data sourced from PostgreSQL and newly ingested user evidence.
* **Connection**: Proceeds to **Node 3: Project RAG Capability Verification Gate**.

---

### Node 3: Project RAG Capability Verification Gate
* **What it does**: Confirms that project knowledge retrieval (RAG) is available before attempting document generation.
* **How it works**: Inspects `BRDLeadAgent.has_rag_capability`. Grounded BRD creation requires access to project documentation. If absent, the workflow fails fast with a structured diagnostic error rather than producing ungrounded, hallucinated content.
* **Tools and agents used**: Internal capability check.
* **Input**: Configured `RAGService` instance on `BRDLeadAgent`.
* **Output**: Binary gate status (`True` / `False`).
* **Connection**:
  * *If RAG is False*: Halts immediately with `success=False` and an explicit error response.
  * *If RAG is True*: Proceeds to **Node 4: Clarification Resumption Router**.

---

### Node 4: Clarification Resumption Router (Decision Branch)
* **What it does**: Determines whether this invocation is a brand-new generation request or a resumption following user clarification.
* **How it works**: Inspects `state.is_waiting_for_user`, `state.pending_clarification`, and metadata flags:
* **Tools and agents used**: Internal state inspection.
* **Input**: `BRDAgentState` flags.
* **Output**: Routing path (`Consolidated Resume`, `Section Resume`, or `Initial Execution`).
* **Connection**:
  * **Branch A (Consolidated Clarification Resume)**: If user responded to end-of-document questions, the workflow incorporates the answers, updates affected sections via `update_section_async()`, re-validates them, clears unresolved flags, and jumps directly to **Node 16: Phase 6 Document Assembly Gate**.
  * **Branch B (Section Clarification Resume)**: If user answered a specific blocking section question, the answer is recorded, the targeted section is restored as `current_section`, and the workflow jumps directly to **Node 11: Phase 5 Section Evidence Retrieval**.
  * **Branch C (Fresh Workflow Execution)**: Proceeds to **Node 5: Phase 1 Context & Objective Initialization**.

---

### Node 5: Phase 1 — Context & Objective Initialization
* **What it does**: Initializes the overarching document objective and loads authoritative template sections.
* **How it works**: Loads `brd_template.md` and parses all top-level sections (e.g., *Document Header*, *Version History*, *1. Purpose & Scope*, through *14. Quality Gate*). Sets the working objective (defaulting to the user's prompt or project scope) and marks all sections as `NOT_STARTED`.
* **Tools and agents used**: `load_brd_template()`, `extract_brd_sections()`.
* **Input**: Project metadata, user prompt, and `brd_template.md`.
* **Output**: Initialized `state.template_sections` and `state.section_progress`.
* **Connection**: Proceeds to **Node 6: Phase 2 Action Decision**.

---

### Node 6: Phase 2 — Action Decision (Lead Agent Intelligence)
* **What it does**: The Lead Agent evaluates the objective and available project context to choose the initial information-gathering strategy.
* **How it works**: The Lead Agent analyzes project metadata, available indexed documents, and the user's prompt using LLM reasoning. It outputs an `ActionDecision` selecting one of three execution modes:
  1. `RAG`: Broad semantic search across project knowledge.
  2. `DELEGATION`: Decomposing complex discovery into bounded sub-tasks.
  3. `DIRECT_WORK`: Direct analytical synthesis when sufficient initial context is already present.
* **Tools and agents used**: `BRDLeadAgent` (LLM reasoning).
* **Input**: Objective, project name, project description, available document inventory, user request.
* **Output**: Structured `ActionDecision` (`action_type`, `query`, `delegated_tasks`, `reasoning`).
* **Connection**: Proceeds to **Node 7: Phase 3 Action Execution**.

---

### Node 7: Phase 3 — Action Execution
* **What it does**: Executes the information-gathering action decided in Phase 2 and collects raw findings into working state.
* **How it works**:
  * If `RAG`: Executes the `search_project_knowledge` tool with the formulated query. Retrieved chunks are labeled `source="rag"`, `is_authoritative=True`, and stored in `state.evidence`.
  * If `DELEGATION`: Decomposes work into `DelegatedTask` objects and executes temporary, task-scoped sub-agents (depth-capped at 1). Results are aggregated into `state.agent_work`.
  * If `DIRECT_WORK`: Performs analytical synthesis and records findings in `state.agent_work`.
* **Tools and agents used**: `search_project_knowledge` tool (for RAG), temporary sub-agents (for Delegation), or Lead Agent synthesis.
* **Input**: `ActionDecision` parameters and runtime `AgentContext`.
* **Output**: Populated `state.evidence` or `state.agent_work` containing raw domain facts.
* **Connection**: Proceeds to **Node 8: Phase 4 Evidence Evaluation**.

---

### Node 8: Phase 4 — Evidence Evaluation (Evaluation Sub-Agent)
* **What it does**: Objectively assesses whether the collected evidence is sufficient, clear, and complete enough to begin drafting BRD sections.
* **How it works**: Invokes the dedicated `BRDEvaluationAgent`. This sub-agent is strictly tool-less and evaluates evidence against template criteria using categorical classification (`SUFFICIENT` vs `INSUFFICIENT`). It classifies individual requirement items as *Present*, *Missing*, *Unclear*, *Contradictory*, or *Not Applicable*.
* **Tools and agents used**: `BRDEvaluationAgent` (`tools=[]`).
* **Input**: `EvaluationContext` (objective, section requirements, collected evidence).
* **Output**: Structured `EvaluationResult` (`outcome`, `findings`, `missing_information`, `unresolved_information`, `contradictions`).
* **Connection**: Proceeds to **Node 9: Lead Agent Gap Interpretation & Resolution Strategy**.

---

### Node 9: Lead Agent Gap Interpretation & Resolution Strategy
* **What it does**: Interprets evaluation findings and decides whether to draft immediately, perform a targeted RAG retry, or pause for user clarification.
* **How it works**: The Lead Agent applies business intelligence to distinguish between **document specification gaps** (technical details that might exist in other project files) and **business decision gaps** (budget approvals, pricing tiers, or strategic trade-offs requiring human sign-off).
* **Tools and agents used**: `BRDLeadAgent` interpretation logic and optional `search_project_knowledge`.
* **Input**: `EvaluationResult` from Node 8.
* **Output**: `GapResolutionDecision` (`PROCEED_TO_SECTION_GENERATION`, `RAG`, or `ASK_USER`).
* **Connection**:
  * **If SUFFICIENT**: Sets gap decision to proceed. Moves to **Node 10: Phase 5 Section Progression Initialization**.
  * **If INSUFFICIENT (Document Spec Gaps)**: Triggers a targeted RAG retry and re-evaluates. If still insufficient, substantive gaps are recorded in `state.unresolved_information` to prevent blocking, and the workflow advances to **Node 10**.
  * **If INSUFFICIENT (Business Decision Gaps)**: Formulates a targeted clarification question, marks `state.waiting_for_user = True`, and pauses execution, returning the question to the user.

---

### Node 10: Phase 5 — Section Progression Initialization
* **What it does**: Activates the first section in authoritative template order to begin the sequential drafting loop.
* **How it works**: Calls `initialize_progression()`, which sets `state.current_section` to the first section (e.g., `Dexmiq Standard Document Header`) and initializes the section iteration loop.
* **Tools and agents used**: `initialize_progression()`, `load_brd_template()`.
* **Input**: `state.template_sections`.
* **Output**: Updated `state.current_section` and `state.section_progress`.
* **Connection**: Enters the Section Iteration Loop at **Node 11: Section Evidence Retrieval**.

---

### Node 11: Phase 5 — Section Evidence Retrieval (Targeted Section RAG)
* **What it does**: Retrieves focused, section-specific project evidence before drafting a domain section.
* **How it works**: Checks whether the current section is administrative (e.g., *Document Header*, *Version History*, *Quality Gate*). Administrative sections bypass external retrieval. For domain sections (e.g., *3. In-Scope Modules*, *6. High-Level Requirements*), the agent constructs a targeted query from section requirements, checks an in-memory cache to prevent duplicate queries, and retrieves relevant chunks via `search_project_knowledge`.
* **Tools and agents used**: `search_project_knowledge` tool (bypassed for administrative sections).
* **Input**: `current_section`, section requirements, cached queries.
* **Output**: Newly retrieved evidence appended to `state.evidence`.
* **Connection**: Proceeds to **Node 12: Section Generation**.

---

### Node 12: Phase 5 — Section Generation (Section Generation Sub-Agent)
* **What it does**: Drafts the Markdown content for the active section strictly adhering to template structure and provided evidence.
* **How it works**: Invokes `BRDSectionGenerationAgent` with `operation=GENERATE`. The sub-agent receives the section's exact Markdown template, specific requirements, and available evidence. It is strictly tool-less and forbidden from inventing ungrounded facts. If information is absent, it leaves explicit placeholders or flags it rather than hallucinating.
* **Tools and agents used**: `BRDSectionGenerationAgent` (`tools=[]`).
* **Input**: `SectionGenerationContext` (section name, template structure, section requirements, evidence).
* **Output**: `SectionGenerationResult` containing generated Markdown content, recorded in `state.section_content[current_section]`.
* **Connection**: Proceeds to **Node 13: Section Validation**.

---

### Node 13: Phase 5 — Section Validation (Section Validation Sub-Agent)
* **What it does**: Evaluates the drafted section across 7 strict quality and grounding dimensions.
* **How it works**: Invokes `BRDSectionValidationAgent` (`tools=[]`). The validator assesses the section against:
  1. *Template Compliance* (correct headings, tables, formatting)
  2. *Requirement Coverage* (all required topics addressed)
  3. *Completeness* (sufficient detail for high-level BRD)
  4. *Specificity* (concrete statements, no vague fluff)
  5. *Grounding* (strictly backed by evidence, no hallucinations)
  6. *Consistency* (internal coherence)
  7. *Relevance* (focuses on WHAT, not technical HOW)
* **Tools and agents used**: `BRDSectionValidationAgent` (`tools=[]`).
* **Input**: Section Markdown content, section template, requirements, and evidence.
* **Output**: `ValidationResult` (`outcome`: `VALID` or `NEEDS_REWORK`, list of `ValidationFinding` items with severity and location).
* **Connection**: Proceeds to **Node 14: Lead Agent Validation Interpretation & Bounded Rework Loop**.

---

### Node 14: Phase 5 — Lead Agent Validation Interpretation & Bounded Rework Loop
* **What it does**: Interprets validation findings, filters benign observations, and manages a bounded rework loop (maximum 2 attempts).
* **How it works**:
  * **Filtering**: The Lead Agent filters out benign administrative metadata placeholders (e.g., TBD author names in headers) and minor documentation-quality observations.
  * **If VALID**: Marks section `COMPLETED` and advances to **Node 15: Section Progression Gate**.
  * **If NEEDS_REWORK (Attempts < 2)**:
    1. If missing evidence was identified, executes targeted RAG retrieval.
    2. Invokes `BRDSectionGenerationAgent` with `operation=UPDATE` and actionable rework guidance.
    3. Re-invokes `BRDSectionValidationAgent` and re-interprets findings.
    4. If valid, marks `COMPLETED`; if still invalid, increments attempt counter (up to 2).
  * **If Rework Limit Exhausted (2 Attempts Failed)**:
    * *Evidence Gap (Consolidated Mode enabled)*: Genuine substantive business gaps are recorded in `state.unresolved_information`, the draft is preserved, the section is marked `COMPLETED`, and the loop continues to the next section.
    * *Evidence Gap (Immediate Mode)*: Sets section to `NEEDS_REVISION`, sets `state.waiting_for_user = True`, emits clarification question, and pauses workflow.
    * *Generation Quality / Template Failure*: Formats a structured diagnostic failure report, marks `NEEDS_REVISION`, and terminates execution (`success=False`).
* **Tools and agents used**: `BRDLeadAgent`, `BRDSectionGenerationAgent`, `BRDSectionValidationAgent`, and `search_project_knowledge`.
* **Input**: `ValidationResult` and attempt counters.
* **Output**: Validated section content or recorded unresolved gaps.
* **Connection**:
  * If valid or preserved: Proceeds to **Node 15: Section Progression Gate**.
  * If paused for clarification / failed: Emits response and halts.

---

### Node 15: Phase 5 — Section Progression Gate
* **What it does**: Checks if all template sections are completed; advances `current_section` to the next section.
* **How it works**: Calls `determine_next_section()`. If uncompleted sections remain, it deterministically activates the next section in template order. When all sections reach `COMPLETED`, the loop terminates.
* **Tools and agents used**: `progress_to_next_section()`, `get_remaining_sections()`.
* **Input**: `state.section_progress` and `state.template_sections`.
* **Output**: Next active `state.current_section` or completion signal.
* **Connection**:
  * *More sections remain*: Loops back to **Node 11: Section Evidence Retrieval** for the next section.
  * *All sections completed*: Advances to **Node 16: Phase 6 Document Assembly Gate**.

---

### Node 16: Phase 6 — Document Assembly Gate
* **What it does**: Combines all individually validated sections into a unified, complete BRD document.
* **How it works**: Executes the deterministic assembly engine (`assemble_brd_document()`). This is a pure Python algorithm (no LLM call) that verifies all sections are `COMPLETED`, stitches them together in exact template order, harmonizes Markdown heading levels (`#` vs `##`), and formats clean section boundaries.
* **Tools and agents used**: Deterministic assembly engine (`assembly.py`).
* **Input**: `state.section_content` dictionary and template section order.
* **Output**: Complete Markdown document stored in `state.assembled_brd`.
* **Connection**: Proceeds to **Node 17: Phase 7 Document Final Validation (Pass 1)**.

---

### Node 17: Phase 7 — Document Final Validation (Pass 1)
* **What it does**: Evaluates the whole assembled document for cross-section consistency, coherence, and grounding.
* **How it works**: Invokes `BRDFinalValidationAgent` (`tools=[]`). The sub-agent evaluates the complete BRD across 8 document-level dimensions:
  1. *Cross-Section Consistency* (e.g., in-scope modules match requirement sections)
  2. *Requirement Consistency* (no conflicting requirement statements)
  3. *Terminology Consistency* (uniform business terminology throughout)
  4. *Grounding* (document-level facts match project evidence)
  5. *Completeness* (no missing dependencies between sections)
  6. *Duplication* (no redundant requirement blocks)
  7. *Template Compliance* (overall document structure)
  8. *Overall Coherence* (smooth narrative flow)
* **Tools and agents used**: `BRDFinalValidationAgent` (`tools=[]`).
* **Input**: Complete assembled BRD document and project evidence.
* **Output**: `FinalValidationResult` (`outcome`: `VALID` or `NEEDS_REWORK`, list of structured `FinalValidationFinding` items with finding IDs, exact locations, and required corrections).
* **Connection**:
  * **If VALID**: Skips rewriter correction. Jumps directly to **Node 21: Phase 9 Workflow Completion**.
  * **If NEEDS_REWORK**: Proceeds to **Node 18: Phase 8 BRD Rewriter Correction Cycle**.

---

### Node 18: Phase 8 — BRD Rewriter Correction Cycle (BRD Rewriter Sub-Agent)
* **What it does**: Applies minimal, targeted text corrections to fix verified findings without modifying untouched sections.
* **How it works**: The Lead Agent invokes `BRDRewriterAgent` (`tools=[]`). The Rewriter receives the assembled BRD, the specific actionable findings from Final Validation, and *only* the evidence attached to those findings. It outputs structured `DocumentEdit` items (`original_fragment`, `corrected_fragment`, `target_location`). It is forbidden from re-interpreting broader evidence or rewriting unrelated text.
* **Tools and agents used**: `BRDRewriterAgent` (`tools=[]`).
* **Input**: Assembled BRD Markdown, actionable `FinalValidationFinding` items, finding-specific evidence.
* **Output**: `BRDRewriterResult` containing a list of `DocumentEdit` operations.
* **Connection**: Proceeds to **Node 19: Document Edit Application & Section Synchronization**.

---

### Node 19: Phase 8 — Document Edit Application & Section Synchronization
* **What it does**: Applies text replacements to the assembled BRD and synchronizes section content in state.
* **How it works**: The application replaces each `original_fragment` with `corrected_fragment` directly in `state.assembled_brd`. It then synchronizes individual `state.section_content` blocks using template heading boundaries to keep state consistent for session persistence.
* **Tools and agents used**: Internal deterministic string replacement and heading synchronizer.
* **Input**: Current assembled BRD and `DocumentEdit` list.
* **Output**: Updated `state.assembled_brd` and synchronized `state.section_content`.
* **Connection**: Proceeds to **Node 20: Phase 7 Document Final Validation (Pass 2)**.

---

### Node 20: Phase 7 — Document Final Validation (Pass 2 & Terminal Fallback)
* **What it does**: Re-evaluates the edited document to verify that corrections succeeded.
* **How it works**: Executes a second (and final) validation pass via `BRDFinalValidationAgent`.
  * If Pass 2 is `VALID`: The correction cycle is successful.
  * If Pass 2 is `NEEDS_REWORK`: The correction cycle terminates immediately. Remaining unresolved findings and evidence gaps are deterministically converted into numbered items under a `## Open Questions / Clarifications` section appended to the end of the BRD. `state.final_validation_recovery_exhausted` is set to `True`.
* **Tools and agents used**: `BRDFinalValidationAgent` (`tools=[]`).
* **Input**: Updated assembled BRD document.
* **Output**: Final validation outcome (`VALID` or appended Open Questions).
* **Connection**: Advances to **Node 21: Phase 9 Workflow Completion & Consolidated Clarification Gate**.

---

### Node 21: Phase 9 — Workflow Completion & Consolidated Clarification Gate
* **What it does**: Formats the final response and attaches a consolidated questionnaire if substantive business gaps were flagged.
* **How it works**: The Lead Agent inspects `state.unresolved_information`.
  * If substantive business gaps exist and the clarification gate has not yet executed: The agent constructs a consolidated, numbered clarification questionnaire and appends it beneath the assembled BRD (`[Assembled BRD] --- [Clarification Questions]`). It flags `state.waiting_for_user = True` and sets `pending_consolidated_clarification = True`.
  * If no substantive gaps exist: The assembled BRD is finalized directly.
* **Tools and agents used**: `BRDLeadAgent` formatting logic.
* **Input**: Final assembled BRD, `state.unresolved_information`.
* **Output**: Final response text and updated state flags.
* **Connection**: Proceeds to **Node 22: API Response Persistence & Delivery**.

---

### Node 22: API Response Persistence & Delivery
* **What it does**: Atomically persists the canonical BRD sections, the final assistant turn, and workflow state to the database, then delivers output to the user.
* **How it works**: The API layer checks out a fresh database session (`_persist_assistant_turn`) and performs the following in a single transaction:
  1. **BRD Section Persistence**: `BRDService.save_working_draft_from_state()` writes updated section content and statuses back to `brd_sections` rows in PostgreSQL, deriving the `assembled_markdown` via the existing deterministic assembly engine (`assemble_brd_document()`) to keep structured data and Markdown consistent. Only the canonical working draft (`is_current=True`) is mutated.
  2. **Message Metadata Sanitization**: `BRDAgentState` is serialized with `section_content` and `assembled_brd` stripped from `workflow_state` metadata. Only foreign-key references (`brd_id`, `brd_version_id`, `version_number`, `version_label`) are stored in message metadata to prevent a second source of truth.
  3. The assistant message, sanitized state, and LLM telemetry metrics are committed atomically.
* **Tools and agents used**: `BRDService.save_working_draft_from_state()`, `ConversationService`, SQLAlchemy async session, SSE streaming engine.
* **Input**: Final output text, `BRDAgentState`, `brd_id`, `brd_version_id`, telemetry summary.
* **Output**: Updated `brd_sections` rows in PostgreSQL, persisted assistant `MessageModel`, and client HTTP/SSE response.
* **Connection**: Workflow execution completes. System waits for next user turn.

---

## 3. Tools and Agents

This section details every tool, agent, and internal subsystem component actively present in the codebase.

```
+----------------------------------------------------------------------------------------------------+
|                                    SYSTEM COMPONENT TAXONOMY                                        |
|                                                                                                    |
|  1. AGENTS & SUB-AGENTS (Reasoning Units)                                                          |
|     * BRDLeadAgent (Orchestrator, Reasoning, State Owner)                                          |
|     * BRDEvaluationAgent (Tool-less Evidence Evaluator)                                            |
|     * BRDSectionGenerationAgent (Tool-less Section Author & Updater)                                |
|     * BRDSectionValidationAgent (Tool-less Section Quality Gatekeeper)                              |
|     * BRDFinalValidationAgent (Tool-less Whole-Document Validator)                                  |
|     * BRDRewriterAgent (Tool-less Minimal Document Editor)                                         |
|     * Delegated Sub-Agents (Temporary Task Executors, Depth=1)                                     |
|                                                                                                    |
|  2. TOOLS (Callable Capabilities with I/O Schemas)                                                 |
|     * search_project_knowledge (StructuredTool: Tenant-bounded vector & lexical RAG retrieval)     |
|     * echo_diagnostic_tool (StructuredTool: Smoke testing and runtime verification)                |
|                                                                                                    |
|  3. DETERMINISTIC WORKFLOW OPERATIONS (Pure Python Logic / No LLM)                                 |
|     * State Serialization & Restoration (BRDAgentState)                                            |
|     * Progression Engine (SectionProgressionResult, template order traversal)                      |
|     * Assembly Engine (Deterministic markdown stitching & heading level resolver)                  |
|     * Edit Synchronizer (Applies DocumentEdit text diffs to section state)                         |
|     * Telemetry & Rate-Limit Middleware (LLMTelemetryTracker, token usage, backoff retry)          |
+----------------------------------------------------------------------------------------------------+
```

---

### A. Tools in the Codebase

Tools are executable functions provided to agents to interact with external systems. In our architecture, tools are strictly separated from reasoning.

#### 1. `search_project_knowledge`
* **Name**: `search_project_knowledge`
* **File Location**: `backend/src/tools/rag.py`
* **Purpose**: Performs semantic and keyword knowledge retrieval across indexed project documentation.
* **How it works**: Accepts a focused search query string from the LLM. It resolves the trusted `project_id` strictly from the application runtime context (`AgentContext` via contextvars), preventing prompt-injection tenant bypasses. It invokes `RAGService.retrieve()` and returns formatted markdown chunks.
* **Input Schema**:
  ```json
  {
    "query": "string (Focused search query describing needed requirements, architecture, or facts)"
  }
  ```
* **Output**: Formatted string with explicit outcome tags:
  * `[RETRIEVAL_SUCCESS]`: Contains chunk count and formatted document excerpts with source references.
  * `[NO_EVIDENCE]`: Explains that search executed successfully but found zero matching chunks.
  * `[RETRIEVAL_ERROR]`: Diagnostic error string if the retrieval pipeline encountered a failure.
* **Where used**:
  * Lead Agent Phase 3 (initial broad evidence retrieval).
  * Lead Agent Phase 4 (targeted gap recovery retries).
  * Phase 5 (pre-generation section evidence retrieval).
  * Phase 5 (targeted rework evidence retrieval).
  * Phase 3 Temporary Delegated Sub-Agents.

#### 2. `echo_diagnostic_tool`
* **Name**: `echo_diagnostic_tool`
* **File Location**: `backend/src/tools/diagnostic.py`
* **Purpose**: Deterministic diagnostic testing and tool-invocation verification.
* **How it works**: Echoes back the provided input string with a `[DIAGNOSTIC_OK]` prefix. Has no external dependencies and no side effects.
* **Input Schema**: `{"message": "string"}`
* **Output**: `"[DIAGNOSTIC_OK] Echo: <message>"`
* **Where used**: System smoke testing, tool execution path validation, and integration tests.

> [!IMPORTANT]
> **Tool Scoping Guardrail**: All specialized sub-agents (`BRDEvaluationAgent`, `BRDSectionGenerationAgent`, `BRDSectionValidationAgent`, `BRDFinalValidationAgent`, and `BRDRewriterAgent`) are explicitly instantiated with `tools=[]`. They operate purely on in-memory context provided by the Lead Agent, guaranteeing deterministic, bounded execution.

---

### B. Agents and Sub-Agents

Agents are LLM-powered reasoning units configured with specialized system instructions, input contexts, and bounded output schemas.

| Agent Name | Primary Responsibility | Available Tools | Inputs Received | Outputs Produced |
|---|---|---|---|---|
| **BRDLeadAgent** | Orchestrates 9-phase workflow, manages state, decides actions, resolves gaps, coordinates sub-agents | `search_project_knowledge`, `echo_diagnostic_tool` | User message, `AgentContext`, `BRDAgentState` | `ActionDecision`, `GapResolutionDecision`, `SectionReworkStrategy`, `AgentRunResponse` |
| **BRDEvaluationAgent** | Evaluates raw evidence against template requirements prior to drafting | *None* (`tools=[]`) | `EvaluationContext` (objective, section requirements, evidence) | `EvaluationResult` (`SUFFICIENT` / `INSUFFICIENT`, missing items, status per item) |
| **BRDSectionGenerationAgent** | Drafts new sections (`GENERATE`) or updates existing sections based on feedback (`UPDATE`) | *None* (`tools=[]`) | `SectionGenerationContext` (section name, template, evidence, feedback) | `SectionGenerationResult` (generated Markdown section text) |
| **BRDSectionValidationAgent** | Validates individual sections against 7 quality & grounding dimensions | *None* (`tools=[]`) | `SectionValidationContext` (section Markdown, template, evidence) | `ValidationResult` (`VALID` / `NEEDS_REWORK`, structured findings with severity) |
| **BRDFinalValidationAgent** | Validates whole assembled BRD across 8 cross-section & coherence dimensions | *None* (`tools=[]`) | `FinalValidationContext` (assembled BRD Markdown, project evidence) | `FinalValidationResult` (`VALID` / `NEEDS_REWORK`, actionable findings with locations) |
| **BRDRewriterAgent** | Applies minimal text edits (`DocumentEdit`) to fix verified final validation findings | *None* (`tools=[]`) | `BRDRewriterContext` (assembled BRD, findings, finding-specific evidence) | `BRDRewriterResult` (list of `DocumentEdit` items, unapplied finding IDs) |
| **Temporary Sub-Agents** | Executes bounded, decomposed analytical tasks during Phase 3 delegation | Scoped (`search_project_knowledge`) | `DelegatedTask` (task objective, scope, constraints) | `TaskResult` (task output, execution metadata, success status) |

---

### C. Agent Interaction & Collaboration Dynamics

1. **Hierarchical Orchestration (No Peer-to-Peer Agent Chat)**: Sub-agents do not communicate directly with each other. All data flows strictly through the `BRDLeadAgent` and the central `BRDAgentState`. This prevents circular agent debates and unbounded token consumption.
2. **Strict Single-Depth Delegation**: The Lead Agent can delegate tasks to temporary sub-agents, but sub-agents cannot delegate further (delegation depth is strictly capped at 1).
3. **Deterministic State Synchronization**: When the `BRDRewriterAgent` produces text edits on the assembled document in Phase 8, the Lead Agent deterministically synchronizes individual section blocks in `state.section_content` using template heading delimiters. Section validation is never re-opened once Phase 5 completes, preserving clean phase gates.

---

## 4. Key Decision Points & Failure Recovery Reference

When presenting or explaining this workflow, key decision branches to highlight include:

```
                                    KEY DECISION BRANCHES
                                    
  [Evidence Evaluation] ─────────────> SUFFICIENT?
                                        ├── Yes ──> Start Section Loop
                                        └── No  ──> Business Decision?
                                                     ├── Yes (Budget/Scope) ──> Pause for User Clarification
                                                     └── No (Doc Details)   ──> Targeted RAG Retry ──> Proceed
  
  [Section Validation] ──────────────> VALID?
                                        ├── Yes ──> Advance to Next Section
                                        └── No  ──> Rework Attempts < 2?
                                                     ├── Yes ──> RAG (if needed) ──> Update Section ──> Re-validate
                                                     └── No  ──> Evidence Gap?
                                                                  ├── Yes ──> Record in Unresolved ──> Preserve Draft & Continue
                                                                  └── No  ──> Emit Failure Diagnostic & Halt
  
  [Final Validation (Pass 1)] ───────> VALID?
                                        ├── Yes ──> Finalize Document
                                        └── No  ──> Phase 8 Rewriter (Minimal DocumentEdits)
                                                     └── Final Validation (Pass 2)
                                                          ├── Valid ──> Finalize Document
                                                          └── Needs Rework ──> Append ## Open Questions & Finalize
```

* **Max 2 Section Rework Attempts**: Prevents infinite generation loops. If a section cannot pass validation after 2 rework cycles due to missing project evidence, the draft is preserved, the gap is recorded, and drafting proceeds.
* **Single-Pass Rewriter Cycle**: Final document correction runs at most once (Pass 1 -> Rewriter -> Pass 2). If issues remain after Pass 2, they are converted into clear `## Open Questions / Clarifications` for human review rather than looping indefinitely.
* **Consolidated Clarification vs Interruptive Pauses**: By default, genuine evidence gaps encountered during drafting are batched into a clean, numbered questionnaire presented at the end of the BRD, ensuring the user receives a complete draft immediately rather than being interrupted at every section.

---

## 5. BRD Storage Layer

This section documents the persistent, versioned BRD storage layer introduced in Task 1.

### 5.1 Data Model

```
 projects
   └─── brds          (1:1 per project, UNIQUE project_id)
          └─── brd_versions  (working draft + historical snapshots)
                 └─── brd_sections  (structured section rows, ordered)
```

| Table | Key Constraints | Purpose |
|---|---|---|
| `brds` | `UNIQUE project_id` | One canonical BRD per project |
| `brd_versions` | Partial UNIQUE index on `(brd_id) WHERE is_current` | At most one active working draft per BRD; monotonic `version_number` via `UNIQUE (brd_id, version_number)` |
| `brd_sections` | `UNIQUE (brd_version_id, section_key)` | Stable per-section identity within a version |

### 5.2 Draft vs Snapshot Lifecycle

- **Working Draft** (`is_current=True`, `status=DRAFT`): Mutated in-place on every conversation turn by `BRDService.save_working_draft_from_state()`. Only one working draft can exist per BRD at any time, enforced by the database partial unique index.
- **Historical Snapshot** (`is_current=False`, `status=SNAPSHOT`): Created on explicit request via `BRDService.create_version_snapshot()`. All sections are deep-cloned into a new `BRDVersionModel` row. Once frozen, snapshot sections are immutable — any attempt to modify them raises `BRDVersionImmutableError`.

### 5.3 Single Source of Truth

- **PostgreSQL is authoritative** for BRD structured sections, section statuses, and assembled Markdown.
- **Message metadata** (`workflow_state`) stores only turn-specific execution state and foreign-key references (`brd_id`, `brd_version_id`, `version_number`, `version_label`). Fields `section_content` and `assembled_brd` are explicitly stripped before serialization to prevent a second source of truth.
- **State hydration** reads section content and statuses from PostgreSQL into `BRDAgentState` at the start of each turn rather than trusting stale message metadata.

### 5.4 Markdown Derivation

Assembled Markdown is always derived from structured section data via the existing `assemble_brd_document()` engine, not independently trusted from state. This ensures structural data and the Markdown representation remain consistent at all times.

### 5.5 Relevant Files

| File | Role |
|---|---|
| `backend/src/models/brd.py` | `BRDModel`, `BRDVersionModel`, `BRDSectionModel`, `BRDVersionStatus` |
| `backend/src/services/brd_service.py` | `BRDService`: canonical BRD lifecycle, draft updates, snapshots, state hydration |
| `backend/src/exceptions/brd.py` | Domain exceptions: `BRDNotFoundError`, `BRDVersionImmutableError`, `BRDVersionNotFoundError`, `InvalidBRDDataError` |
| `backend/alembic/versions/20261014_0001_create_brd_tables.py` | Alembic migration creating `brds`, `brd_versions`, `brd_sections` tables |
| `backend/src/api/conversations.py` | Integration point: `BRDService` injected, state hydrated pre-agent, sections persisted atomically post-agent |
| `backend/tests/test_brd_storage.py` | Integration tests covering persistence, update, hydration, snapshot, immutability, and metadata sanitization lifecycles |

---
*Document Reference: Dexmiq BRD Agent Architecture — Instructions & Operational Guide.*
