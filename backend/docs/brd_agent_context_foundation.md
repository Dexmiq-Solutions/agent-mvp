# Phase 4.3 — BRD Agent Context Foundation: Template Integration + Agent State

## 1. Executive Overview

This document describes the foundational operating context established for the **BRD Lead Agent** in **Phase 4.3**.

The BRD Lead Agent operates as the primary domain-specific Agent responsible for producing an evidence-grounded Business Requirements Document (BRD). To progress toward this objective, the Agent requires three distinct context pillars:

```text
                  BRD Lead Agent
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
System Instruction   BRD Template    Agent State
        │               │               │
        ↓               ↓               ↓
     Behavior        BRD Structure   Working Context
```

Together, these three components establish the Agent's foundational context:
- **Who am I?** Defined by the **System Instruction**
- **What am I producing?** Defined by the **BRD Template**
- **Where am I and what do I know?** Defined by **Agent State**

---

## 2. Architectural Boundaries & Component Responsibilities

The three context components address fundamentally different questions and remain strictly decoupled:

| Component | Authoritative Artifact | Primary Responsibility | Question Answered |
| :--- | :--- | :--- | :--- |
| **System Instruction** | `system_instruction.md` | Defines Agent identity, behavioral principles, evidence discipline, and project isolation boundaries. | *How should the Agent behave?* |
| **BRD Template** | `brd_template.md` | Defines required document structure, section order, headings, and structural contract. Single source of truth. | *What must the Agent produce?* |
| **Agent State** | `BRDAgentState` | Represents working progress against the template, overall objective vs immediate task, gathered evidence, and unresolved information. | *What is the current working context?* |

### Why They Remain Separate
1. **Behavior vs Output Contract**: How an agent reasons and behaves (truthfulness, refusing to hallucinate, respecting tenancy) is invariant to changes in document sectioning. Conversely, updating a section heading in the template should not require altering behavioral guidelines.
2. **Output Contract vs Working Progress**: The template defines the target destination (the required BRD structure). The state defines current progress toward that destination (which sections are completed, in progress, or require revision).
3. **No Prompt Bloat**: Keeping the template and state distinct from the behavioral instruction prevents prompt dilution and maintains clear architectural boundaries.

---

## 3. BRD Template Integration

### Domain Artifact Ownership
The BRD template is a domain-specific artifact belonging directly to the BRD Agent:

```text
src/
└── agents/
    └── brd/
        ├── __init__.py
        ├── agent.py
        ├── state.py
        ├── system_instruction.md
        └── brd_template.md
```

There is no global, monolithic template registry. Each domain agent owns its required document specifications.

### Single Source of Truth & DRY
The Markdown file `brd_template.md` is the **single source of truth** for the BRD structure:
- Section order, section titles, and section hierarchy are defined strictly within the Markdown file.
- Python code does not maintain hardcoded section lists or duplicate constants (e.g. no `BRD_SECTIONS = [...]`).
- The Agent dynamically parses template sections via `extract_brd_sections()`.
- Modifying or adding a section heading in `brd_template.md` immediately reflects across the Agent's section progress tracking without modifying application code.

### Canonical Template Structure
The approved BRD template defines five canonical sections:
1. **1. Introduction**: Purpose & Executive Summary, Project Scope, Target Audience & Stakeholders.
2. **2. Business Requirements**: Business Goals & Success Metrics, Stakeholders & User Personas, Business Processes & Rules.
3. **3. Functional Requirements**: Features & System Capabilities, User Workflows & Use Cases, Data & Integration Requirements.
4. **4. Non-Functional Requirements**: Performance & Scalability, Security & Compliance, Availability & Reliability.
5. **5. Assumptions**: Business & Operational Assumptions, Dependencies & Technical Constraints.

---

## 4. Agent State Architecture

`BRDAgentState` encapsulates the working context needed for the BRD generation workflow without premature complexity (YAGNI).

### State Model
```python
@dataclass
class BRDAgentState:
    objective: str = "Produce an evidence-grounded Business Requirements Document"
    current_task: Optional[str] = None
    template_sections: list[str] = field(default_factory=list)
    current_section: Optional[str] = None
    section_progress: dict[str, BRDSectionStatus] = field(default_factory=dict)
    evidence: list[Any] = field(default_factory=list)
    unresolved_information: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Core Concepts Represented

1. **BRD Objective vs Immediate Task Context**:
   - `objective`: High-level goal (e.g., producing the full BRD for the project).
   - `current_task`: Bounded immediate work (e.g., analyzing scope, gathering stakeholder requirements).
2. **Template Context & Section Progress**:
   - Initialized dynamically from template sections.
   - Tracks section lifecycle status via `BRDSectionStatus`:
     - `Not Started`: Section has not yet been drafted.
     - `In Progress`: Section is actively being analyzed or populated.
     - `Completed`: Section requirements are fulfilled and validated.
     - `Needs Revision`: Section has gaps, ambiguities, or requires updates.
3. **Evidence / Working Information**:
   - Holds verified facts, retrieved excerpts, or user-confirmed statements.
   - Minimal and extensible; ready for Phase 4.4 RAG evidence grounding.
4. **Unresolved Information & Gaps**:
   - Tracks missing requirements, ambiguities, or unverified assumptions.
   - Supports resolution via `resolve_unresolved()`.
5. **Project Isolation & Metadata**:
   - State metadata captures `project_id` and execution attributes, respecting tenancy boundaries.

---

## 5. Runtime & DeepAgents Integration

The state and template integrate directly with the existing DeepAgents execution harness:

```text
Application Layer
       ↓ (AgentRunRequest: input_text, context, state)
Agent Runtime (DeepAgents Graph)
       ↓
BRD Lead Agent
  ├── system_instruction (system_instruction.md)
  ├── brd_template       (brd_template.md -> extract_brd_sections)
  └── state              (BRDAgentState)
       ↓ (AgentRunResponse: output_text, tool_calls, context, state)
Application Layer
```

- **Harness Compatibility**: `BRDDeepAgentState` provides a TypedDict representation compatible with DeepAgents/LangGraph graph compilation.
- **Request/Response Propagation**: `AgentRunRequest` accepts an initial or restored state, and `AgentRunResponse` returns the updated state upon completion of execution cycles.
- **Zero Framework Bloat**: No secondary database, state engine, or external state-management service is introduced. State is plain, serializable, and inspectable.

---

## 6. Forward Compatibility with Future Capabilities

The state foundation established in this phase is designed to support upcoming capabilities without rework:

- **RAG Capability (Phase 4.4)**: Retrieved project evidence will be appended directly to `state.evidence`, allowing the Agent to reason over project facts.
- **Direct Work / Analysis**: Intermediate analysis results attach to the active section working context.
- **Bounded Delegation**: Delegated sub-tasks can return structured outcomes that populate `state.evidence` or update `state.section_progress`.
- **Section Generation**: As sections are drafted, their status transitions from `Not Started` to `In Progress` to `Completed`.
- **Validation & Revision**: Section validation checks can transition a section status to `Needs Revision` and record specific gaps in `state.unresolved_information`.
