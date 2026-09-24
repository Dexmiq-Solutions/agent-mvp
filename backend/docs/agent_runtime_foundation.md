# Phase 2: DeepAgents + OpenRouter Agent Foundation Architecture

## 1. Executive Overview

This document describes the foundational Agent runtime established in **Phase 2** of the AI Agent project.
The purpose of Phase 2 is to provide a clean, modular, and observable execution runtime connecting **DeepAgents** to **OpenRouter** models through an OpenAI-compatible interface, while establishing strict architectural boundaries with the existing RAG retrieval system.

> [!IMPORTANT]
> **Current Status: DeepAgents + OpenRouter Agent Foundation**
> This phase establishes the execution harness and model provider connectivity. It is **NOT** the BRD Agent implementation. Business logic, section-by-section BRD generation, and the RAG retrieval tool are explicitly deferred to later phases.

---

## 2. Why DeepAgents Was Introduced

DeepAgents serves as the **Agent execution harness and state graph runtime**.
- **Separation of Concerns**: Decouples domain/business logic (such as future BRD generation) from the operational execution harness.
- **Tool Orchestration**: Provides stateful execution loops where models can reason, select tools, receive deterministic outputs, and continue generation.
- **Provider Agnostic**: DeepAgents operates on top of standard LangChain/LangGraph model interfaces (`BaseChatModel`), allowing the underlying provider to be replaced without redesigning the agent harness.

### Responsibilities
| Component | Primary Responsibility | Explicit Non-Responsibilities |
| :--- | :--- | :--- |
| **DeepAgents** | Graph execution, tool-calling dispatch loop, message history reducer | Business rules, document retrieval, persistence |
| **Agent Runtime (`src/agents/runtime`)** | Tenancy/Context boundary (`AgentContext`), provider model factory (`create_agent_model`), structured lifecycle logging, exception normalization | Direct database access, RAG vector queries |
| **OpenRouter (`langchain_openai`)** | Model inference over OpenAI-compatible HTTP interface | Agent state management, tool execution |
| **RAG Retrieval (`src/rag`)** | Knowledge retrieval ending at `RetrievalResult` | Response generation, prompt engineering |

---

## 3. Architecture & Execution Flow

```text
               ┌────────────────────────────────────────────────────────┐
               │                   Application Layer                    │
               │         (Authenticated Session / project_id)           │
               └───────────────────────────┬────────────────────────────┘
                                           │ AgentRunRequest
                                           ▼ (AgentContext: project_id)
               ┌────────────────────────────────────────────────────────┐
               │              Agent Runtime (src/agents)                │
               │                                                        │
               │   ┌────────────────────────────────────────────────┐   │
               │   │             DeepAgents Harness                 │   │
               │   │           (Compiled State Graph)               │   │
               │   └───────────────────────┬────────────────────────┘   │
               │                           │                            │
               │                           ▼                            │
               │   ┌────────────────────────────────────────────────┐   │
               │   │            Agent Model Factory                 │   │
               │   │      (ChatOpenAI / BaseChatModel)              │   │
               │   └───────────────────────┬────────────────────────┘   │
               └───────────────────────────┼────────────────────────────┘
                                           │
                                           ▼ (HTTP / OpenAI-compatible)
               ┌────────────────────────────────────────────────────────┐
               │                       OpenRouter                       │
               │          (e.g., liquid/lfm-2.5-2.6b:free)              │
               └────────────────────────────────────────────────────────┘
```

### Tool Execution Cycle
```text
Agent Runtime
      ↓
Model Invocation
      ↓
Tool Call (e.g. echo_diagnostic_tool)
      ↓
Deterministic Tool Execution
      ↓
Tool Result (ToolMessage)
      ↓
Model Synthesis
      ↓
AgentRunResponse (output_text, messages, tool_calls, context)
```

---

## 4. OpenRouter & Model Configuration

The model provider connects to OpenRouter via an OpenAI-compatible interface using `langchain_openai.ChatOpenAI`. Configuration is centralized in `src/core/config.py` (`Settings`) and `src/agents/runtime/config.py` (`AgentConfig`).

### Configuration Parameters
- `AGENT_MODEL`: Active model identifier (e.g., `liquid/lfm-2.5-2.6b:free`). Falls back to `LLM_MODEL` or `"gpt-4o"`.
- `OPENROUTER_API_KEY`: OpenRouter authentication credential. Falls back to `OPENAI_API_KEY`.
- `OPENROUTER_BASE_URL`: Endpoint base URL (defaults to `https://openrouter.ai/api/v1`). Falls back to `LLM_BASE_URL`.
- `AGENT_TEMPERATURE`: Generation sampling temperature (default `0.0`).
- `AGENT_TIMEOUT`: Request timeout in seconds (default `60.0`).
- `AGENT_MAX_RETRIES`: Upstream retry attempts (default `2`).

### Security & Credential Hygiene
- All credentials are read strictly from environment variables or `Settings`.
- `AgentConfig.__repr__()` explicitly masks sensitive credentials (`api_key="***"`).
- Secrets are never emitted in structured logs, exceptions, or reports.

---

## 5. Relationship With Existing LLM Infrastructure

The existing `src/llm/` module (`LLMService`, `BaseLLMInterface`, `OpenAICompatibleLLMAdapter`) remains the source of truth for generic, non-agent application LLM needs (such as query transformation in RAG).

- **No Duplicated LLM System**: The Agent runtime uses standard LangChain `BaseChatModel` wrappers directly configured from shared application settings, without altering or competing with `src/llm/`.
- **Clean Adapter Boundary**: If the application requires interop between `LLMService` and Agent components in the future, thin adapters can bridge the two interfaces without coupling.

---

## 6. Boundary Between Agent and RAG

Phase 1 successfully retired the old RAG generation layer, establishing `RetrievalResult` as the hard boundary of RAG retrieval. Phase 2 strictly respects and preserves this boundary:

1. `src/agents/` does NOT import from `src/rag/`.
2. `src/rag/` does NOT import from `src/agents/`.
3. No RAG generation components (`GenerationService`, prompt construction, post-processing, evaluation loops) have been recreated or reintroduced.
4. RAG retrieval tool integration is explicitly deferred to **Phase 3**.

---

## 7. Project Isolation Boundary

Multi-tenancy and project isolation are core architectural constraints:
- The `AgentContext` object carries `project_id`, `conversation_id`, and `user_id`.
- The application (not the model) is the sole authority for project authorization and boundary assignment.
- Future tools (including the Phase 3 RAG tool) receive their project scoping exclusively through the execution context, preventing cross-tenant leakage.

---

## 8. Deterministic Test Tool

To validate tool calling mechanics without introducing external dependencies or domain business logic, Phase 2 implements `echo_diagnostic_tool` in `src/tools/diagnostic.py`:
- Input: `message: str`
- Output: `"[DIAGNOSTIC_OK] Echo: {message}"`
- Function: Validates model tool selection, parameter passing, execution dispatch, and result synthesis within the DeepAgents harness.

---

## 9. Package Management & Test Execution via UV

In accordance with mandatory project standards, **UV is the single source of truth** for all Python dependency management and test execution:

```powershell
# Adding dependencies
uv add <package>

# Synchronizing environment
uv sync

# Running tests
uv run pytest

# Running targeted Agent tests
uv run pytest tests/test_agent_config.py tests/test_agent_model.py tests/test_diagnostic_tool.py tests/test_agent_runtime.py tests/test_agent_integration_smoke.py
```

`pip`, `pip install`, and manual unmanaged virtual environment manipulations are strictly prohibited.

---

## 10. Running the Agent Manually

A developer-facing manual interactive runner is provided to easily exercise and debug the Phase 2 Agent runtime directly from the terminal without writing ad-hoc scripts.

### Start
From the `backend` directory, launch the runner via UV:

```bash
uv run python scripts/run_agent.py
```

### What It Does
- Resolves configuration via `AgentConfig.from_settings()`.
- Initializes `AgentRuntime` with DeepAgents and equips it with `echo_diagnostic_tool`.
- Establishes a development tenant context using project ID `development-test`.
- Enters an interactive terminal loop accepting user prompts and printing synthesized agent responses and tool invocations.

### Example Interaction
```text
========================================
 Agent MVP - Development Runner
========================================
Model:      liquid/lfm-2.5-2.6b:free
Project ID: development-test
Tools:      echo_diagnostic_tool

Type 'exit', 'quit', or press Ctrl+C to exit.
----------------------------------------

Enter your prompt:
> Please call the echo_diagnostic_tool with message 'hello'

Executing request...
  [Tool Invocation: echo_diagnostic_tool({'message': 'hello'})]

Agent:
The echo_diagnostic_tool returned [DIAGNOSTIC_OK] Echo: hello. The diagnostic test completed successfully.

----------------------------------------

Enter your prompt:
> exit

Exiting development runner. Goodbye!
```

### Important Limitation
This script is strictly a **developer convenience and test runner**. It is **not** a production Agent API, a FastAPI route, or a frontend integration. The production application will eventually invoke the exact same `AgentRuntime` interface via authenticated API endpoints and service layers.

---

## 11. Deferred Work

The following capabilities are deliberately out of scope for Phase 2:
- **Phase 3**: RAG Agent Tool (`Agent -> RAG Tool -> RAGService.retrieve() -> RetrievalResult`)
- **Phase 4**: BRD Lead Agent workflow, evidence evaluation, gap detection
- **Phase 5**: Template-driven BRD section state and generation
- **Phase 6**: Human-in-the-Loop (HITL) pauses and bounded task delegation
- **Phase 7**: Final BRD completion, document formatting, and artifact generation

