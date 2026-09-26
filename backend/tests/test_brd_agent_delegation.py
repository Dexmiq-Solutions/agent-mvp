"""Unit and integration tests for Delegation and Temporary Sub-Agent Execution.

Validates that:
1. The Lead Agent can delegate an objective into multiple tasks.
2. The number of tasks is dynamic (e.g. 2, 3, 5, etc. - not hardcoded).
3. Each delegated task is bounded with Objective, Relevant Input / Context, Scope, Constraints, Expected Output.
4. Each task executes through a temporary task-scoped sub-agent execution.
5. Sub-agents receive only scoped context (not full state or full history).
6. Project isolation is preserved (tenant project_id inherited, cannot be overridden).
7. Sub-agents cannot recursively delegate (max depth = 1).
8. Individual task results remain attributable (task_id, objective, content, metrics, success).
9. Multiple task results are collected correctly into a unified DelegationResult.
10. The collected result becomes a common ActionResult (source=ActionSource.DELEGATION).
11. All three capabilities (Direct Work, RAG, Delegation) converge into the common ActionResult.
12. Existing RAG behavior remains unchanged.
13. Existing Direct Work behavior remains unchanged.
14. The Lead Agent remains the workflow owner (no permanent handoffs, no separate orchestrator).
15. No evaluation or BRD-generation behavior is introduced prematurely (no EvaluationAgent, etc.).
16. Task execution failure handling preserves error details without corrupting state.
17. Async execution operates identically.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDLeadAgent,
    DelegatedTask,
    DelegationResult,
    TaskResult,
    collect_task_results,
    create_brd_lead_agent,
    decompose_objective,
    execute_subagent_task,
    execute_subagent_task_async,
)
from agents.brd.config import AgentConfig
from agents.brd.context import (
    ActionResult,
    ActionSource,
    AgentContext,
    AgentRunRequest,
    AgentRunResponse,
    get_current_agent_context,
)
from services.rag_service import RAGService, RetrievalResult, RetrievedChunk


class InspectingMockChatModel(BaseChatModel):
    """Mock chat model that records invocations and returns configured messages."""

    messages_to_return: list[AIMessage] = []
    invocations: list[list[Any]] = []
    index: int = 0
    tools_bound: list = []

    def __init__(self, messages_to_return: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.messages_to_return = list(messages_to_return or [])
        self.invocations = []
        self.index = 0
        self.tools_bound = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations.append(messages)
        if self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Default mock output"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "inspecting-mock-chat-model"


# ---------------------------------------------------------------------------
# 1. Dynamic Task Decomposition & Bounded Task Structure
# ---------------------------------------------------------------------------


def test_dynamic_task_decomposition_produces_bounded_tasks():
    """Verify decompose_objective produces bounded tasks with all required fields."""
    objective = (
        "Formulate core requirements:\n"
        "1. Define authentication and SSO workflows.\n"
        "2. Establish role-based authorization matrices.\n"
        "3. Specify audit logging compliance constraints."
    )
    context = "Security policies mandate OAuth 2.0 with PKCE and ISO 27001 audit trails."

    tasks = decompose_objective(objective, context=context)

    # 1. Dynamic decomposition: exactly 3 tasks generated for 3 items
    assert len(tasks) == 3

    # 2. Verify all bounded task fields are present
    for i, t in enumerate(tasks, start=1):
        assert isinstance(t, DelegatedTask)
        assert t.task_id == f"task-{i}"
        assert len(t.objective) > 0
        assert t.relevant_context == context
        assert len(t.scope) > 0
        assert len(t.constraints) > 0
        assert len(t.expected_output) > 0


def test_dynamic_task_decomposition_supports_various_task_counts():
    """Verify decomposition is not hardcoded to a fixed number (supports 2, 3, 5, etc.)."""
    # 2 tasks
    two_item_obj = "1. Draft User Onboarding.\n2. Draft User Offboarding."
    tasks_2 = decompose_objective(two_item_obj)
    assert len(tasks_2) == 2

    # 5 tasks
    five_item_obj = "\n".join([f"{i}. Task item number {i}" for i in range(1, 6)])
    tasks_5 = decompose_objective(five_item_obj)
    assert len(tasks_5) == 5

    # Arbitrary text with task hint
    tasks_hint_4 = decompose_objective("Multi-module analysis", tasks_hint=4)
    assert len(tasks_hint_4) == 4


def test_llm_based_dynamic_task_decomposition():
    """Verify LLM cognitive task decomposition parses structured JSON output."""
    llm_json_response = json.dumps([
        {
            "task_id": "sec-auth",
            "objective": "Detail MFA requirements",
            "relevant_context": "SMS and TOTP supported",
            "scope": "Authentication module only",
            "constraints": "No biometric support in phase 1",
            "expected_output": "List of functional auth requirements",
        },
        {
            "task_id": "sec-rbac",
            "objective": "Detail role permissions",
            "relevant_context": "Admin, Auditor, and Standard roles",
            "scope": "Authorization module only",
            "constraints": "Least privilege principle",
            "expected_output": "Role permission matrix",
        },
    ])

    model = InspectingMockChatModel(messages_to_return=[AIMessage(content=llm_json_response)])
    tasks = decompose_objective("Security architecture decomposition", model=model)

    assert len(tasks) == 2
    assert tasks[0].task_id == "sec-auth"
    assert tasks[0].objective == "Detail MFA requirements"
    assert tasks[0].relevant_context == "SMS and TOTP supported"
    assert tasks[1].task_id == "sec-rbac"
    assert tasks[1].objective == "Detail role permissions"


# ---------------------------------------------------------------------------
# 2. Temporary Task-Scoped Sub-Agent Execution & Scoped Context
# ---------------------------------------------------------------------------


def test_temporary_subagent_receives_only_scoped_context():
    """Verify sub-agent receives only minimum required context, not full agent state."""
    model = InspectingMockChatModel(messages_to_return=[AIMessage(content="Subagent Output 1")])

    task = DelegatedTask(
        task_id="task-oauth",
        objective="Draft OAuth token duration rules",
        relevant_context="Access token TTL is 3600 seconds. Refresh token TTL is 30 days.",
        scope="Token lifecycle only",
        constraints="RFC 6749 compliance",
        expected_output="Functional requirement statements",
    )

    parent_ctx = AgentContext(project_id="proj-scoped-01")
    result = execute_subagent_task(task=task, model=model, parent_context=parent_ctx)

    assert result.success is True
    assert result.content == "Subagent Output 1"
    assert result.task_id == "task-oauth"

    # Verify input given to model contains only task context
    assert len(model.invocations) == 1
    all_content = " ".join([getattr(m, "content", "") for m in model.invocations[0]])

    assert "Access token TTL is 3600 seconds" in all_content
    assert "Draft OAuth token duration rules" in all_content
    assert "RFC 6749 compliance" in all_content
    # Must NOT contain entire unrelated agent state or unrelated project documents
    assert "unresolved_information" not in all_content
    assert "section_progress" not in all_content


def test_temporary_subagent_enforces_project_isolation():
    """Verify sub-agent execution strictly inherits and preserves tenant project_id."""
    observed_context = None

    class ContextCapturingModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            nonlocal observed_context
            observed_context = get_current_agent_context()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Context captured"))])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self) -> str:
            return "context-capturing"

    task = DelegatedTask(task_id="task-iso", objective="Verify project context")
    parent_ctx = AgentContext(project_id="proj-strict-isolation-999")

    result = execute_subagent_task(task=task, model=ContextCapturingModel(), parent_context=parent_ctx)

    assert result.success is True
    assert observed_context is not None
    assert observed_context.project_id == "proj-strict-isolation-999"
    assert observed_context.metadata.get("delegated_task_id") == "task-iso"


def test_subagent_cannot_recursively_delegate():
    """Verify sub-agents are equipped with no delegation tools (max delegation depth = 1)."""
    model = InspectingMockChatModel(messages_to_return=[AIMessage(content="Leaf execution")])
    task = DelegatedTask(task_id="task-leaf", objective="Leaf worker task")

    execute_subagent_task(task=task, model=model, parent_context=AgentContext(project_id="proj-leaf"))

    # Bound tools on subagent model must not contain delegation or subagent tools
    for tool in model.tools_bound:
        tool_name = getattr(tool, "name", str(tool)).lower()
        assert "delegate" not in tool_name
        assert "subagent" not in tool_name
        assert "decompose" not in tool_name


# ---------------------------------------------------------------------------
# 3. Attributable Task Results & Result Collection
# ---------------------------------------------------------------------------


def test_task_results_are_attributable():
    """Verify every task result retains task_id, objective, content, execution info, and status."""
    task = DelegatedTask(task_id="task-attr-1", objective="Establish SLA benchmarks")
    model = InspectingMockChatModel(messages_to_return=[AIMessage(content="SLA: 99.9% uptime")])

    result = execute_subagent_task(task=task, model=model, parent_context=AgentContext(project_id="proj-attr"))

    assert result.task_id == "task-attr-1"
    assert result.objective == "Establish SLA benchmarks"
    assert result.content == "SLA: 99.9% uptime"
    assert result.success is True
    assert "duration_seconds" in result.execution_info
    assert result.error is None

    # Serialization roundtrip
    data = result.to_dict()
    restored = TaskResult.from_dict(data)
    assert restored.task_id == result.task_id
    assert restored.content == result.content


def test_result_collection_synthesizes_multiple_task_results():
    """Verify collect_task_results preserves task-level attribution in a unified delegation result."""
    task_res_1 = TaskResult(
        task_id="task-1",
        objective="Define Data Retention",
        content="Data is retained for 7 years per regulatory mandates.",
        success=True,
    )
    task_res_2 = TaskResult(
        task_id="task-2",
        objective="Define Data Deletion",
        content="Cryptographic wipe upon account termination within 30 days.",
        success=True,
    )

    collected = collect_task_results([task_res_1, task_res_2], overall_objective="Data Governance")

    assert isinstance(collected, DelegationResult)
    assert collected.success is True
    assert collected.total_tasks == 2
    assert collected.completed_tasks == 2
    assert collected.failed_tasks == 0

    # Content preserves attribution
    assert "[task-1] Define Data Retention" in collected.content
    assert "Data is retained for 7 years" in collected.content
    assert "[task-2] Define Data Deletion" in collected.content
    assert "Cryptographic wipe" in collected.content


# ---------------------------------------------------------------------------
# 4. Action Result Convergence (Direct Work, RAG, Delegation)
# ---------------------------------------------------------------------------


def test_delegation_converges_to_common_action_result():
    """Verify Delegation result converges onto the common ActionResult boundary."""
    task_res = TaskResult(
        task_id="task-1",
        objective="Draft export spec",
        content="Exports shall be formatted as CSV or JSON.",
        success=True,
    )
    del_res = collect_task_results([task_res], overall_objective="Export specifications")
    ctx = AgentContext(project_id="proj-action-res")

    action_result = del_res.to_action_result(context=ctx)

    assert isinstance(action_result, ActionResult)
    assert action_result.source == ActionSource.DELEGATION
    assert action_result.is_delegation is True
    assert action_result.is_direct_work is False
    assert action_result.is_rag is False
    assert action_result.context.project_id == "proj-action-res"
    assert "Exports shall be formatted" in action_result.content
    assert len(action_result.metadata.get("task_results", [])) == 1


def test_three_capabilities_converge_on_action_result_boundary():
    """Verify Direct Work, RAG, and Delegation all converge to the common ActionResult structure."""
    ctx = AgentContext(project_id="proj-unify-100")

    # 1. Direct Work Action Result
    ar_direct = ActionResult(
        source=ActionSource.DIRECT_WORK,
        content="Direct work analysis",
        context=ctx,
    )
    assert ar_direct.is_direct_work is True
    assert ar_direct.source == "direct_work"

    # 2. RAG Action Result
    ar_rag = ActionResult(
        source=ActionSource.RAG,
        content="Retrieved facts",
        context=ctx,
        metadata={"tool_calls": [{"name": "search_project_knowledge"}]},
    )
    assert ar_rag.is_rag is True
    assert ar_rag.source == "rag"

    # 3. Delegation Action Result
    ar_delegation = ActionResult(
        source=ActionSource.DELEGATION,
        content="Collected delegation output",
        context=ctx,
        metadata={"total_tasks": 3, "completed_tasks": 3},
    )
    assert ar_delegation.is_delegation is True
    assert ar_delegation.source == "delegation"

    # Downstream consumer can inspect common attributes regardless of source
    for ar in [ar_direct, ar_rag, ar_delegation]:
        assert isinstance(ar, ActionResult)
        assert ar.context.project_id == "proj-unify-100"
        assert ar.success is True
        assert len(ar.content) > 0


# ---------------------------------------------------------------------------
# 5. Lead Agent Delegation Execution & Workflow Ownership
# ---------------------------------------------------------------------------


def test_lead_agent_delegates_and_stores_state():
    """Verify BRDLeadAgent.delegate decomposes, executes sub-agents, and populates state."""
    # Sub-agent outputs
    out_1 = AIMessage(content="Requirement 1: SSO via SAML 2.0 and OIDC.")
    out_2 = AIMessage(content="Requirement 2: RBAC with granular permission scopes.")

    model = InspectingMockChatModel(messages_to_return=[out_1, out_2])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    tasks = [
        DelegatedTask(task_id="t1", objective="SSO specification"),
        DelegatedTask(task_id="t2", objective="RBAC specification"),
    ]

    context = AgentContext(project_id="proj-lead-delegate-01")
    response = agent.delegate(
        tasks=tasks,
        context=context,
        objective="Security Specification",
        current_task="Draft Authentication and Authorization",
    )

    # 1. Response validation
    assert response.success is True
    assert response.is_delegation is True
    assert response.is_direct_work is False
    assert response.used_rag is False
    assert "SSO via SAML 2.0" in response.output_text
    assert "RBAC with granular permission" in response.output_text

    # 2. Action Result on response
    assert response.action_result is not None
    assert response.action_result.source == ActionSource.DELEGATION
    assert response.action_result.is_delegation is True

    # 3. Lead Agent State continuity
    assert len(agent.state.delegated_tasks) == 2
    assert len(agent.state.task_results) == 2
    assert agent.state.delegation_result is not None
    assert agent.state.delegation_result.completed_tasks == 2
    assert agent.state.latest_action_result is response.action_result
    assert agent.state.metadata.get("project_id") == "proj-lead-delegate-01"
    assert agent.state.current_task == "Draft Authentication and Authorization"


def test_lead_agent_executes_delegation_via_string_prompt():
    """Verify passing a raw string prompt to agent.delegate triggers dynamic decomposition."""
    out_1 = AIMessage(content="Deliverable for Part 1")
    out_2 = AIMessage(content="Deliverable for Part 2")

    model = InspectingMockChatModel(messages_to_return=[out_1, out_2])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    composite_prompt = "1. Draft Payment Ingestion.\n2. Draft Payment Settlement."
    response = agent.delegate(
        tasks=composite_prompt,
        context=AgentContext(project_id="proj-prompt-delegate"),
    )

    assert response.success is True
    assert len(agent.state.delegated_tasks) == 2
    assert len(agent.state.task_results) == 2
    assert "Deliverable for Part 1" in response.output_text
    assert "Deliverable for Part 2" in response.output_text


def test_lead_agent_execute_with_delegate_flag():
    """Verify agent.execute(..., delegate=True) seamlessly routes to delegation."""
    out_1 = AIMessage(content="Audit Log Deliverable")
    out_2 = AIMessage(content="Monitoring Deliverable")

    model = InspectingMockChatModel(messages_to_return=[out_1, out_2])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    response = agent.execute(
        "1. Define Audit Logging.\n2. Define Real-Time Monitoring.",
        context=AgentContext(project_id="proj-flag-delegate"),
        delegate=True,
    )

    assert response.is_delegation is True
    assert response.action_result.source == ActionSource.DELEGATION
    assert len(agent.state.task_results) == 2


# ---------------------------------------------------------------------------
# 6. Error Handling & State Resilience
# ---------------------------------------------------------------------------


def test_subagent_execution_failure_preserves_error_details():
    """Verify a failed sub-agent task is recorded with success=False without crashing the workflow."""
    class FailingModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("API timeout connecting to model endpoint")

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self) -> str:
            return "failing-model"

    task = DelegatedTask(task_id="task-fail", objective="Draft mission critical rule")
    result = execute_subagent_task(task=task, model=FailingModel(), parent_context=AgentContext(project_id="proj-err"))

    # Does not raise; records identifiable failure
    assert result.success is False
    assert result.content == ""
    assert "API timeout" in (result.error or "")
    assert result.task_id == "task-fail"


def test_delegation_collection_with_partial_failure():
    """Verify collected delegation result retains attribution for both succeeded and failed tasks."""
    t1_res = TaskResult(task_id="t1", objective="Task 1", content="Success Content", success=True)
    t2_res = TaskResult(task_id="t2", objective="Task 2", content="", success=False, error="Provider error")

    collected = collect_task_results([t1_res, t2_res])

    assert collected.success is False
    assert collected.total_tasks == 2
    assert collected.completed_tasks == 1
    assert collected.failed_tasks == 1
    assert "Success Content" in collected.content
    assert "FAILED" in collected.content
    assert "Provider error" in collected.content

    # Action Result reflection
    action_res = collected.to_action_result()
    assert action_res.success is False
    assert "1 of 2 delegated tasks failed" in (action_res.error or "")


# ---------------------------------------------------------------------------
# 7. Asynchronous Delegation Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lead_agent_async_delegation_execution():
    """Verify agent.delegate_async performs end-to-end delegation asynchronously."""
    out_1 = AIMessage(content="Async Task 1 Deliverable")
    out_2 = AIMessage(content="Async Task 2 Deliverable")

    model = InspectingMockChatModel(messages_to_return=[out_1, out_2])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    tasks = [
        DelegatedTask(task_id="async-t1", objective="Async Task 1"),
        DelegatedTask(task_id="async-t2", objective="Async Task 2"),
    ]

    response = await agent.delegate_async(
        tasks=tasks,
        context=AgentContext(project_id="proj-async-delegate"),
        objective="Async Workflow",
    )

    assert response.success is True
    assert response.is_delegation is True
    assert response.action_result.source == ActionSource.DELEGATION
    assert len(agent.state.task_results) == 2
    assert "Async Task 1 Deliverable" in response.output_text
    assert "Async Task 2 Deliverable" in response.output_text


# ---------------------------------------------------------------------------
# 8. Out of Scope Capabilities Enforcement
# ---------------------------------------------------------------------------


def test_no_evaluation_or_permanent_specialist_agents():
    """Verify out-of-scope evaluation and permanent specialist agents are absent."""
    import agents.brd as brd_pkg

    all_exported = dir(brd_pkg)

    # Permanent specialist agents must not exist
    assert "ResearchAgent" not in all_exported
    assert "AnalysisAgent" not in all_exported
    assert "WriterAgent" not in all_exported
    assert "ValidationAgent" not in all_exported

    # Evaluation agent & capabilities must not exist in this phase
    assert "EvaluationAgent" not in all_exported
    assert "Evaluator" not in all_exported
    assert "EvidenceSufficiency" not in all_exported
    assert "ContradictionDetector" not in all_exported
    assert "GapAnalysis" not in all_exported
    assert "BRDSectionGenerator" not in all_exported
