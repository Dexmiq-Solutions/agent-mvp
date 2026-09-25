"""Unit and integration tests for Phase 4.5 — BRD Agent Direct Work Capability.

Validates that:
1. The BRD Lead Agent can handle tasks directly when required information is already available.
2. Direct Work does not require a separate tool (no direct_work_tool, execute_task, etc.).
3. Direct Work does not unnecessarily invoke RAG.
4. Direct Work is not defined only by simplicity (substantial analytical reasoning over context is Direct Work).
5. Contrast: RAG is invoked when external project knowledge is genuinely missing.
6. Existing Agent State remains intact and current_task flows properly.
7. Workflow continuity across turns: RAG retrieval -> Direct Work refinement.
8. DeepAgents remains the execution harness (message flow is HumanMessage -> AIMessage directly).
9. Async execution of Direct Work works identically.
10. No delegation primitives or sub-agent architectures are introduced.
11. System instruction provides authoritative Action Determination & Direct Work guidance.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDLeadAgent,
    create_brd_lead_agent,
    load_system_instruction,
)
from agents.runtime.agent import AgentRuntime
from agents.runtime.config import AgentConfig
from agents.runtime.state import AgentContext, AgentRunRequest, AgentRunResponse
from services.rag_service import RAGService, RetrievalResult, RetrievedChunk
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import search_project_knowledge


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for verifying BRD Lead Agent execution branches."""

    messages_to_return: list[AIMessage]
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default fallback response"))]
            )
        msg = self.messages_to_return[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def _create_mock_retrieval_result(
    project_id: str,
    query: str,
    chunks: list[dict] | None = None,
) -> MagicMock:
    """Helper creating a mock RetrievalResult satisfying the RAG tool contract."""
    mock_result = MagicMock(spec=RetrievalResult)
    mock_result.project_id = project_id
    mock_result.original_query = query
    mock_result.retrieval_query = query

    if chunks:
        mock_chunks = []
        formatted_parts = []
        for i, c in enumerate(chunks, start=1):
            chunk_obj = RetrievedChunk(
                chunk_id=c.get("chunk_id", f"chunk-{i}"),
                document_id=c.get("document_id", f"doc-{i}"),
                project_id=project_id,
                content=c.get("content", f"Content {i}"),
                score=c.get("score", 0.9),
                rank=i,
                metadata=c.get("metadata", {}),
            )
            mock_chunks.append(chunk_obj)
            formatted_parts.append(
                f"[Source: {chunk_obj.document_id} | Score: {chunk_obj.score:.2f}]\n{chunk_obj.content}"
            )
        mock_result.chunks = tuple(mock_chunks)
        mock_result.is_empty = False
        mock_result.formatted_text = "\n\n".join(formatted_parts)
    else:
        mock_result.chunks = ()
        mock_result.is_empty = True
        mock_result.formatted_text = ""

    return mock_result


# ---------------------------------------------------------------------------
# 1. Primary Direct Work Verification (Prompt Examples)
# ---------------------------------------------------------------------------


def test_direct_work_rewrite_requirement_example_1():
    """Verify Example 1: Lead Agent rewrites an existing requirement directly without RAG."""
    input_text = "Rewrite this requirement clearly:\nUsers must be able to reset their password."
    rewritten_text = (
        "The system shall provide an automated, self-service password reset workflow enabling "
        "registered users to securely initiate a password reset via email verification link "
        "with a 15-minute token expiration window."
    )

    mock_rag_service = MagicMock(spec=RAGService)
    model = MockChatModel(messages_to_return=[AIMessage(content=rewritten_text)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_rag_service)

    context = AgentContext(project_id="proj-direct-001")
    response = agent.execute(input_text, context=context)

    # 1. Direct Work executed successfully
    assert response.success is True
    assert response.output_text == rewritten_text

    # 2. No tools were invoked (Direct Work is internal reasoning)
    assert len(response.tool_calls) == 0
    assert response.is_direct_work is True
    assert response.used_rag is False

    # 3. RAG retrieval was NOT called
    mock_rag_service.retrieve.assert_not_called()

    # 4. Message flow is direct: HumanMessage -> AIMessage
    message_types = [type(m).__name__ for m in response.messages]
    assert message_types == ["HumanMessage", "AIMessage"]


def test_direct_work_summarize_existing_context_example_2():
    """Verify Example 2: Lead Agent summarizes existing requirements without RAG."""
    input_text = (
        "Summarize the requirements listed above:\n"
        "1. Users must authenticate with MFA.\n"
        "2. Admins must have role-based access control.\n"
        "3. System must maintain audit logs for all security events."
    )
    summary_text = (
        "Executive Summary of Security Requirements:\n"
        "- MFA is mandatory for all user authentications.\n"
        "- Granular RBAC governs administrative privileges.\n"
        "- Comprehensive audit logging records all security events."
    )

    mock_rag_service = MagicMock(spec=RAGService)
    model = MockChatModel(messages_to_return=[AIMessage(content=summary_text)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_rag_service)

    response = agent.execute(input_text, context=AgentContext(project_id="proj-direct-002"))

    assert response.success is True
    assert response.output_text == summary_text
    assert response.is_direct_work is True
    assert len(response.tool_calls) == 0
    mock_rag_service.retrieve.assert_not_called()


def test_direct_work_substantial_reasoning_deduplicate_and_conflict_example_3():
    """Verify Example 3: Substantial analytical reasoning over context is Direct Work (Complexity != Delegation)."""
    input_text = (
        "Identify duplicate and conflicting requirements in the following list:\n"
        "REQ-1: Passwords must be at least 8 characters long.\n"
        "REQ-2: User sessions expire after 15 minutes of inactivity.\n"
        "REQ-3: All user passwords shall have a minimum length of 8 chars.\n"
        "REQ-4: User sessions must remain active indefinitely without timeout."
    )
    analysis_text = (
        "Analysis of Requirements:\n"
        "1. Duplicate Identified: REQ-1 and REQ-3 both mandate an 8-character minimum password length.\n"
        "2. Contradiction Identified: REQ-2 (15-min idle timeout) directly contradicts REQ-4 (indefinite session duration)."
    )

    mock_rag_service = MagicMock(spec=RAGService)
    model = MockChatModel(messages_to_return=[AIMessage(content=analysis_text)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_rag_service)

    response = agent.execute(input_text, context=AgentContext(project_id="proj-direct-003"))

    assert response.success is True
    assert response.output_text == analysis_text
    assert response.is_direct_work is True
    assert len(response.tool_calls) == 0
    mock_rag_service.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Tool Boundaries & No Separate Direct Work Tool
# ---------------------------------------------------------------------------


def test_direct_work_does_not_create_or_require_separate_tool():
    """Verify no direct_work_tool, execute_task, or perform_work tools exist on the agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    tool_names = [t.name for t in agent.tools]

    # Prohibited tool abstractions
    assert "direct_work" not in tool_names
    assert "direct_work_tool" not in tool_names
    assert "execute_task" not in tool_names
    assert "perform_work" not in tool_names
    assert "lead_agent_work" not in tool_names

    # Only approved tools are equipped
    assert "search_project_knowledge" in tool_names
    assert "echo_diagnostic_tool" in tool_names
    assert len(tool_names) == 2


# ---------------------------------------------------------------------------
# 3. Decision Contrast: Direct Work vs RAG
# ---------------------------------------------------------------------------


def test_agent_selects_direct_work_when_info_present_vs_rag_when_info_missing():
    """Verify clear branch selection on the same agent: Direct Work when info present, RAG when missing."""
    mock_rag_service = MagicMock(spec=RAGService)
    project_id = "proj-contrast-101"

    # Turn 1: Information is missing -> Agent calls RAG
    rag_query = "What is the token expiration duration for OAuth?"
    mock_result = _create_mock_retrieval_result(
        project_id=project_id,
        query=rag_query,
        chunks=[{"content": "OAuth access tokens expire after 3600 seconds (1 hour)."}],
    )
    mock_rag_service.retrieve = AsyncMock(return_value=mock_result)

    turn1_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": rag_query},
            "id": "call_oauth_1",
            "type": "tool_call",
        }],
    )
    turn1_synthesis = AIMessage(content="OAuth access tokens expire in 1 hour.")

    # Turn 2: Information is already present -> Agent performs Direct Work
    turn2_direct = AIMessage(content="Format requirement: The system access token TTL is 3600 seconds.")

    model = MockChatModel(messages_to_return=[turn1_call, turn1_synthesis, turn2_direct])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_rag_service)

    # Execute Turn 1 (RAG Branch)
    resp1 = agent.execute("What is the token expiration duration?", context=AgentContext(project_id=project_id))
    assert resp1.used_rag is True
    assert resp1.is_direct_work is False
    assert len(resp1.tool_calls) == 1
    assert mock_rag_service.retrieve.call_count == 1

    # Execute Turn 2 (Direct Work Branch)
    resp2 = agent.execute("Format that into a requirement statement", context=AgentContext(project_id=project_id))
    assert resp2.used_rag is False
    assert resp2.is_direct_work is True
    assert len(resp2.tool_calls) == 0
    # RAG retrieve call count must remain 1 (no new calls!)
    assert mock_rag_service.retrieve.call_count == 1


# ---------------------------------------------------------------------------
# 4. Agent State & Context Lifecycle
# ---------------------------------------------------------------------------


def test_direct_work_preserves_agent_state_and_updates_current_task():
    """Verify that existing BRDAgentState remains intact and current_task flows through execution."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Requirement rewritten.")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Establish initial state context
    agent.state.objective = "Draft Security Section"
    agent.state.set_current_section("3. Functional Requirements")
    agent.state.add_evidence("Existing Fact: TLS 1.3 mandated")

    task_desc = "Rewrite cipher suite requirements"
    response = agent.execute(
        "Rewrite the following requirement: ...",
        context=AgentContext(project_id="proj-state-004"),
        current_task=task_desc,
    )

    # State verification
    assert response.state is agent.state
    assert response.state.objective == "Draft Security Section"
    assert response.state.current_section == "3. Functional Requirements"
    assert response.state.current_task == task_desc
    assert "Existing Fact: TLS 1.3 mandated" in response.state.evidence
    assert response.state.metadata.get("project_id") == "proj-state-004"


def test_brd_agent_state_set_current_task_method():
    """Verify BRDAgentState.set_current_task updates the task attribute."""
    state = BRDAgentState()
    assert state.current_task is None

    state.set_current_task("Analyze persona mapping")
    assert state.current_task == "Analyze persona mapping"

    state.set_current_task(None)
    assert state.current_task is None


def test_direct_work_with_agent_run_request_payload():
    """Verify Direct Work operates identically when provided an AgentRunRequest model."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Direct Work Result from AgentRunRequest")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    req_state = BRDAgentState(
        objective="Custom Run Objective",
        current_task="Direct Task in Request",
    )
    request = AgentRunRequest(
        input_text="Clarify the user role definitions",
        context=AgentContext(project_id="proj-req-005"),
        state=req_state,
    )

    response = agent.execute(request)
    assert response.success is True
    assert response.is_direct_work is True
    assert response.output_text == "Direct Work Result from AgentRunRequest"
    assert response.state.objective == "Custom Run Objective"
    assert response.state.current_task == "Direct Task in Request"


# ---------------------------------------------------------------------------
# 5. DeepAgents Runtime Harness & Async Execution
# ---------------------------------------------------------------------------


def test_direct_work_executes_through_deepagents_harness():
    """Verify Direct Work runs via DeepAgents graph runnable, not an ad-hoc model invoke bypass."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Direct harness response")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Verify underlying runtime graph is DeepAgents
    assert agent.graph is agent.runtime.graph
    assert hasattr(agent.graph, "invoke")

    response = agent.execute("Direct work via harness")
    assert response.success is True
    assert len(response.messages) >= 2
    assert response.messages[0].content == "Direct work via harness"
    assert response.messages[1].content == "Direct harness response"


@pytest.mark.asyncio
async def test_direct_work_async_execution():
    """Verify asynchronous execute_async handles Direct Work identically."""
    direct_content = "Async direct work: All non-admin sessions expire in 30 minutes."
    model = MockChatModel(messages_to_return=[AIMessage(content=direct_content)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    response = await agent.execute_async(
        "Refine session timeout rule",
        context=AgentContext(project_id="proj-async-006"),
        current_task="Session timeout refinement",
    )

    assert response.success is True
    assert response.is_direct_work is True
    assert response.output_text == direct_content
    assert response.state.current_task == "Session timeout refinement"
    assert response.state.metadata.get("project_id") == "proj-async-006"


# ---------------------------------------------------------------------------
# 6. Absence of Delegation Primitives (Out of Scope Enforcement)
# ---------------------------------------------------------------------------


def test_direct_work_does_not_delegate():
    """Verify Direct Work handles tasks directly without triggering delegation or sub-agents."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Direct execution only")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    response = agent.execute("Rewrite this requirement: Users must log in.")
    assert response.is_direct_work is True
    assert response.is_delegation is False
    assert len(agent.state.delegated_tasks) == 0
    assert len(agent.state.task_results) == 0
    assert agent.state.delegation_result is None


# ---------------------------------------------------------------------------
# 7. System Instruction Guidance for Action Determination & Direct Work
# ---------------------------------------------------------------------------


def test_system_instruction_contains_action_determination_and_direct_work():
    """Verify system_instruction.md contains authoritative guidance for Direct Work and Action Determination."""
    instruction = load_system_instruction()

    # Required sections and conceptual frameworks
    assert "## Action Determination & Operational Branches" in instruction
    assert "## Direct Work Capability" in instruction
    assert "Direct Work" in instruction
    assert "search_project_knowledge" in instruction
    assert "Definition & Scope" in instruction
    assert "When to Use Direct Work" in instruction
    assert "Not Defined Only by Simplicity" in instruction or "NOT defined only by simplicity" in instruction
    assert "Workflow Continuity" in instruction

    # Ensures RAG sections remain preserved and unchanged
    assert "## Knowledge Retrieval & RAG Decisions" in instruction
    assert "When to Retrieve" in instruction
    assert "When to Proceed Without Retrieval" in instruction
