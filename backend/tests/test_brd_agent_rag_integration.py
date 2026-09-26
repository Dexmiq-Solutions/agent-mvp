"""Unit and integration tests for Phase 4.4 — BRD Agent RAG Capability Integration.

Validates that:
1. The BRD Lead Agent can access the existing RAG tool (search_project_knowledge).
2. The tool executes through the DeepAgents harness.
3. The correct project context reaches RAG (strict project isolation).
4. Existing RAG retrieval executes successfully via RAGService.
5. A RetrievalResult returns to the BRD Lead Agent.
6. The Agent can continue its reasoning after receiving the result.
7. The Agent can proceed without RAG when information is already available.
8. Existing runtime-Agent RAG behavior remains unaffected.
9. No duplicate RAG implementation was introduced.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDLeadAgent,
    create_brd_lead_agent,
    load_system_instruction,
)
from agents.brd.config import AgentConfig
from agents.brd.context import AgentContext, AgentRunRequest, AgentRunResponse
from services.rag_service import RAGService, RetrievalResult, RetrievedChunk
from tools.diagnostic import echo_diagnostic_tool
from tools.rag import (
    SearchProjectKnowledgeInput,
    create_search_project_knowledge_tool,
    search_project_knowledge,
)


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for unit testing BRD Lead Agent RAG execution."""

    messages_to_return: list[AIMessage]
    index: int = 0
    tools_bound: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.messages_to_return):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Default fallback"))]
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
# 1. Access to Existing RAG Tool & Registration Tests
# ---------------------------------------------------------------------------


def test_brd_lead_agent_has_rag_capability_by_default():
    """Verify BRDLeadAgent equips the canonical search_project_knowledge tool by default."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    assert agent.has_rag_capability is True
    tool_names = [t.name for t in agent.tools]
    assert "search_project_knowledge" in tool_names
    assert "echo_diagnostic_tool" in tool_names

    # Verify tool contract and schema
    rag_tool = next(t for t in agent.tools if t.name == "search_project_knowledge")
    assert rag_tool.args_schema == SearchProjectKnowledgeInput
    assert "search project documentation" in rag_tool.description.lower()


def test_create_brd_lead_agent_factory_equips_rag():
    """Verify create_brd_lead_agent factory equips search_project_knowledge."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = create_brd_lead_agent(config=config, model=model)

    assert agent.has_rag_capability is True
    assert any(t.name == "search_project_knowledge" for t in agent.tools)


def test_brd_lead_agent_rag_can_be_explicitly_disabled():
    """Verify enable_rag=False disables the RAG tool when constructing standalone agent."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, enable_rag=False)

    assert agent.has_rag_capability is False
    tool_names = [t.name for t in agent.tools]
    assert "search_project_knowledge" not in tool_names
    assert "echo_diagnostic_tool" in tool_names


def test_brd_lead_agent_with_injected_rag_service():
    """Verify BRDLeadAgent accepts an injected RAGService instance."""
    mock_service = MagicMock(spec=RAGService)
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    assert agent.has_rag_capability is True
    tool_names = [t.name for t in agent.tools]
    assert "search_project_knowledge" in tool_names


def test_no_duplicate_rag_tool_created():
    """Verify no duplicate tools such as brd_search_project_knowledge exist."""
    model = MockChatModel(messages_to_return=[AIMessage(content="Ready")])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    tool_names = [t.name for t in agent.tools]
    assert "brd_search_project_knowledge" not in tool_names
    assert tool_names.count("search_project_knowledge") == 1


# ---------------------------------------------------------------------------
# 2. DeepAgents Execution Harness & RAG Retrieval Flow
# ---------------------------------------------------------------------------


def test_brd_lead_agent_executes_rag_through_deepagents_sync():
    """Verify full sync execution: BRD Agent -> DeepAgents -> RAG Tool -> RAG Service -> Agent."""
    project_id = "proj-test-sync-101"
    query = "What authentication mechanism is used?"
    evidence_text = "The system uses Okta SAML 2.0 with MFA for enterprise SSO."

    # Setup mock RAG service returning evidence
    mock_service = MagicMock(spec=RAGService)
    mock_result = _create_mock_retrieval_result(
        project_id=project_id,
        query=query,
        chunks=[{
            "chunk_id": "auth-chunk-1",
            "document_id": "auth_architecture.md",
            "content": evidence_text,
            "score": 0.95,
        }],
    )
    mock_service.retrieve = AsyncMock(return_value=mock_result)

    # 1st LLM call: decides to invoke search_project_knowledge
    call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": query},
            "id": "call_rag_sync_001",
            "type": "tool_call",
        }],
    )
    # 2nd LLM call: synthesizes evidence into BRD requirement
    final_brd_text = (
        "## Functional Requirements: Authentication\n"
        "The application shall authenticate users via Okta SAML 2.0 with MFA."
    )
    final_msg = AIMessage(content=final_brd_text)

    model = MockChatModel(messages_to_return=[call_msg, final_msg])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    context = AgentContext(project_id=project_id)
    response = agent.execute("Define authentication requirements", context=context)

    # Verification: Tool executed through DeepAgents
    assert response.success is True
    assert response.output_text == final_brd_text
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "search_project_knowledge"
    assert response.tool_calls[0]["args"] == {"query": query}

    # Verification: RAG service invoked with correct project context
    mock_service.retrieve.assert_called_once_with(
        project_id=project_id,
        query=query,
    )

    # Verification: Message history contains HumanMessage -> AIMessage -> ToolMessage -> AIMessage
    message_types = [type(m).__name__ for m in response.messages]
    assert "HumanMessage" in message_types
    assert "ToolMessage" in message_types
    assert "AIMessage" in message_types

    # Verification: ToolMessage received evidence
    tool_messages = [m for m in response.messages if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "[RETRIEVAL_SUCCESS]" in tool_messages[0].content
    assert evidence_text in tool_messages[0].content


@pytest.mark.asyncio
async def test_brd_lead_agent_executes_rag_through_deepagents_async():
    """Verify full async execution: BRD Agent -> DeepAgents -> RAG Tool -> RAG Service -> Agent."""
    project_id = "proj-test-async-202"
    query = "What are the data retention constraints?"
    evidence_text = "Audit logs must be retained for 7 years under HIPAA compliance."

    mock_service = MagicMock(spec=RAGService)
    mock_result = _create_mock_retrieval_result(
        project_id=project_id,
        query=query,
        chunks=[{
            "chunk_id": "compliance-chunk-1",
            "document_id": "regulatory_requirements.md",
            "content": evidence_text,
            "score": 0.88,
        }],
    )
    mock_service.retrieve = AsyncMock(return_value=mock_result)

    call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": query},
            "id": "call_rag_async_002",
            "type": "tool_call",
        }],
    )
    final_text = "Data retention policy: 7 years per regulatory guidelines."
    final_msg = AIMessage(content=final_text)

    model = MockChatModel(messages_to_return=[call_msg, final_msg])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    context = AgentContext(project_id=project_id)
    response = await agent.execute_async("Analyze data retention requirements", context=context)

    assert response.success is True
    assert response.output_text == final_text
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "search_project_knowledge"

    mock_service.retrieve.assert_called_once_with(
        project_id=project_id,
        query=query,
    )


# ---------------------------------------------------------------------------
# 3. Project Isolation & Tenant Boundary Enforcement
# ---------------------------------------------------------------------------


def test_project_isolation_tenant_separation():
    """Verify Project A and Project B receive strictly isolated retrieval contexts."""
    mock_service = MagicMock(spec=RAGService)

    # Result for Project A
    result_a = _create_mock_retrieval_result("tenant-a-uuid", "database engine", [
        {"content": "Project A uses PostgreSQL 16."}
    ])
    # Result for Project B
    result_b = _create_mock_retrieval_result("tenant-b-uuid", "database engine", [
        {"content": "Project B uses DynamoDB."}
    ])

    async def _retrieve_dispatch(project_id, query):
        if project_id == "tenant-a-uuid":
            return result_a
        elif project_id == "tenant-b-uuid":
            return result_b
        return _create_mock_retrieval_result(project_id, query)

    mock_service.retrieve = AsyncMock(side_effect=_retrieve_dispatch)

    # Agent execution for Tenant A
    model_a = MockChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": "database engine"},
            "id": "call_a",
            "type": "tool_call",
        }]),
        AIMessage(content="Tenant A DB is PostgreSQL"),
    ])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent_a = BRDLeadAgent(config=config, model=model_a, rag_service=mock_service)

    resp_a = agent_a.execute("Check DB", context=AgentContext(project_id="tenant-a-uuid"))
    assert resp_a.context.project_id == "tenant-a-uuid"

    # Agent execution for Tenant B
    model_b = MockChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": "database engine"},
            "id": "call_b",
            "type": "tool_call",
        }]),
        AIMessage(content="Tenant B DB is DynamoDB"),
    ])
    agent_b = BRDLeadAgent(config=config, model=model_b, rag_service=mock_service)

    resp_b = agent_b.execute("Check DB", context=AgentContext(project_id="tenant-b-uuid"))
    assert resp_b.context.project_id == "tenant-b-uuid"

    # Verify RAG was called with distinct project IDs
    assert mock_service.retrieve.call_count == 2
    assert mock_service.retrieve.call_args_list[0].kwargs["project_id"] == "tenant-a-uuid"
    assert mock_service.retrieve.call_args_list[1].kwargs["project_id"] == "tenant-b-uuid"


def test_missing_project_context_rejects_retrieval():
    """Verify RAG tool rejects invocation when project context is missing, preserving isolation."""
    mock_service = MagicMock(spec=RAGService)
    model = MockChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": "any requirements"},
            "id": "call_no_proj",
            "type": "tool_call",
        }]),
        AIMessage(content="Error acknowledged: Project context required."),
    ])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    # Execute without a valid project_id
    response = agent.execute("Search knowledge", context=AgentContext(project_id=None))

    # Verify RAG was NOT called
    mock_service.retrieve.assert_not_called()

    # Verify tool returned boundary violation error
    tool_messages = [m for m in response.messages if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "[RETRIEVAL_ERROR] Project boundary violation" in tool_messages[0].content


def test_model_cannot_select_or_modify_project_identifier():
    """Verify tool schema accepts only query, preventing model from injecting a project ID."""
    tool = create_search_project_knowledge_tool()
    schema_properties = tool.args_schema.model_json_schema()["properties"]

    assert "query" in schema_properties
    assert "project_id" not in schema_properties
    assert "tenant_id" not in schema_properties


# ---------------------------------------------------------------------------
# 4. Retrieval Outcomes: Success, No-Evidence, and Error Handling
# ---------------------------------------------------------------------------


def test_brd_lead_agent_handles_no_evidence_outcome():
    """Verify BRDLeadAgent receives [NO_EVIDENCE] when retrieval finds zero chunks."""
    project_id = "proj-no-evidence-303"
    query = "quantum computing acceleration requirements"

    mock_service = MagicMock(spec=RAGService)
    mock_result = _create_mock_retrieval_result(project_id=project_id, query=query, chunks=[])
    mock_service.retrieve = AsyncMock(return_value=mock_result)

    call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": query},
            "id": "call_no_ev",
            "type": "tool_call",
        }],
    )
    final_msg = AIMessage(
        content="No project evidence found for quantum computing. Recorded as open question."
    )

    model = MockChatModel(messages_to_return=[call_msg, final_msg])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    response = agent.execute("Search for quantum requirements", context=AgentContext(project_id=project_id))

    assert response.success is True
    tool_messages = [m for m in response.messages if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "[NO_EVIDENCE]" in tool_messages[0].content


def test_brd_lead_agent_handles_retrieval_service_error():
    """Verify BRDLeadAgent receives [RETRIEVAL_ERROR] when RAG service raises an exception."""
    project_id = "proj-err-404"
    mock_service = MagicMock(spec=RAGService)
    mock_service.retrieve = AsyncMock(side_effect=RuntimeError("Vector index timeout"))

    call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": "fail query"},
            "id": "call_err",
            "type": "tool_call",
        }],
    )
    final_msg = AIMessage(content="Retrieval service experienced an error.")

    model = MockChatModel(messages_to_return=[call_msg, final_msg])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    response = agent.execute("Trigger error", context=AgentContext(project_id=project_id))

    assert response.success is True
    tool_messages = [m for m in response.messages if type(m).__name__ == "ToolMessage"]
    assert len(tool_messages) == 1
    assert "[RETRIEVAL_ERROR]" in tool_messages[0].content
    assert "Vector index timeout" in tool_messages[0].content


# ---------------------------------------------------------------------------
# 5. RAG Decision: Selective Retrieval (Not Every Request -> RAG)
# ---------------------------------------------------------------------------


def test_brd_lead_agent_proceeds_without_rag_when_context_sufficient():
    """Verify the Agent proceeds directly without retrieval when information is already in context."""
    mock_service = MagicMock(spec=RAGService)

    # Prompt can be answered directly from the prompt content
    direct_reply = "The document title is: 'Dexmiq Billing System Phase 1'."
    model = MockChatModel(messages_to_return=[AIMessage(content=direct_reply)])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, rag_service=mock_service)

    context = AgentContext(project_id="proj-direct-505")
    response = agent.execute(
        "Please format the document title as 'Dexmiq Billing System Phase 1'.",
        context=context,
    )

    # Agent answered directly without calling search_project_knowledge
    assert response.success is True
    assert response.output_text == direct_reply
    assert len(response.tool_calls) == 0
    mock_service.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Runtime Agent RAG Behavior Unaffected
# ---------------------------------------------------------------------------


def test_agent_rag_behavior():
    """Verify that agent continues to use search_project_knowledge identically."""
    mock_service = MagicMock(spec=RAGService)
    mock_result = _create_mock_retrieval_result("agent-proj-606", "general query", [
        {"content": "Agent knowledge chunk."}
    ])
    mock_service.retrieve = AsyncMock(return_value=mock_result)

    rag_tool = create_search_project_knowledge_tool(rag_service=mock_service)
    model = MockChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[{
            "name": "search_project_knowledge",
            "args": {"query": "general query"},
            "id": "agent_call_1",
            "type": "tool_call",
        }]),
        AIMessage(content="Agent answered using retrieved knowledge."),
    ])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model, tools=[rag_tool])

    response = agent.execute("Search knowledge", context=AgentContext(project_id="agent-proj-606"))

    assert response.success is True
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "search_project_knowledge"
    mock_service.retrieve.assert_called_once_with(
        project_id="agent-proj-606",
        query="general query",
    )


# ---------------------------------------------------------------------------
# 7. Architecture & Boundary Verification
# ---------------------------------------------------------------------------


def test_brd_agent_has_no_direct_infrastructure_dependencies():
    """Verify BRD Lead Agent code has no direct imports or dependencies on DB/vector stores."""
    import agents.brd.agent as brd_agent_mod

    source = inspect.getsource(brd_agent_mod)

    # Agent must not directly talk to database or vector storage
    assert "qdrant" not in source.lower()
    assert "asyncpg" not in source.lower()
    assert "psycopg" not in source.lower()
    assert "sqlalchemy" not in source.lower()
    assert "supabase" not in source.lower()
    assert "bm25" not in source.lower()
    assert "rerank" not in source.lower()


def test_system_instruction_contains_knowledge_retrieval_guidance():
    """Verify system instruction contains authoritative guidance on RAG decisions and boundaries."""
    instruction = load_system_instruction()

    assert "## Knowledge Retrieval & RAG Decisions" in instruction
    assert "search_project_knowledge" in instruction
    assert "When to Retrieve" in instruction
    assert "When to Proceed Without Retrieval" in instruction
    assert "Focused Queries" in instruction
    assert "RetrievalResult" in instruction
    assert "Project Context" in instruction
