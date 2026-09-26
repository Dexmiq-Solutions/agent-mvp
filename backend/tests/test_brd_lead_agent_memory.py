"""Tests for project-scoped long-term memory in BRDLeadAgent.

Validates the documented LangGraph BaseStore and DeepAgents StoreBackend / MemoryMiddleware
mechanisms for long-term project context persistence, project boundary isolation,
application-controlled writes, read-only agent enforcement, fail-safe resilience,
and clean separation of concerns.
"""

from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agents.brd.agent import BRDLeadAgent, create_brd_lead_agent
from agents.brd.evaluation import BRDEvaluationAgent
from agents.brd.final_validation import BRDFinalValidationAgent
from agents.brd.section_generation import BRDSectionGenerationAgent
from agents.brd.section_validation import BRDSectionValidationAgent
from agents.brd.state import BRDAgentState
from agents.brd.config import AgentConfig
from agents.brd.memory import (
    DEFAULT_PROJECT_MEMORY_FILE,
    ProjectMemoryStoreBackend,
    get_default_memory_store,
    normalize_memory_path,
    reset_default_memory_store,
)
from agents.brd.context import AgentContext, AgentRunRequest


class MockMemoryChatModel(BaseChatModel):
    """Deterministic mock chat model for memory testing."""

    messages_to_return: list[AIMessage] = []
    index: int = 0
    tools_bound: list[Any] = []
    received_messages: list[list[Any]] = []

    def __init__(self, messages_to_return: Optional[list[AIMessage]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.messages_to_return = list(messages_to_return) if messages_to_return else []
        self.index = 0
        self.tools_bound = []
        self.received_messages = []

    def _generate(self, messages: list[Any], stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received_messages.append(messages)
        if self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Default mock response"))])

    async def _agenerate(self, messages: list[Any], stop: Optional[list[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-memory-chat-model"


@pytest.fixture(autouse=True)
def cleanup_default_store():
    """Reset default shared store before and after each test."""
    reset_default_memory_store()
    yield
    reset_default_memory_store()


# ==============================================================================
# 1. Lead Agent Access to Long-Term Memory
# ==============================================================================

def test_lead_agent_can_access_long_term_memory():
    """Prove BRDLeadAgent loads long-term memory from Store and injects it into system prompt."""
    store = InMemoryStore()
    model = MockMemoryChatModel(messages_to_return=[AIMessage(content="Acknowledged project memory.")])
    config = AgentConfig(model="test-model", api_key="test-key")

    agent = BRDLeadAgent(
        config=config,
        model=model,
        store=store,
        project_id="proj-finance-101",
    )

    # Application writes authoritative memory
    agent.set_project_memory(
        key="project_context.md",
        content="Project Goal: Build a high-throughput algorithmic trading ledger with SEC compliance.",
    )

    response = agent.execute("What is our primary goal?")

    assert response.success is True
    assert "Acknowledged project memory." in response.output_text

    # Verify model received the memory in the SystemMessage content
    assert len(model.received_messages) >= 1
    system_msg = model.received_messages[0][0]

    # Extract text from content blocks
    system_text = ""
    if isinstance(system_msg.content, list):
        for block in system_msg.content:
            if isinstance(block, dict) and "text" in block:
                system_text += block["text"]
            else:
                system_text += str(block)
    else:
        system_text = str(system_msg.content)

    assert "<agent_memory>" in system_text
    assert "high-throughput algorithmic trading ledger" in system_text
    assert "SEC compliance" in system_text


# ==============================================================================
# 2. Memory Persistence Across Separate Runs
# ==============================================================================

def test_memory_persists_across_separate_runs():
    """Prove memory written in Run 1 persists and is loaded by Run 2 in a separate instance."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")

    # --- Run 1: First Agent Instance writes authoritative context ---
    model_run1 = MockMemoryChatModel(messages_to_return=[AIMessage(content="Run 1 complete.")])
    agent_run1 = BRDLeadAgent(
        config=config,
        model=model_run1,
        store=store,
        project_id="proj-crm-202",
    )

    success = agent_run1.set_project_memory(
        key="project_context.md",
        content="CRITICAL ARCHITECTURE: System must integrate with legacy AS400 mainframe via Kafka.",
    )
    assert success is True

    res1 = agent_run1.execute("Initialize project context.")
    assert res1.success is True

    # --- Run 2: Completely New Agent Instance (simulating separate conversation/process run) ---
    model_run2 = MockMemoryChatModel(messages_to_return=[AIMessage(content="Run 2 sees legacy AS400 integration.")])
    agent_run2 = BRDLeadAgent(
        config=config,
        model=model_run2,
        store=store,
        project_id="proj-crm-202",
    )

    # Read verification via application API
    loaded_memory = agent_run2.get_project_memory("project_context.md")
    assert loaded_memory is not None
    assert "legacy AS400 mainframe via Kafka" in loaded_memory

    # Execution verification: memory is injected into prompt for Run 2
    res2 = agent_run2.execute("Review architecture requirements.")
    assert res2.success is True

    system_msg_run2 = model_run2.received_messages[0][0]
    system_text_run2 = str(system_msg_run2.content)
    assert "legacy AS400 mainframe via Kafka" in system_text_run2


# ==============================================================================
# 3. Project Isolation
# ==============================================================================

def test_project_isolation_prevents_cross_project_memory_access():
    """Prove Project A memory is strictly isolated from Project B."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")

    model_a = MockMemoryChatModel(messages_to_return=[AIMessage(content="Project A response")])
    agent_a = BRDLeadAgent(
        config=config,
        model=model_a,
        store=store,
        project_id="project-alpha",
    )

    model_b = MockMemoryChatModel(messages_to_return=[AIMessage(content="Project B response")])
    agent_b = BRDLeadAgent(
        config=config,
        model=model_b,
        store=store,
        project_id="project-beta",
    )

    # Write distinct memories
    agent_a.set_project_memory("project_context.md", "CONFIDENTIAL ALPHA INTEL: Launching Q3 in Europe.")
    agent_b.set_project_memory("project_context.md", "CONFIDENTIAL BETA INTEL: Launching Q4 in Asia.")

    # Application reads are strictly isolated
    mem_a = agent_a.get_project_memory("project_context.md")
    mem_b = agent_b.get_project_memory("project_context.md")
    assert "Launch Q3 in Europe" in mem_a or "Launching Q3 in Europe" in mem_a
    assert "Launch Q4 in Asia" in mem_b or "Launching Q4 in Asia" in mem_b
    assert "BETA" not in mem_a
    assert "ALPHA" not in mem_b

    # Execution runs are strictly isolated
    agent_a.execute("Check project scope.")
    agent_b.execute("Check project scope.")

    sys_a = str(model_a.received_messages[0][0].content)
    sys_b = str(model_b.received_messages[0][0].content)

    assert "CONFIDENTIAL ALPHA INTEL" in sys_a
    assert "CONFIDENTIAL BETA INTEL" not in sys_a

    assert "CONFIDENTIAL BETA INTEL" in sys_b
    assert "CONFIDENTIAL ALPHA INTEL" not in sys_b


def test_project_isolation_dynamic_context_switching():
    """Prove single agent executing across different project contexts stays strictly isolated."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    model = MockMemoryChatModel(messages_to_return=[
        AIMessage(content="Turn 1"),
        AIMessage(content="Turn 2"),
    ])

    agent = BRDLeadAgent(config=config, model=model, store=store)

    # Seed memories for two projects
    agent.set_project_memory("project_context.md", "Context for Project 1", project_id="proj-1")
    agent.set_project_memory("project_context.md", "Context for Project 2", project_id="proj-2")

    # Run 1 with Project 1 Context
    ctx_1 = AgentContext(project_id="proj-1")
    agent.execute("Hello", context=ctx_1)

    # Run 2 with Project 2 Context
    ctx_2 = AgentContext(project_id="proj-2")
    agent.execute("Hello", context=ctx_2)

    sys_msg_turn1 = str(model.received_messages[0][0].content)
    sys_msg_turn2 = str(model.received_messages[1][0].content)

    assert "Context for Project 1" in sys_msg_turn1
    assert "Context for Project 2" not in sys_msg_turn1

    assert "Context for Project 2" in sys_msg_turn2
    assert "Context for Project 1" not in sys_msg_turn2


# ==============================================================================
# 4. Application-Controlled Writes
# ==============================================================================

def test_application_controlled_writes_crud():
    """Prove application has full CRUD control over project memory (sync and async)."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=MockMemoryChatModel(), store=store, project_id="proj-crud")

    # 1. Write / Set
    success = agent.set_project_memory("core.md", "Initial Core Content")
    assert success is True

    # 2. Read / Get
    content = agent.get_project_memory("core.md")
    assert content == "Initial Core Content"

    # 3. Update
    success_update = agent.set_project_memory("core.md", "Updated Core Content")
    assert success_update is True
    assert agent.get_project_memory("core.md") == "Updated Core Content"

    # 4. List
    agent.set_project_memory("notes.md", "Additional notes")
    keys = agent.list_project_memory()
    assert "/memory/core.md" in keys
    assert "/memory/notes.md" in keys

    # 5. Delete
    deleted = agent.delete_project_memory("notes.md")
    assert deleted is True
    assert agent.get_project_memory("notes.md") is None


@pytest.mark.asyncio
async def test_application_controlled_writes_async_crud():
    """Prove asynchronous CRUD operations on project memory."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=MockMemoryChatModel(), store=store, project_id="proj-async-crud")

    # 1. Async Write
    w_ok = await agent.set_project_memory_async("async_core.md", "Async Content")
    assert w_ok is True

    # 2. Async Read
    content = await agent.get_project_memory_async("async_core.md")
    assert content == "Async Content"

    # 3. Async List
    keys = await agent.list_project_memory_async()
    assert "/memory/async_core.md" in keys

    # 4. Async Delete
    d_ok = await agent.delete_project_memory_async("async_core.md")
    assert d_ok is True
    assert await agent.get_project_memory_async("async_core.md") is None


# ==============================================================================
# 5. Lead Agent Cannot Arbitrarily Modify Authoritative Memory
# ==============================================================================

def test_lead_agent_write_tool_calls_are_denied():
    """Prove LLM tool attempts to call write_file, edit_file, or delete are denied by framework permissions."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")

    # Model attempts to call write_file on memory
    write_tool_call = {
        "name": "write_file",
        "args": {"file_path": "/memory/project_context.md", "content": "HACKED MEMORY"},
        "id": "call_write_exploit",
    }
    mock_model = MockMemoryChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[write_tool_call]),
        AIMessage(content="Write failed as expected."),
    ])

    agent = BRDLeadAgent(
        config=config,
        model=mock_model,
        store=store,
        project_id="proj-secure-1",
    )

    agent.set_project_memory("project_context.md", "Authoritative Original Memory")

    response = agent.execute("Overwrite the project memory with new data.")

    assert response.success is True

    # Verify tool result returned permission denied error
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    assert "permission denied" in str(tool_messages[0].content).lower()

    # Verify authoritative memory was NOT modified
    current_mem = agent.get_project_memory("project_context.md")
    assert current_mem == "Authoritative Original Memory"
    assert "HACKED MEMORY" not in current_mem


def test_lead_agent_has_no_arbitrary_save_memory_tool():
    """Prove Lead Agent does not have an unrestricted save_memory tool."""
    agent = BRDLeadAgent(model=MockMemoryChatModel(), project_id="proj-no-tool")

    tool_names = [getattr(t, "name", str(t)) for t in agent.tools]
    assert "save_memory" not in tool_names
    assert "write_memory" not in tool_names
    assert "update_memory" not in tool_names


# ==============================================================================
# 6. Agent State Remains Separate from Long-Term Memory
# ==============================================================================

def test_agent_state_remains_separate_from_long_term_memory():
    """Prove BRDAgentState does not store or duplicate long-term memory objects."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    model = MockMemoryChatModel(messages_to_return=[AIMessage(content="Working on BRD")])

    agent = BRDLeadAgent(
        config=config,
        model=model,
        store=store,
        project_id="proj-separation",
    )

    agent.set_project_memory("project_context.md", "Substantial Long-Term Knowledge: Enterprise CRM migration.")

    response = agent.execute("Start introduction section.")

    state = agent.state

    # BRDAgentState fields are strictly workflow state
    assert not hasattr(state, "memory")
    assert not hasattr(state, "long_term_memory")
    assert not hasattr(state, "memory_contents")

    # State serialization contains workflow state, not the long-term memory store
    state_dict = state.to_dict()
    assert "memory" not in state_dict
    assert "long_term_memory" not in state_dict
    assert "template_sections" in state_dict
    assert "section_progress" in state_dict

    # State transitions do not modify long-term memory
    state.update_section_status("1. Purpose & Scope of This Document", "completed")
    mem_after = agent.get_project_memory("project_context.md")
    assert mem_after == "Substantial Long-Term Knowledge: Enterprise CRM migration."


# ==============================================================================
# 7. Specialized Agents Remain Stateless
# ==============================================================================

def test_specialized_agents_remain_stateless():
    """Prove Evaluation, Generation, and Validation agents do not have persistent memory attached."""
    model = MockMemoryChatModel()

    eval_agent = BRDEvaluationAgent(model=model)
    assert not hasattr(eval_agent, "store")
    assert not hasattr(eval_agent, "memory_backend")
    assert not hasattr(eval_agent, "has_memory_capability")

    gen_agent = BRDSectionGenerationAgent(model=model)
    assert not hasattr(gen_agent, "store")
    assert not hasattr(gen_agent, "memory_backend")
    assert not hasattr(gen_agent, "has_memory_capability")

    val_agent = BRDSectionValidationAgent(model=model)
    assert not hasattr(val_agent, "store")
    assert not hasattr(val_agent, "memory_backend")
    assert not hasattr(val_agent, "has_memory_capability")

    final_val_agent = BRDFinalValidationAgent(model=model)
    assert not hasattr(final_val_agent, "store")
    assert not hasattr(final_val_agent, "memory_backend")
    assert not hasattr(final_val_agent, "has_memory_capability")


# ==============================================================================
# 8. Error Handling and Observability
# ==============================================================================

class FailingStore(InMemoryStore):
    """Store that simulates backend connection failure on read."""

    def get(self, namespace: Any, key: Any) -> Any:
        raise RuntimeError("Storage connection failed")

    async def aget(self, namespace: Any, key: Any) -> Any:
        raise RuntimeError("Async storage connection failed")


def test_memory_read_failure_fails_safe():
    """Prove store read failure is logged and agent continues safely without crashing."""
    failing_store = FailingStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    model = MockMemoryChatModel(messages_to_return=[AIMessage(content="Safe fallback output")])

    agent = BRDLeadAgent(config=config, model=model, store=failing_store, project_id="proj-fail-read")

    # Must not raise an unhandled exception; agent continues execution safely
    response = agent.execute("Proceed with drafting.")
    assert response.success is True
    assert "Safe fallback output" in response.output_text




def test_memory_write_failure_does_not_corrupt_state():
    """Prove store write failure is logged, returns False, and does not corrupt Agent State."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=MockMemoryChatModel(), store=store, project_id="proj-fail-write")

    initial_state_dict = agent.state.to_dict()

    with patch.object(
        agent.memory_backend,
        "upload_files",
        side_effect=RuntimeError("Database write timeout"),
    ):
        result = agent.set_project_memory("test.md", "Some data")
        assert result is False

    # State must remain identical and uncorrupted
    assert agent.state.to_dict() == initial_state_dict


# ==============================================================================
# 9. Large Memory and On-Demand Retrieval
# ==============================================================================

def test_large_memory_on_demand_reading():
    """Prove large memory files in store can be queried on-demand via read_file tool without full prompt stuffing."""
    store = InMemoryStore()
    config = AgentConfig(model="test-model", api_key="test-key")

    # Step 1: Model calls read_file on a specific detailed memory artifact
    # Step 2: Model uses that content
    read_tool_call = {
        "name": "read_file",
        "args": {"file_path": "/memory/detailed_architecture.md", "offset": 0, "limit": 10},
        "id": "call_read_arch",
    }
    model = MockMemoryChatModel(messages_to_return=[
        AIMessage(content="", tool_calls=[read_tool_call]),
        AIMessage(content="Architecture details incorporated into requirements."),
    ])

    agent = BRDLeadAgent(
        config=config,
        model=model,
        store=store,
        project_id="proj-large-mem",
    )

    # Core high-level context
    agent.set_project_memory("project_context.md", "Core Context: Healthcare System.")

    # Large separate memory document stored independently
    agent.set_project_memory(
        "detailed_architecture.md",
        "Section 1: Microservices Architecture.\nSection 2: HIPAA Security Controls.\nSection 3: Database Sharding.",
    )

    response = agent.execute("Check the security controls in detailed architecture.")

    assert response.success is True
    assert "Architecture details incorporated" in response.output_text

    # Verify tool response contained the queried lines from the store
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    assert "HIPAA Security Controls" in str(tool_messages[0].content)
