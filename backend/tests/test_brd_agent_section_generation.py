"""Unit and integration tests for BRD Section Generation / Update Sub-Agent capability.

Validates that:
1. The Section Generation Sub-Agent can be constructed and loads its System Instruction from markdown.
2. The System Instruction is separate from the Lead Agent's and Evaluation Sub-Agent's instructions.
3. Focused SectionGenerationContext receives section_name, section_requirements, template_structure, available_information, existing_content, rework_feedback.
4. Section template structure is dynamically extracted from authoritative template (brd_template.md).
5. Structured SectionGenerationResult is returned (section_name, content, operation, summary).
6. Generator behavior handles both Initial Generation and Section Update / Rework.
7. Rework feedback can be supplied and consumed during updates.
8. Fallback parsing handles direct Markdown output when model does not return strict JSON.
9. Strict boundaries: generator has no tools, cannot call RAG, cannot access DBs, cannot ask user.
10. Lead Agent integration: Section Generation Sub-Agent -> Generated/Updated Section -> BRD Lead Agent -> Agent State.
11. State tracks section_content, rework_feedback, and latest_section_result.
12. Iterative lifecycle: Section generation -> Rework feedback -> Section update -> Stored in State.
13. Execution failures are safely caught without crashing the agent.
14. Async generation behaves identically to sync.
"""

from __future__ import annotations

import json
from typing import Any, Optional
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.brd import (
    BRDAgentState,
    BRDEvaluationAgent,
    BRDLeadAgent,
    BRDSectionGenerationAgent,
    BRDSectionStatus,
    SectionGenerationContext,
    SectionGenerationResult,
    SectionOperation,
    create_brd_lead_agent,
    extract_section_requirements,
    extract_section_template,
    get_evaluation_system_instruction_path,
    get_generation_system_instruction_path,
    get_system_instruction_path,
    load_evaluation_system_instruction,
    load_generation_system_instruction,
    load_system_instruction,
)
from agents.runtime.config import AgentConfig
from agents.runtime.state import ActionResult, ActionSource, AgentContext


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing section generation cycles."""

    messages_to_return: list[AIMessage] = []
    invocations: list[list[Any]] = []
    index: int = 0
    tools_bound: list = []
    error_to_raise: Optional[Exception] = None

    def __init__(
        self,
        messages_to_return: Optional[list[AIMessage]] = None,
        error_to_raise: Optional[Exception] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.messages_to_return = list(messages_to_return or [])
        self.error_to_raise = error_to_raise
        self.invocations = []
        self.index = 0
        self.tools_bound = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations.append(messages)
        if self.error_to_raise is not None:
            raise self.error_to_raise
        if self.index < len(self.messages_to_return):
            msg = self.messages_to_return[self.index]
            self.index += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=json.dumps({
                            "section_name": "5. Stakeholders & Personas",
                            "operation": "generate",
                            "summary": "Default mock section generation",
                            "content": "# 5. Stakeholders & Personas\n\n## 5.1 Personas\n- Default persona",
                        })
                    )
                )
            ]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        self.tools_bound = list(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "mock-generation-chat-model"


# ---------------------------------------------------------------------------
# 1. Generator Construction & Separate System Instruction
# ---------------------------------------------------------------------------


def test_generator_construction_and_defaults():
    """Verify Section Generation Sub-Agent can be instantiated and has empty tools list."""
    model = MockChatModel()
    generator = BRDSectionGenerationAgent(model=model)

    assert generator.agent_name == "BRDSectionGenerationAgent"
    assert generator.model is model
    # Strict boundary: Generator has NO tools
    assert len(generator.tools) == 0
    assert "BRD Section Generation / Update Sub-Agent" in generator.system_instruction


def test_generation_system_instruction_loaded_from_markdown():
    """Verify system instruction is loaded from external markdown file with required sections."""
    path = get_generation_system_instruction_path()
    assert path.is_file()
    assert path.name == "system_instruction.md"
    assert path.parent.name == "section_generation"

    content = load_generation_system_instruction()
    assert len(content) > 100
    assert "## Identity" in content
    assert "## Objective" in content
    assert "## Section Generation Rules" in content
    assert "## Section Update Rules" in content
    assert "## Strict Boundaries & Prohibitions" in content
    assert "## Output Contract" in content


def test_system_instructions_are_physically_distinct():
    """Verify Lead Agent, Evaluation, and Section Generation have separate instructions."""
    lead_path = get_system_instruction_path()
    eval_path = get_evaluation_system_instruction_path()
    gen_path = get_generation_system_instruction_path()

    assert lead_path != gen_path
    assert eval_path != gen_path
    assert lead_path != eval_path

    lead_content = load_system_instruction()
    eval_content = load_evaluation_system_instruction()
    gen_content = load_generation_system_instruction()

    assert "BRD Lead Agent" in lead_content
    assert "BRD Evaluation Sub-Agent" in eval_content
    assert "BRD Section Generation / Update Sub-Agent" in gen_content


# ---------------------------------------------------------------------------
# 2. Template Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_section_template_for_various_sections():
    """Verify extract_section_template correctly extracts markdown slices from template."""
    # Test Section 3: In-Scope Business Modules
    sec3_template = extract_section_template("3. In-Scope Business Modules & Feature Groups")
    assert "# 3. In-Scope Business Modules & Feature Groups" in sec3_template
    assert "Module ID (HL-MOD-X)" in sec3_template
    assert "Module Name" in sec3_template
    # Must NOT bleed into Section 4
    assert "# 4. Out-of-Scope Items" not in sec3_template

    # Test Section 5: Stakeholders & Personas (with subsections 5.1 and 5.2)
    sec5_template = extract_section_template("5. Stakeholders & Personas")
    assert "# 5. Stakeholders & Personas" in sec5_template
    assert "## 5.1 Personas" in sec5_template
    assert "## 5.2 Persona–Module Mapping" in sec5_template
    # Must NOT bleed into Section 6
    assert "# 6. High-Level Business Requirements by Module" not in sec5_template

    # Test Section 5.1 Subsection direct extraction
    sec51_template = extract_section_template("5.1 Personas")
    assert "## 5.1 Personas" in sec51_template
    assert "Persona name, responsibilities" in sec51_template


def test_extract_section_template_fallback():
    """Verify fallback for non-existent section."""
    custom_template = extract_section_template("Non-Existent Section")
    assert "# Non-Existent Section" in custom_template


# ---------------------------------------------------------------------------
# 3. Context & Result Data Models
# ---------------------------------------------------------------------------


def test_section_generation_context_auto_detects_operation():
    """Verify SectionGenerationContext auto-configures operation based on existing content."""
    # Initial generation (no existing content)
    ctx_gen = SectionGenerationContext(
        section_name="5. Stakeholders & Personas",
        section_requirements=["Define personas"],
        available_information=["Ops Manager, Warehouse Lead"],
    )
    assert ctx_gen.operation == SectionOperation.GENERATE

    # Update operation (existing content present)
    ctx_upd = SectionGenerationContext(
        section_name="5. Stakeholders & Personas",
        section_requirements=["Define personas"],
        available_information=["Add Regional Director"],
        existing_content="# 5. Stakeholders & Personas\n\n## 5.1 Personas\n- Ops Manager",
    )
    assert ctx_upd.operation == SectionOperation.UPDATE


def test_section_generation_context_serialization():
    """Verify SectionGenerationContext to_dict and from_dict roundtrip."""
    ctx = SectionGenerationContext(
        section_name="3. In-Scope Business Modules & Feature Groups",
        section_requirements=["List modules with IDs", "Assign business owners"],
        template_structure="| Module ID | Module Name |",
        available_information=["Inventory Module (HL-MOD-1), Owner: Sarah Chen"],
        existing_content=None,
        rework_feedback="Ensure table format is used",
        operation=SectionOperation.GENERATE,
        metadata={"project_id": "proj-123"},
    )

    data = ctx.to_dict()
    assert data["section_name"] == "3. In-Scope Business Modules & Feature Groups"
    assert data["operation"] == "generate"
    assert data["metadata"]["project_id"] == "proj-123"

    deserialized = SectionGenerationContext.from_dict(data)
    assert deserialized.section_name == ctx.section_name
    assert deserialized.operation == SectionOperation.GENERATE
    assert deserialized.rework_feedback == "Ensure table format is used"


def test_section_generation_result_properties_and_serialization():
    """Verify SectionGenerationResult properties and serialization."""
    res = SectionGenerationResult(
        section_name="5. Stakeholders & Personas",
        content="# 5. Stakeholders & Personas\n\nContent here",
        operation=SectionOperation.GENERATE,
        summary="Generated initial personas section",
        metadata={"duration_seconds": 1.25},
    )

    assert res.is_generation is True
    assert res.is_update is False

    data = res.to_dict()
    assert data["operation"] == "generate"

    restored = SectionGenerationResult.from_dict(data)
    assert restored.section_name == res.section_name
    assert restored.content == res.content
    assert restored.operation == SectionOperation.GENERATE
    assert restored.is_generation is True


# ---------------------------------------------------------------------------
# 4. Initial Section Generation Execution
# ---------------------------------------------------------------------------


def test_initial_section_generation_with_supplied_information():
    """Verify initial section generation produces structured markdown from supplied evidence."""
    expected_content = (
        "# 5. Stakeholders & Personas\n\n"
        "## 5.1 Personas\n\n"
        "- **Warehouse Manager (John Doe):** Oversees inventory audits and stock movements.\n"
        "- **Operations Lead (Jane Smith):** Analyzes logistics throughput and warehouse capacity.\n\n"
        "## 5.2 Persona–Module Mapping\n\n"
        "| Persona | Interacts With Modules | Notes |\n"
        "|---|---|---|\n"
        "| Warehouse Manager | HL-MOD-1 (Inventory) | Core audit workflow |\n"
        "| Operations Lead | HL-MOD-2 (Analytics) | Daily reporting |"
    )

    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "5. Stakeholders & Personas",
            "operation": "generate",
            "summary": "Generated complete stakeholders and personas section.",
            "content": expected_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    generator = BRDSectionGenerationAgent(model=model)

    ctx = SectionGenerationContext(
        section_name="5. Stakeholders & Personas",
        section_requirements=["Define personas", "Create persona-module mapping"],
        template_structure=extract_section_template("5. Stakeholders & Personas"),
        available_information=[
            "Personas: John Doe (Warehouse Manager) and Jane Smith (Operations Lead).",
            "John uses Inventory module (HL-MOD-1). Jane uses Analytics (HL-MOD-2).",
        ],
    )

    result = generator.generate(ctx)

    assert isinstance(result, SectionGenerationResult)
    assert result.section_name == "5. Stakeholders & Personas"
    assert result.is_generation is True
    assert "Warehouse Manager (John Doe)" in result.content
    assert "Persona–Module Mapping" in result.content
    assert len(model.invocations) == 1

    # Verify prompt contained the template structure and available info
    prompt_sent = str(model.invocations[0][-1].content)
    assert "5. Stakeholders & Personas" in prompt_sent
    assert "Initial Section Generation" in prompt_sent
    assert "John Doe" in prompt_sent


# ---------------------------------------------------------------------------
# 5. Section Update & Rework Execution
# ---------------------------------------------------------------------------


def test_section_update_incorporates_new_information_and_preserves_valid():
    """Verify updating a section preserves existing valid content while adding new information."""
    baseline_content = (
        "# 5. Stakeholders & Personas\n\n"
        "## 5.1 Personas\n\n"
        "- **Warehouse Manager (John Doe):** Oversees inventory audits."
    )

    updated_content = (
        "# 5. Stakeholders & Personas\n\n"
        "## 5.1 Personas\n\n"
        "- **Warehouse Manager (John Doe):** Oversees inventory audits.\n"
        "- **Quality Inspector (Alice Brown):** Performs sample checks on incoming goods."
    )

    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "5. Stakeholders & Personas",
            "operation": "update",
            "summary": "Added Quality Inspector persona while preserving Warehouse Manager.",
            "content": updated_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    generator = BRDSectionGenerationAgent(model=model)

    ctx = SectionGenerationContext(
        section_name="5. Stakeholders & Personas",
        section_requirements=["Capture all active personas"],
        template_structure=extract_section_template("5. Stakeholders & Personas"),
        available_information=["New role identified: Quality Inspector (Alice Brown)."],
        existing_content=baseline_content,
        operation=SectionOperation.UPDATE,
    )

    result = generator.generate(ctx)

    assert result.is_update is True
    assert "Warehouse Manager (John Doe)" in result.content
    assert "Quality Inspector (Alice Brown)" in result.content

    # Verify prompt contained existing content as baseline
    prompt_sent = str(model.invocations[0][-1].content)
    assert "Section Update / Rework" in prompt_sent
    assert "Current Section Content (Baseline to Update)" in prompt_sent
    assert "Warehouse Manager (John Doe)" in prompt_sent


def test_section_update_with_rework_feedback():
    """Verify rework feedback from section validation is supplied and addressed."""
    baseline_content = (
        "# 3. In-Scope Business Modules & Feature Groups\n\n"
        "- Module 1: Inventory Management\n"
        "- Module 2: Order Fulfillment"
    )

    corrected_content = (
        "# 3. In-Scope Business Modules & Feature Groups\n\n"
        "| Module ID (HL-MOD-X) | Module Name | Description (Business) | Business Owner |\n"
        "|---|---|---|---|\n"
        "| HL-MOD-1 | Inventory Management | Stock tracking and batch management | Sarah Chen |\n"
        "| HL-MOD-2 | Order Fulfillment | Pick, pack, and ship operations | Mark Davis |"
    )

    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "3. In-Scope Business Modules & Feature Groups",
            "operation": "update",
            "summary": "Restructured modules into standard Markdown table with IDs and owners per validation feedback.",
            "content": corrected_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    generator = BRDSectionGenerationAgent(model=model)

    rework_feedback = (
        "Validation Defect: Modules were formatted as bullet points instead of the required "
        "Markdown table. Also missing Module IDs (HL-MOD-X) and Business Owners."
    )

    ctx = SectionGenerationContext(
        section_name="3. In-Scope Business Modules & Feature Groups",
        section_requirements=["Structured table with Module ID, Name, Description, Owner"],
        template_structure=extract_section_template("3. In-Scope Business Modules & Feature Groups"),
        available_information=[
            "Module 1 is HL-MOD-1 (Inventory Management), Owner: Sarah Chen.",
            "Module 2 is HL-MOD-2 (Order Fulfillment), Owner: Mark Davis.",
        ],
        existing_content=baseline_content,
        rework_feedback=rework_feedback,
        operation=SectionOperation.UPDATE,
    )

    result = generator.generate(ctx)

    assert result.is_update is True
    assert "| Module ID (HL-MOD-X) |" in result.content
    assert "HL-MOD-1" in result.content
    assert "Sarah Chen" in result.content

    # Verify prompt contained rework feedback
    prompt_sent = str(model.invocations[0][-1].content)
    assert "Rework Feedback to Address" in prompt_sent
    assert "Validation Defect: Modules were formatted as bullet points" in prompt_sent


# ---------------------------------------------------------------------------
# 6. Fallback Parsing & Robustness Tests
# ---------------------------------------------------------------------------


def test_fallback_parsing_handles_raw_markdown_output():
    """Verify fallback parser handles models that output direct markdown without JSON wrapper."""
    raw_markdown = (
        "# 4. Out-of-Scope Items\n\n"
        "- Hardware procurement and barcode scanner hardware.\n"
        "- Customer-facing mobile application (deferred to Phase 2)."
    )

    # Model returns raw markdown string rather than JSON
    mock_llm_response = AIMessage(content=raw_markdown)

    model = MockChatModel(messages_to_return=[mock_llm_response])
    generator = BRDSectionGenerationAgent(model=model)

    ctx = SectionGenerationContext(
        section_name="4. Out-of-Scope Items",
        available_information=["Exclude hardware and mobile app"],
    )

    result = generator.generate(ctx)

    assert result.section_name == "4. Out-of-Scope Items"
    assert "Hardware procurement" in result.content
    assert "mobile application" in result.content
    assert result.metadata.get("fallback_parsing") is True


def test_generator_runtime_error_returns_safe_failure_result():
    """Verify model runtime exception is safely caught and returns error result without crashing."""
    model = MockChatModel(error_to_raise=RuntimeError("Provider API timeout"))
    generator = BRDSectionGenerationAgent(model=model)

    ctx = SectionGenerationContext(
        section_name="5. Stakeholders & Personas",
        existing_content="Existing baseline content",
    )

    result = generator.generate(ctx)

    assert "Provider API timeout" in result.summary
    assert "Provider API timeout" in result.metadata.get("error", "")
    # Preserved existing content on failure so state is not erased
    assert result.content == "Existing baseline content"


# ---------------------------------------------------------------------------
# 7. Asynchronous Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_section_generation():
    """Verify async generation works identically to sync."""
    expected_content = "# 9. Assumptions & Constraints\n\n- Standard 99.9% uptime required."
    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "9. Assumptions & Constraints",
            "operation": "generate",
            "summary": "Generated assumptions.",
            "content": expected_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    generator = BRDSectionGenerationAgent(model=model)

    ctx = SectionGenerationContext(
        section_name="9. Assumptions & Constraints",
        available_information=["99.9% uptime constraint"],
    )

    result = await generator.generate_async(ctx)

    assert result.section_name == "9. Assumptions & Constraints"
    assert "99.9% uptime" in result.content
    assert result.is_generation is True


# ---------------------------------------------------------------------------
# 8. BRD Lead Agent Integration Tests
# ---------------------------------------------------------------------------


def test_lead_agent_has_section_generator_property():
    """Verify BRDLeadAgent encapsulates section_generator capability."""
    model = MockChatModel()
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    assert agent.section_generator is not None
    assert isinstance(agent.section_generator, BRDSectionGenerationAgent)
    # Reuses Lead Agent model
    assert agent.section_generator.model is model


def test_lead_agent_generates_section_and_stores_in_state():
    """Verify Lead Agent generates section, updates current section, and stores content in State."""
    mock_content = (
        "# 5. Stakeholders & Personas\n\n"
        "## 5.1 Personas\n\n"
        "- **Warehouse Manager (John Doe):** Lead operator."
    )
    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "5. Stakeholders & Personas",
            "operation": "generate",
            "summary": "Initial generation of Section 5.",
            "content": mock_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Pre-populate state evidence
    agent.state.add_evidence({
        "source": "knowledge_retrieval",
        "content": "Warehouse Manager is John Doe.",
    })

    gen_result = agent.generate_section(
        section="5. Stakeholders & Personas",
    )

    # 1. Returned structured result
    assert isinstance(gen_result, SectionGenerationResult)
    assert gen_result.is_generation is True
    assert "Warehouse Manager (John Doe)" in gen_result.content

    # 2. Lead Agent stored section content in Agent State
    assert agent.state.get_section_content("5. Stakeholders & Personas") == mock_content
    # Matching clean or stripped name also returns content
    assert agent.state.get_section_content("Stakeholders & Personas") == mock_content

    # 3. Lead Agent updated section status and current section
    assert agent.state.current_section == "5. Stakeholders & Personas"
    assert agent.state.get_section_status("5. Stakeholders & Personas") == BRDSectionStatus.IN_PROGRESS

    # 4. Latest section result recorded in state
    assert agent.state.latest_section_result is gen_result


def test_lead_agent_rework_cycle():
    """Verify complete rework workflow: section exists -> rework feedback added -> update_section -> state updated."""
    initial_content = (
        "# 3. In-Scope Business Modules & Feature Groups\n\n"
        "- Module A: Inventory\n"
    )
    revised_content = (
        "# 3. In-Scope Business Modules & Feature Groups\n\n"
        "| Module ID (HL-MOD-X) | Module Name | Description (Business) | Business Owner |\n"
        "|---|---|---|---|\n"
        "| HL-MOD-1 | Inventory Management | Real-time tracking | Sarah Chen |"
    )

    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "3. In-Scope Business Modules & Feature Groups",
            "operation": "update",
            "summary": "Updated Section 3 to tabular structure per rework feedback.",
            "content": revised_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Set initial section content in state
    agent.state.set_section_content("3. In-Scope Business Modules & Feature Groups", initial_content)
    # Simulate Section Validation producing rework feedback
    agent.state.set_rework_feedback(
        "3. In-Scope Business Modules & Feature Groups",
        "Convert to Markdown table with Module ID and Business Owner.",
    )

    update_result = agent.update_section(
        section="3. In-Scope Business Modules & Feature Groups",
        new_information="Inventory is HL-MOD-1, Owner: Sarah Chen",
    )

    assert update_result.is_update is True
    assert "| Module ID (HL-MOD-X) |" in update_result.content
    assert "Sarah Chen" in update_result.content

    # State receives the updated content
    stored = agent.state.get_section_content("3. In-Scope Business Modules & Feature Groups")
    assert stored == revised_content

    # Rework feedback was cleared upon update
    assert agent.state.get_rework_feedback("3. In-Scope Business Modules & Feature Groups") is None



@pytest.mark.asyncio
async def test_lead_agent_async_generate_section():
    """Verify asynchronous section generation through Lead Agent."""
    mock_content = "# 4. Out-of-Scope Items\n\n- Excluded items"
    mock_llm_response = AIMessage(
        content=json.dumps({
            "section_name": "4. Out-of-Scope Items",
            "operation": "generate",
            "summary": "Generated out of scope.",
            "content": mock_content,
        })
    )

    model = MockChatModel(messages_to_return=[mock_llm_response])
    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    res = await agent.generate_section_async(
        section="4. Out-of-Scope Items",
        available_information=["No hardware"],
    )

    assert res.section_name == "4. Out-of-Scope Items"
    assert agent.state.get_section_content("4. Out-of-Scope Items") == mock_content


# ---------------------------------------------------------------------------
# 9. Multi-Section Iterative Workflow Lifecycle
# ---------------------------------------------------------------------------


def test_multi_section_iterative_lifecycle():
    """Verify capability can be invoked repeatedly across multiple sections and rework cycles."""
    model = MockChatModel(messages_to_return=[
        # 1. Section 1 Generate
        AIMessage(content=json.dumps({
            "section_name": "1. Purpose & Scope of This Document",
            "operation": "generate",
            "summary": "Generated Section 1",
            "content": "# 1. Purpose & Scope\n\nInitial scope statement.",
        })),
        # 2. Section 2 Generate
        AIMessage(content=json.dumps({
            "section_name": "2. Business Context (Summary from Discovery)",
            "operation": "generate",
            "summary": "Generated Section 2",
            "content": "# 2. Business Context\n\nContext narrative.",
        })),
        # 3. Section 2 Update/Rework
        AIMessage(content=json.dumps({
            "section_name": "2. Business Context (Summary from Discovery)",
            "operation": "update",
            "summary": "Updated Section 2 with discovery link",
            "content": "# 2. Business Context\n\nContext narrative.\n\nLink: DISC-2026-001.",
        })),
    ])

    config = AgentConfig(model="test-model", api_key="test-key")
    agent = BRDLeadAgent(config=config, model=model)

    # Step 1: Generate Section 1
    res1 = agent.generate_section(section="1. Purpose & Scope of This Document", available_information="Scope notes")
    assert res1.is_generation is True
    assert agent.state.get_section_content("1. Purpose & Scope of This Document") == "# 1. Purpose & Scope\n\nInitial scope statement."

    # Step 2: Generate Section 2
    res2 = agent.generate_section(section="2. Business Context (Summary from Discovery)", available_information="Discovery summary")
    assert res2.is_generation is True
    assert agent.state.get_section_content("2. Business Context (Summary from Discovery)") == "# 2. Business Context\n\nContext narrative."

    # Step 3: Rework Section 2
    res2_updated = agent.update_section(
        section="2. Business Context (Summary from Discovery)",
        new_information="Discovery document ID is DISC-2026-001",
        rework_feedback="Add link to discovery notes.",
    )
    assert res2_updated.is_update is True
    assert "DISC-2026-001" in res2_updated.content
    assert agent.state.get_section_content("2. Business Context (Summary from Discovery)") == "# 2. Business Context\n\nContext narrative.\n\nLink: DISC-2026-001."

    # Section 1 content was preserved untouched
    assert "Initial scope statement." in agent.state.get_section_content("1. Purpose & Scope of This Document")
