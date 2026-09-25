"""Automated architectural boundary and dependency direction verification tests.

Validates that:
1. LLM is a root-level platform capability and does not import RAG, App, or FastAPI.
2. Core is foundational and does not import application, domain, or delivery layers.
3. Observability is foundational and does not import RAG, LLM, or App.
4. Storage, DB, and RAG do not import App (no lower-level -> delivery layer coupling).
5. RAG focuses strictly on knowledge retrieval and returns RetrievalResult without generation coupling.
6. No stale import paths (app.core, rag.generation) exist across the codebase.
"""

import ast
from pathlib import Path
import re
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
ALEMBIC_ROOT = Path(__file__).resolve().parents[1] / "alembic"


def get_imported_modules(py_file_path: Path) -> set[str]:
    """Parse a python source file into an AST and extract all top-level module names imported."""
    with open(py_file_path, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read(), filename=str(py_file_path))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    return imported


def get_all_raw_import_strings(py_file_path: Path) -> list[str]:
    """Extract full import paths from a python source file."""
    with open(py_file_path, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read(), filename=str(py_file_path))

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_llm_does_not_import_rag():
    """Verify that src/llm/ does not import from rag at all."""
    llm_dir = SRC_ROOT / "llm"
    assert llm_dir.is_dir(), "src/llm/ directory must exist"

    violations = []
    for py_file in llm_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        if "rag" in imported:
            violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: src/llm/ files import 'rag': {violations}"


def test_llm_does_not_import_app():
    """Verify that src/llm/ does not import from app at all."""
    llm_dir = SRC_ROOT / "llm"
    violations = []
    for py_file in llm_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        if "app" in imported:
            violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: src/llm/ files import 'app': {violations}"


def test_llm_does_not_import_fastapi_or_starlette():
    """Verify that src/llm/ does not depend on HTTP/web delivery frameworks."""
    llm_dir = SRC_ROOT / "llm"
    forbidden = {"fastapi", "starlette"}
    violations = []
    for py_file in llm_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        overlap = imported & forbidden
        if overlap:
            violations.append(f"{py_file.relative_to(SRC_ROOT)}: {overlap}")

    assert not violations, f"Architectural violation: src/llm/ imports web frameworks: {violations}"


def test_core_does_not_import_higher_layers():
    """Verify that src/core/ is foundational and does not depend on other subsystems."""
    core_dir = SRC_ROOT / "core"
    assert core_dir.is_dir(), "src/core/ directory must exist"

    forbidden = {"rag", "llm", "app", "storage", "db", "services", "models", "schemas", "api"}
    violations = []
    for py_file in core_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        overlap = imported & forbidden
        if overlap:
            violations.append(f"{py_file.relative_to(SRC_ROOT)}: {overlap}")

    assert not violations, f"Architectural violation: src/core/ imports higher layers: {violations}"


def test_observability_does_not_import_subsystems():
    """Verify that src/observability/ does not depend on RAG, LLM, or App."""
    obs_dir = SRC_ROOT / "observability"
    assert obs_dir.is_dir(), "src/observability/ directory must exist"

    forbidden = {"rag", "llm", "app", "services", "storage", "api"}
    violations = []
    for py_file in obs_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        overlap = imported & forbidden
        if overlap:
            violations.append(f"{py_file.relative_to(SRC_ROOT)}: {overlap}")

    assert not violations, f"Architectural violation: src/observability/ imports forbidden layers: {violations}"


def test_lower_layers_do_not_import_app():
    """Verify that storage, db, and rag do not import from app."""
    subsystems = [SRC_ROOT / "storage", SRC_ROOT / "db", SRC_ROOT / "rag"]
    violations = []
    for subsystem in subsystems:
        for py_file in subsystem.rglob("*.py"):
            imported = get_imported_modules(py_file)
            if "app" in imported:
                violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: lower layers import 'app': {violations}"


def test_no_stale_import_paths_in_src_and_alembic():
    """Verify that no code in src/ or alembic/ imports from retired generation modules or app.core."""
    pattern = re.compile(r"\b(app\.core|rag\.generation|services\.generation_service|schemas\.generation|exceptions\.generation)\b")
    violations = []
    for target_dir in [SRC_ROOT, ALEMBIC_ROOT]:
        for py_file in target_dir.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, 1):
                    if pattern.search(line):
                        violations.append(f"{py_file}:{idx}: {line.strip()}")

    assert not violations, f"Found stale import paths: {violations}"


def test_rag_generation_directory_retired():
    """Verify that obsolete RAG generation package has been completely removed."""
    gen_dir = SRC_ROOT / "rag" / "generation"
    assert not gen_dir.exists(), "src/rag/generation/ directory must be completely retired"



def test_canonical_llm_exports():
    """Verify that src/llm exposes the canonical public interface."""
    import llm
    from llm import (
        BaseLLMInterface,
        LLMConfig,
        LLMResult,
        LLMService,
        LLMStreamEvent,
        LLMUsage,
        OpenAICompatibleLLMAdapter,
        generate_async,
        generate_stream_async,
        get_llm_service,
        reset_llm_service,
    )

    assert issubclass(OpenAICompatibleLLMAdapter, BaseLLMInterface)
    config = LLMConfig()
    assert config.model == "gpt-4o"
    assert config.timeout == 30.0


def test_canonical_core_exports():
    """Verify that src/core exposes canonical configuration."""
    from core import Settings, get_settings
    from core.config import Settings as ConfigSettings

    assert Settings is ConfigSettings
    settings = get_settings()
    assert settings.APP_NAME == "Agent MVP"


def test_canonical_observability_exports():
    """Verify that src/observability exposes canonical logging."""
    from observability import get_logger, setup_logging
    from observability.logging import get_logger as LoggingGetLogger

    assert get_logger is LoggingGetLogger
    logger = get_logger("test_arch_logger")
    assert logger.name == "test_arch_logger"


def test_agents_does_not_import_rag():
    """Verify that src/agents/ does not import from rag (Agent/RAG boundary preserved in Phase 2)."""
    agents_dir = SRC_ROOT / "agents"
    assert agents_dir.is_dir(), "src/agents/ directory must exist"

    violations = []
    for py_file in agents_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        if "rag" in imported:
            violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: src/agents/ files import 'rag': {violations}"


def test_agents_does_not_import_app():
    """Verify that src/agents/ does not import from app."""
    agents_dir = SRC_ROOT / "agents"
    violations = []
    for py_file in agents_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        if "app" in imported:
            violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: src/agents/ files import 'app': {violations}"


def test_rag_does_not_import_agents():
    """Verify that src/rag/ does not import from agents (RAG remains decoupled)."""
    rag_dir = SRC_ROOT / "rag"
    violations = []
    for py_file in rag_dir.rglob("*.py"):
        imported = get_imported_modules(py_file)
        if "agents" in imported:
            violations.append(str(py_file.relative_to(SRC_ROOT)))

    assert not violations, f"Architectural violation: src/rag/ files import 'agents': {violations}"


def test_canonical_agent_exports():
    """Verify that src/agents exposes canonical public interfaces."""
    import agents
    from agents import (
        AgentConfig,
        AgentContext,
        AgentRunRequest,
        AgentRunResponse,
        AgentRuntime,
        BRDLeadAgent,
        create_agent_model,
        create_brd_lead_agent,
        create_runtime_agent,
    )

    assert AgentConfig is not None
    assert AgentRuntime is not None
    assert AgentContext is not None
    assert AgentRunRequest is not None
    assert AgentRunResponse is not None
    assert BRDLeadAgent is not None
    assert callable(create_agent_model)
    assert callable(create_brd_lead_agent)
    assert callable(create_runtime_agent)



def test_canonical_tools_exports():
    """Verify that src/tools exposes deterministic tools."""
    import tools
    from tools import echo_diagnostic_tool

    assert echo_diagnostic_tool is not None
    assert callable(echo_diagnostic_tool.invoke)
