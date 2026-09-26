"""Project-scoped long-term memory implementation for BRD Lead Agent.

Integrates LangGraph BaseStore and DeepAgents MemoryMiddleware to persist and inject
authoritative project context under strict project boundary isolation.
"""

from typing import Any, Optional, Sequence

from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse
from deepagents.backends.store import NamespaceFactory, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.memory import MemoryMiddleware
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from agents.brd.context import get_current_agent_context
from observability.logging import get_logger

logger = get_logger(__name__)

# Default canonical project memory file path
DEFAULT_PROJECT_MEMORY_FILE = "/memory/project_context.md"

# Authoritative read-only memory instructions injected into the system prompt prefix
READONLY_MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
The above <agent_memory> contains authoritative persistent project context provided by the application.
This memory persists across separate runs and conversations for the current project.
Authoritative memory writes are controlled exclusively by the application.
You may read and utilize this context to inform your work and maintain continuity across project sessions.
</memory_guidelines>
"""

_default_memory_store: Optional[BaseStore] = None


def get_default_memory_store() -> BaseStore:
    """Get or initialize the shared singleton LangGraph BaseStore instance.

    Uses LangGraph's documented InMemoryStore as the default persistence store,
    maintaining cross-run and cross-conversation project memory across the application lifecycle.

    Returns:
        BaseStore: Active shared store instance.
    """
    global _default_memory_store
    if _default_memory_store is None:
        logger.info("Initializing application default InMemoryStore for long-term memory")
        _default_memory_store = InMemoryStore()
    return _default_memory_store


def reset_default_memory_store() -> None:
    """Reset the singleton store instance.

    Primarily used for deterministic test isolation between test cases.
    """
    global _default_memory_store
    _default_memory_store = None


def normalize_memory_path(path: str) -> str:
    """Normalize memory file paths to POSIX standard starting with /memory/ if relative.

    Args:
        path: Raw memory key or file path.

    Returns:
        Normalized POSIX path (e.g., '/memory/project_context.md').
    """
    p = path.strip().replace("\\", "/")
    if not p.startswith("/"):
        return f"/memory/{p}"
    return p


def create_project_namespace_factory(
    bound_project_id: Optional[str] = None,
    agent: Optional[Any] = None,
) -> NamespaceFactory:
    """Create a NamespaceFactory callable resolving project-scoped memory namespaces.

    Enforces the documented LangGraph tuple namespace structure:
        ("projects", <project_id>, "memory")

    Resolution precedence for <project_id>:
    1. Active AgentContext in contextvars (set during agent execution)
    2. Explicitly bound project_id supplied at factory creation
    3. Associated agent's project_id or state metadata
    4. Fallback 'default' tenant
    """
    def namespace_factory(_runtime: Any = None) -> tuple[str, ...]:
        ctx = get_current_agent_context()
        pid: Optional[str] = None

        if ctx and ctx.project_id:
            pid = ctx.project_id
        elif bound_project_id:
            pid = bound_project_id
        elif agent is not None:
            pid = getattr(agent, "project_id", None)
            if not pid and hasattr(agent, "state") and getattr(agent, "state", None) is not None:
                metadata = getattr(agent.state, "metadata", {})
                if isinstance(metadata, dict):
                    pid = metadata.get("project_id")

        resolved = pid or "default"
        return ("projects", str(resolved), "memory")

    return namespace_factory


class ProjectMemoryStoreBackend(StoreBackend):
    """DeepAgents StoreBackend adapter for LangGraph BaseStore with fail-safe error handling and lifecycle logging."""

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download memory files with fail-safe error handling to prevent crashing agent execution on store read error.

        Args:
            paths: Sequence of file paths to download.

        Returns:
            List of FileDownloadResponse objects.
        """
        try:
            responses = super().download_files(paths)
            logger.debug("Downloaded memory files %s (count: %d)", paths, len(responses))
            return responses
        except Exception as exc:
            logger.error(
                "Failed to download project memory files %s from store: %s",
                paths,
                exc,
                exc_info=True,
            )
            return [FileDownloadResponse(path=p, content=None, error="file_not_found") for p in paths]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Asynchronously download memory files with fail-safe error handling.

        Args:
            paths: Sequence of file paths to download.

        Returns:
            List of FileDownloadResponse objects.
        """
        try:
            responses = await super().adownload_files(paths)
            logger.debug("Async downloaded memory files %s (count: %d)", paths, len(responses))
            return responses
        except Exception as exc:
            logger.error(
                "Async failed to download project memory files %s from store: %s",
                paths,
                exc,
                exc_info=True,
            )
            return [FileDownloadResponse(path=p, content=None, error="file_not_found") for p in paths]


def create_readonly_memory_permissions() -> list[FilesystemPermission]:
    """Construct FilesystemPermission rules enforcing read-only access for the agent.

    Denies write, edit, and delete operations across all filesystem paths,
    ensuring that only the application retains authoritative write permissions.

    Returns:
        List containing the deny-write FilesystemPermission rule.
    """
    return [FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")]


def create_readonly_memory_middleware(
    backend: BackendProtocol,
    sources: Sequence[str],
    system_prompt: Optional[str] = READONLY_MEMORY_SYSTEM_PROMPT,
) -> MemoryMiddleware:
    """Construct DeepAgents MemoryMiddleware configured with read-only project memory guidelines.

    Args:
        backend: Storage backend implementing BackendProtocol (e.g., ProjectMemoryStoreBackend).
        sources: Sequence of memory file paths to preload (e.g., ['/memory/project_context.md']).
        system_prompt: System prompt template containing the `{agent_memory}` substitution slot.

    Returns:
        Configured MemoryMiddleware instance.
    """
    return MemoryMiddleware(
        backend=backend,
        sources=list(sources),
        system_prompt=system_prompt,
        add_cache_control=True,
    )


def create_project_memory_backend(
    store: Optional[BaseStore] = None,
    project_id: Optional[str] = None,
    agent: Optional[Any] = None,
) -> ProjectMemoryStoreBackend:
    """Factory creating a ProjectMemoryStoreBackend configured for project-scoped memory.

    Args:
        store: Optional LangGraph BaseStore instance. If omitted, uses get_default_memory_store().
        project_id: Optional project identifier bound to this backend.
        agent: Optional agent instance for dynamic project resolution.

    Returns:
        Configured ProjectMemoryStoreBackend instance.
    """
    resolved_store = store if store is not None else get_default_memory_store()
    ns_factory = create_project_namespace_factory(bound_project_id=project_id, agent=agent)
    return ProjectMemoryStoreBackend(namespace=ns_factory, store=resolved_store)
