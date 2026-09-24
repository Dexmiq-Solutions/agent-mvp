"""Agent tools package."""

from tools.diagnostic import echo_diagnostic_tool
from tools.rag import (
    SearchProjectKnowledgeInput,
    create_search_project_knowledge_tool,
    search_project_knowledge,
)


def get_default_tools() -> list:
    """Return the default suite of registered agent tools."""
    return [echo_diagnostic_tool, search_project_knowledge]


__all__ = [
    "echo_diagnostic_tool",
    "search_project_knowledge",
    "create_search_project_knowledge_tool",
    "SearchProjectKnowledgeInput",
    "get_default_tools",
]
