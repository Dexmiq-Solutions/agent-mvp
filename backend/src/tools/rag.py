"""RAG Retrieval Tool connecting the Agent runtime to the existing RAG service.

Enforces strict architectural boundaries:
- The LLM supplies only the search query.
- The project boundary (project_id) is supplied strictly by the application runtime (AgentContext).
- Retrieval orchestration remains inside the existing RAG service.
- Reasoning and synthesis remain inside the Agent.
"""

import asyncio
import concurrent.futures
import contextvars
from typing import Any, Optional
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from agents.runtime.state import get_current_agent_context
from observability.logging import get_logger
from services.rag_service import RAGService, RetrievalResult, get_rag_service

logger = get_logger(__name__)


class SearchProjectKnowledgeInput(BaseModel):
    """Input payload schema for search_project_knowledge tool."""

    query: str = Field(
        ...,
        description=(
            "Focused search query describing the needed project requirements, "
            "architecture, specifications, business guidelines, or domain facts."
        ),
    )


async def _execute_knowledge_search(
    query: str,
    project_id: Optional[str] = None,
    rag_service: Optional[RAGService] = None,
) -> str:
    """Core asynchronous execution pipeline for project knowledge retrieval.

    Strictly enforces project isolation:
    1. Resolves project_id from trusted execution context or fixed tool configuration.
    2. Invokes the existing RAG retrieval service without reimplementing retrieval.
    3. Formats the outcome into unambiguous, model-readable evidence preserving distinctions
       between successful retrieval, no evidence found, and system retrieval failure.

    Args:
        query: Knowledge query submitted by the agent.
        project_id: Optional fixed project ID override. Defaults to active AgentContext.
        rag_service: Optional injected RAGService instance.

    Returns:
        Structured string representation of retrieved evidence, no-evidence status, or failure.
    """
    # 1. Enforce Project Isolation Boundary
    eff_project_id = project_id
    if not eff_project_id:
        ctx = get_current_agent_context()
        if ctx and ctx.project_id:
            eff_project_id = ctx.project_id

    if not eff_project_id or not str(eff_project_id).strip():
        logger.warning(
            "search_project_knowledge rejected: Missing active project context (project_id)"
        )
        return (
            "[RETRIEVAL_ERROR] Project boundary violation: No active project context. "
            "A valid project_id must be provided by the application runtime."
        )

    clean_project_id = str(eff_project_id).strip()
    clean_query = query.strip() if query else ""

    if not clean_query:
        logger.warning(
            "search_project_knowledge rejected: Empty query for project '%s'",
            clean_project_id,
        )
        return "[RETRIEVAL_ERROR] Invalid query: Search query cannot be empty."

    logger.info(
        "RAG tool invocation: searching project knowledge (project_id: %s, query_len: %d)",
        clean_project_id,
        len(clean_query),
    )

    # 2. Delegate to Existing RAG Service
    try:
        service = rag_service or get_rag_service()
        result: RetrievalResult = await service.retrieve(
            project_id=clean_project_id,
            query=clean_query,
        )
    except Exception as exc:
        logger.error(
            "RAG tool retrieval failed for project '%s': %s",
            clean_project_id,
            exc,
            exc_info=True,
        )
        return f"[RETRIEVAL_ERROR] Retrieval failed for query '{clean_query}': {exc}"

    # 3. Outcome: No Evidence
    if result.is_empty or not result.chunks:
        logger.info(
            "RAG tool retrieval found zero matching chunks (project_id: %s, query: '%s')",
            clean_project_id,
            clean_query,
        )
        return (
            f"[NO_EVIDENCE] Retrieval executed successfully, but no relevant project "
            f"documents or knowledge were found for query: '{clean_query}'."
        )

    # 4. Outcome: Successful Retrieval
    logger.info(
        "RAG tool retrieval succeeded (project_id: %s, chunks: %d)",
        clean_project_id,
        len(result.chunks),
    )
    formatted_context_text = result.formatted_text
    return (
        f"[RETRIEVAL_SUCCESS] Found {len(result.chunks)} relevant knowledge items "
        f"for query '{clean_query}':\n\n"
        f"{formatted_context_text}"
    )


def create_search_project_knowledge_tool(
    project_id: Optional[str] = None,
    rag_service: Optional[RAGService] = None,
) -> StructuredTool:
    """Factory creating an Agent-facing StructuredTool for searching project knowledge.

    Supports both synchronous and asynchronous agent execution loops.
    Preserves project isolation by resolving project_id from runtime AgentContext unless fixed.

    Args:
        project_id: Optional fixed project ID override.
        rag_service: Optional pre-configured RAGService instance.

    Returns:
        StructuredTool configured for LangChain and DeepAgents execution.
    """

    async def _async_search(query: str) -> str:
        return await _execute_knowledge_search(
            query=query,
            project_id=project_id,
            rag_service=rag_service,
        )

    def _sync_search(query: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(ctx.run, asyncio.run, _async_search(query)).result()
        else:
            return asyncio.run(_async_search(query))

    return StructuredTool.from_function(
        func=_sync_search,
        coroutine=_async_search,
        name="search_project_knowledge",
        description=(
            "Search project documentation, requirements, architecture, and specifications. "
            "Use this tool whenever you need specific project context, domain knowledge, "
            "business requirements, architectural guidelines, or verified facts about the current project. "
            "Provide a focused search query describing the needed knowledge."
        ),
        args_schema=SearchProjectKnowledgeInput,
    )


# Default singleton instance for standard runtime equipping
search_project_knowledge = create_search_project_knowledge_tool()

__all__ = [
    "SearchProjectKnowledgeInput",
    "create_search_project_knowledge_tool",
    "search_project_knowledge",
]
