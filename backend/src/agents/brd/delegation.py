"""Delegation and temporary sub-agent execution subsystem for BRD Lead Agent.

Provides bounded task decomposition, temporary task-scoped sub-agent execution,
task result attribution, result collection, and convergence onto the common
ActionResult architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import time
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deepagents import create_deep_agent
from agents.brd.context import (
    ActionResult,
    ActionSource,
    AgentContext,
    reset_current_agent_context,
    set_current_agent_context,
)
from observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DelegatedTask:
    """Bounded, task-scoped unit of work decomposed from the Lead Agent objective.

    Attributes:
        task_id: Unique identifier for the task (e.g. task-1, sec-3-func-req).
        objective: Clear objective statement for this specific task.
        relevant_context: Scoped input / context necessary to execute the task.
        scope: Boundaries and explicit remit of the task.
        constraints: Operational or business constraints governing the task.
        expected_output: Description of the required result / deliverable.
        metadata: Optional task-specific metadata.
    """

    task_id: str
    objective: str
    relevant_context: str = ""
    scope: str = ""
    constraints: str = ""
    expected_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegatedTask":
        """Deserialize a dictionary into a DelegatedTask instance."""
        return cls(
            task_id=str(data.get("task_id", "unspecified-task")),
            objective=str(data.get("objective", "")),
            relevant_context=str(data.get("relevant_context", "")),
            scope=str(data.get("scope", "")),
            constraints=str(data.get("constraints", "")),
            expected_output=str(data.get("expected_output", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TaskResult:
    """Attributable result produced by a temporary task-scoped sub-agent execution.

    Attributes:
        task_id: Identifier of the delegated task that produced this result.
        objective: What the task attempted to accomplish.
        content: The resulting output/content produced by the sub-agent.
        success: Whether the task completed successfully.
        execution_info: Relevant runtime information (e.g. duration, model, tool_calls).
        error: Error details if the task execution failed.
        metadata: Optional extensible metadata.
    """

    task_id: str
    objective: str
    content: str
    success: bool = True
    execution_info: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize TaskResult to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        """Deserialize a dictionary into a TaskResult instance."""
        return cls(
            task_id=str(data.get("task_id", "")),
            objective=str(data.get("objective", "")),
            content=str(data.get("content", "")),
            success=bool(data.get("success", True)),
            execution_info=dict(data.get("execution_info", {})),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DelegationResult:
    """Collected result synthesizing multiple task-level results.

    Attributes:
        content: Coherent aggregated / synthesized result text from all task executions.
        task_results: Individual attributable task results.
        success: Overall success status (True if all tasks succeeded).
        total_tasks: Number of delegated tasks.
        completed_tasks: Number of successfully completed tasks.
        failed_tasks: Number of failed tasks.
        metadata: Delegation metadata.
    """

    content: str
    task_results: list[TaskResult] = field(default_factory=list)
    success: bool = True
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize DelegationResult to a dictionary."""
        return {
            "content": self.content,
            "task_results": [r.to_dict() for r in self.task_results],
            "success": self.success,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationResult":
        """Deserialize a dictionary into a DelegationResult instance."""
        results = [
            TaskResult.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("task_results", [])
        ]
        return cls(
            content=data.get("content", ""),
            task_results=results,
            success=data.get("success", True),
            total_tasks=data.get("total_tasks", len(results)),
            completed_tasks=data.get("completed_tasks", sum(1 for r in results if r.success)),
            failed_tasks=data.get("failed_tasks", sum(1 for r in results if not r.success)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_action_result(self, context: Optional[AgentContext] = None) -> ActionResult:
        """Convert this collected DelegationResult into a unified ActionResult."""
        ctx = context or AgentContext()
        return ActionResult(
            source=ActionSource.DELEGATION,
            content=self.content,
            context=ctx,
            success=self.success,
            error=None if self.success else f"{self.failed_tasks} of {self.total_tasks} delegated tasks failed",
            metadata={
                "task_results": [r.to_dict() for r in self.task_results],
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                **self.metadata,
            },
        )


def _build_subagent_system_prompt(task: DelegatedTask) -> str:
    """Build a scoped, minimal system prompt for a temporary sub-agent execution."""
    sections = [
        "You are a temporary, task-scoped execution sub-agent assisting the BRD Lead Agent.",
        "Your execution remit is strictly bounded to the assigned task. You must not attempt to delegate,",
        "create sub-agents, or perform work outside this assigned scope.",
        "",
        f"Task Objective: {task.objective}",
    ]
    if task.scope:
        sections.append(f"Task Scope: {task.scope}")
    if task.constraints:
        sections.append(f"Task Constraints: {task.constraints}")
    if task.expected_output:
        sections.append(f"Expected Output: {task.expected_output}")

    return "\n".join(sections)


def _build_subagent_task_input(task: DelegatedTask) -> str:
    """Format input string containing only the minimum necessary context for the task."""
    parts = [f"Assigned Task: {task.objective}"]
    if task.relevant_context:
        parts.append(f"Relevant Context:\n{task.relevant_context}")
    if task.expected_output:
        parts.append(f"Expected Output:\n{task.expected_output}")
    return "\n\n".join(parts)


def execute_subagent_task(
    task: DelegatedTask,
    model: BaseChatModel,
    parent_context: Optional[AgentContext] = None,
    tools: Optional[Sequence[Any]] = None,
) -> TaskResult:
    """Execute a single delegated task via a temporary task-scoped DeepAgents harness.

    Enforces:
    1. Task scoping: receives only the minimum required context.
    2. Project isolation: inherits tenant project_id and cannot override it.
    3. Tool scoping: receives only approved tools; recursive delegation is disallowed.
    4. Attributable result: produces a structured TaskResult with execution metrics.

    Args:
        task: Bounded DelegatedTask to execute.
        model: BaseChatModel instance for the temporary sub-agent.
        parent_context: Context preserving tenant identity (project_id).
        tools: Optional sequence of tools (sub-agents do not receive delegation tools).

    Returns:
        TaskResult: Attributable execution result.
    """
    start_time = time.perf_counter()
    ctx = parent_context or AgentContext()
    # Ensure tenant project isolation: sub-agent cannot choose or alter project_id
    task_context = AgentContext(
        project_id=ctx.project_id,
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        metadata={"delegated_task_id": task.task_id},
    )
    token = set_current_agent_context(task_context)

    logger.info(
        "Sub-agent task execution started (task_id: %s, project_id: %s, objective: %s)",
        task.task_id,
        task_context.project_id or "unspecified",
        task.objective,
    )

    try:
        subagent_prompt = _build_subagent_system_prompt(task)
        task_input = _build_subagent_task_input(task)
        # Give sub-agent only required capabilities; no recursive delegation
        subagent_tools = list(tools) if tools else []

        subagent_graph = create_deep_agent(
            model=model,
            tools=subagent_tools,
            system_prompt=subagent_prompt,
        )

        result = subagent_graph.invoke({"messages": [HumanMessage(content=task_input)]})

        duration = time.perf_counter() - start_time
        messages = result.get("messages", [])
        output_text = ""
        tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc)

        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", "")
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                output_text = "".join(text_parts)
            else:
                output_text = str(content)

        logger.info(
            "Sub-agent task execution succeeded (task_id: %s, duration: %.2fs, output_len: %d)",
            task.task_id,
            duration,
            len(output_text),
        )

        return TaskResult(
            task_id=task.task_id,
            objective=task.objective,
            content=output_text,
            success=True,
            execution_info={
                "duration_seconds": duration,
                "tool_calls_count": len(tool_calls),
            },
        )
    except Exception as exc:
        duration = time.perf_counter() - start_time
        err_msg = str(exc).strip() or exc.__class__.__name__
        logger.error(
            "Sub-agent task execution failed (task_id: %s, duration: %.2fs): %s",
            task.task_id,
            duration,
            err_msg,
            exc_info=True,
        )
        return TaskResult(
            task_id=task.task_id,
            objective=task.objective,
            content="",
            success=False,
            execution_info={"duration_seconds": duration},
            error=err_msg,
        )
    finally:
        reset_current_agent_context(token)


async def execute_subagent_task_async(
    task: DelegatedTask,
    model: BaseChatModel,
    parent_context: Optional[AgentContext] = None,
    tools: Optional[Sequence[Any]] = None,
) -> TaskResult:
    """Execute a single delegated task asynchronously via a temporary DeepAgents harness."""
    start_time = time.perf_counter()
    ctx = parent_context or AgentContext()
    task_context = AgentContext(
        project_id=ctx.project_id,
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        metadata={"delegated_task_id": task.task_id},
    )
    token = set_current_agent_context(task_context)

    logger.info(
        "Sub-agent task execution started [async] (task_id: %s, project_id: %s, objective: %s)",
        task.task_id,
        task_context.project_id or "unspecified",
        task.objective,
    )

    try:
        subagent_prompt = _build_subagent_system_prompt(task)
        task_input = _build_subagent_task_input(task)
        subagent_tools = list(tools) if tools else []

        subagent_graph = create_deep_agent(
            model=model,
            tools=subagent_tools,
            system_prompt=subagent_prompt,
        )

        result = await subagent_graph.ainvoke({"messages": [HumanMessage(content=task_input)]})

        duration = time.perf_counter() - start_time
        messages = result.get("messages", [])
        output_text = ""
        tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc)

        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", "")
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                output_text = "".join(text_parts)
            else:
                output_text = str(content)

        logger.info(
            "Sub-agent task execution succeeded [async] (task_id: %s, duration: %.2fs)",
            task.task_id,
            duration,
        )

        return TaskResult(
            task_id=task.task_id,
            objective=task.objective,
            content=output_text,
            success=True,
            execution_info={
                "duration_seconds": duration,
                "tool_calls_count": len(tool_calls),
            },
        )
    except Exception as exc:
        duration = time.perf_counter() - start_time
        err_msg = str(exc).strip() or exc.__class__.__name__
        logger.error(
            "Sub-agent task execution failed [async] (task_id: %s, duration: %.2fs): %s",
            task.task_id,
            duration,
            err_msg,
            exc_info=True,
        )
        return TaskResult(
            task_id=task.task_id,
            objective=task.objective,
            content="",
            success=False,
            execution_info={"duration_seconds": duration},
            error=err_msg,
        )
    finally:
        reset_current_agent_context(token)


def collect_task_results(
    task_results: Sequence[TaskResult],
    overall_objective: Optional[str] = None,
) -> DelegationResult:
    """Collect individual attributable task results into one coherent DelegationResult.

    Preserves task-level attribution while generating a coherent unified downstream output.

    Args:
        task_results: Sequence of TaskResult instances from executed sub-agents.
        overall_objective: Optional overall Lead Agent objective to title the result.

    Returns:
        DelegationResult: Consolidated delegation-level result.
    """
    results_list = list(task_results)
    total = len(results_list)
    completed = sum(1 for r in results_list if r.success)
    failed = total - completed
    overall_success = (failed == 0)

    # Build structured, unified delegation content preserving attribution
    lines: list[str] = []
    title = overall_objective or "Delegated Task Execution"
    lines.append(f"# Delegation Result: {title}")
    lines.append("")
    lines.append(f"**Execution Summary**: {completed}/{total} tasks completed successfully.")
    lines.append("")
    lines.append("## Task Attributions & Deliverables")
    lines.append("")

    for res in results_list:
        status_tag = "COMPLETED" if res.success else "FAILED"
        lines.append(f"### [{res.task_id}] {res.objective} ({status_tag})")
        if res.success:
            lines.append(res.content)
        else:
            lines.append(f"*Execution Error*: {res.error or 'Unknown task execution error'}")
        lines.append("")

    content = "\n".join(lines).strip()

    return DelegationResult(
        content=content,
        task_results=results_list,
        success=overall_success,
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=failed,
        metadata={"overall_objective": title},
    )


def decompose_objective(
    objective: str,
    context: Optional[str] = None,
    model: Optional[BaseChatModel] = None,
    tasks_hint: Optional[int] = None,
) -> list[DelegatedTask]:
    """Decompose an objective dynamically into an appropriate number of bounded DelegatedTasks.

    Supports dynamic task decomposition (2, 3, 5, etc.) based on objective requirements.
    Can utilize LLM reasoning if model is provided, or structured rule-based decomposition.

    Args:
        objective: The overarching objective or composite request to decompose.
        context: Optional relevant context or specifications.
        model: Optional BaseChatModel for cognitive decomposition.
        tasks_hint: Optional hint for expected number of tasks.

    Returns:
        list[DelegatedTask]: Dynamic list of bounded tasks.
    """
    # Check if objective explicitly defines structured numbered or bulleted items
    bullet_items = re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s*(.+?)(?=(?:\n\s*(?:\d+[\.\)]|[-*•]))|\Z)",
        objective,
        re.DOTALL,
    )
    if len(bullet_items) >= 2:
        return _heuristic_decompose(objective, context, tasks_hint)

    if model is not None:
        try:
            decomposition_prompt = (
                "You are an expert systems analyst assisting the BRD Lead Agent in task decomposition.\n"
                "Decompose the following objective into an appropriate, dynamic number of bounded tasks (e.g. 2, 3, 5, etc.).\n"
                "Do not fix a rigid count; choose the number of tasks that best fits the work.\n\n"
                f"Objective: {objective}\n"
                f"Context: {context or 'None provided'}\n\n"
                "Each task must have:\n"
                "- task_id (e.g. task-1, task-2)\n"
                "- objective (clear bounded statement)\n"
                "- relevant_context (only the specific context required for this task)\n"
                "- scope (boundaries and exclusions)\n"
                "- constraints (applicable constraints)\n"
                "- expected_output (exact expected deliverable format)\n\n"
                "Respond ONLY with a valid JSON array of objects with the keys above."
            )
            response = model.invoke([HumanMessage(content=decomposition_prompt)])
            text = response.content if isinstance(response.content, str) else str(response.content)

            # Extract JSON block
            json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if json_match:
                raw_json = json_match.group(0)
                parsed = json.loads(raw_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    tasks = [DelegatedTask.from_dict(item) for item in parsed]
                    logger.info("Dynamically decomposed objective into %d tasks via LLM", len(tasks))
                    return tasks
        except Exception as exc:
            logger.warning("LLM task decomposition encountered exception: %s. Using structured fallback.", exc)

    # Fallback / heuristic decomposition if no model or LLM parsing fails
    return _heuristic_decompose(objective, context, tasks_hint)


def _heuristic_decompose(
    objective: str,
    context: Optional[str] = None,
    tasks_hint: Optional[int] = None,
) -> list[DelegatedTask]:
    """Heuristic decomposition parsing bullet points or numbered requirements."""
    # Look for numbered or bulleted items
    bullet_items = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s*(.+?)(?=(?:\n\s*(?:\d+[\.\)]|[-*•]))|\Z)", objective, re.DOTALL)
    if len(bullet_items) >= 2:
        tasks = []
        for i, item in enumerate(bullet_items, start=1):
            cleaned = item.strip()
            tasks.append(
                DelegatedTask(
                    task_id=f"task-{i}",
                    objective=cleaned,
                    relevant_context=context or "",
                    scope=f"Execute: {cleaned}",
                    constraints="Do not delegate or exceed task boundaries.",
                    expected_output="Structured requirements or analysis for this item.",
                )
            )
        return tasks

    # Default 2-task split for multi-part objectives
    count = tasks_hint or 2
    tasks = []
    for i in range(1, count + 1):
        tasks.append(
            DelegatedTask(
                task_id=f"task-{i}",
                objective=f"Part {i} of objective: {objective}",
                relevant_context=context or "",
                scope=f"Bounded scope for Part {i}",
                constraints="Do not delegate or exceed task boundaries.",
                expected_output=f"Deliverable for Part {i}.",
            )
        )
    return tasks
