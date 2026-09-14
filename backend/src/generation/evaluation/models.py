"""Domain models for Groundedness and Safety evaluation stage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Structured result of Groundedness and Safety evaluation quality gate.

    Faithfully represents the evaluation decision:
    - grounded: True if all claims in the generated response are supported by the context.
    - safe: True if the response complies with all safety requirements.
    - reason: Concise, actionable explanation justifying the decision.
    - passed: True only if BOTH grounded AND safe are True.
    """

    grounded: bool
    safe: bool
    reason: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Quality gate decision: True ONLY if both grounded AND safe."""
        return bool(self.grounded and self.safe)

    @property
    def is_acceptable(self) -> bool:
        """Alias for passed representing whether response can be returned to user."""
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation result to standard dictionary representation."""
        return {
            "grounded": self.grounded,
            "safe": self.safe,
            "passed": self.passed,
            "reason": self.reason,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Safe debug representation avoiding dumping sensitive text into logs."""
        reason_preview = (
            (self.reason[:50] + "...") if len(self.reason) > 50 else self.reason
        )
        return (
            f"EvaluationResult("
            f"grounded={self.grounded}, "
            f"safe={self.safe}, "
            f"passed={self.passed}, "
            f"reason={reason_preview!r})"
        )


@dataclass(frozen=True)
class EvaluationRequest:
    """Encapsulates validated inputs for a Groundedness and Safety evaluation run."""

    query: str
    context: str
    response: str
    project_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize request to dictionary representation."""
        return {
            "query": self.query,
            "context": self.context,
            "response": self.response,
            "project_id": self.project_id,
            "metadata": dict(self.metadata),
        }
