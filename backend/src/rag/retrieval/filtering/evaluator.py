"""Deterministic in-memory evaluation engine for retrieval metadata filtering."""

from datetime import date, datetime, time
from typing import Any, Callable, Mapping, Optional

from observability.logging import get_logger
from exceptions.retrieval import FilterEvaluationError
from rag.retrieval.filtering.models import (
    BaseFilterCondition,
    BooleanCondition,
    CompoundCondition,
    ExactMatchCondition,
    LogicalOperator,
    MetadataFilter,
    RangeCondition,
    SetCondition,
)
from rag.retrieval.models import FusedCandidate

logger = get_logger(__name__)


def _parse_comparable(val: Any) -> Any:
    """Normalize numeric and date/datetime values for chronological and numerical comparisons."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, (datetime, date)):
        return val
    if isinstance(val, str):
        clean = val.strip()
        # Attempt ISO 8601 parsing for dates and datetimes
        try:
            return datetime.fromisoformat(clean)
        except (ValueError, TypeError):
            try:
                return date.fromisoformat(clean)
            except (ValueError, TypeError):
                pass
    return val


def _compare_bounds(candidate_val: Any, bound_val: Any, op: str) -> bool:
    """Compare candidate value with a range boundary with type harmonization."""
    c_parsed = _parse_comparable(candidate_val)
    b_parsed = _parse_comparable(bound_val)

    # Harmonize datetime with date
    if isinstance(c_parsed, datetime) and isinstance(b_parsed, date) and not isinstance(b_parsed, datetime):
        # Convert date boundary to datetime at midnight
        b_parsed = datetime.combine(b_parsed, time.min, tzinfo=c_parsed.tzinfo)
    elif isinstance(b_parsed, datetime) and isinstance(c_parsed, date) and not isinstance(c_parsed, datetime):
        c_parsed = datetime.combine(c_parsed, time.min, tzinfo=b_parsed.tzinfo)

    # If both have tzinfo or both are naive, or handle naive/aware mismatch safely
    if isinstance(c_parsed, datetime) and isinstance(b_parsed, datetime):
        if (c_parsed.tzinfo is None) != (b_parsed.tzinfo is None):
            # Convert aware to naive UTC representation for fair comparison
            if c_parsed.tzinfo is not None:
                c_parsed = c_parsed.replace(tzinfo=None)
            if b_parsed.tzinfo is not None:
                b_parsed = b_parsed.replace(tzinfo=None)

    try:
        if op == "gte":
            return c_parsed >= b_parsed
        elif op == "lte":
            return c_parsed <= b_parsed
        elif op == "gt":
            return c_parsed > b_parsed
        elif op == "lt":
            return c_parsed < b_parsed
        return False
    except TypeError:
        # Incomparable types
        return False


class CandidateMetadataResolver:
    """Extracts metadata values from a candidate, its metadata payload, and optional external lookup."""

    @staticmethod
    def resolve_field(
        candidate: FusedCandidate,
        field_name: str,
        metadata_lookup: Optional[Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any]]] = None,
    ) -> tuple[bool, Any]:
        """Resolve field value from candidate hierarchy.

        Resolution order:
            1. External metadata_lookup (if provided)
            2. Candidate's internal `metadata` dictionary
            3. Candidate's top-level dataclass attributes (document_id, project_id, etc.)

        Returns:
            tuple[bool, Any]: (field_exists, value). If missing, returns (False, None).
        """
        # 1. External lookup
        if metadata_lookup is not None:
            external_meta: Optional[Mapping[str, Any]] = None
            if callable(metadata_lookup):
                try:
                    external_meta = metadata_lookup(candidate.chunk_id)
                except Exception:
                    external_meta = None
            elif isinstance(metadata_lookup, Mapping):
                external_meta = metadata_lookup.get(candidate.chunk_id)

            if external_meta is not None and field_name in external_meta:
                val = external_meta[field_name]
                return (True, val)

        # 2. Candidate metadata dict
        cand_meta = getattr(candidate, "metadata", None)
        if isinstance(cand_meta, Mapping) and field_name in cand_meta:
            return (True, cand_meta[field_name])

        # 3. Candidate top-level attributes
        if hasattr(candidate, field_name):
            val = getattr(candidate, field_name)
            if val is not None:
                return (True, val)

        return (False, None)


class MetadataConditionEvaluator:
    """Evaluates BaseFilterCondition and MetadataFilter instances against candidate records."""

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize evaluator.

        Args:
            strict_mode: When True, unexpected evaluation errors raise FilterEvaluationError.
        """
        self._strict_mode = strict_mode

    def evaluate_condition(
        self,
        condition: BaseFilterCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any]]] = None,
    ) -> bool:
        """Evaluate a single filter condition against a candidate record."""
        try:
            if isinstance(condition, ExactMatchCondition):
                return self._evaluate_exact(condition, candidate, metadata_lookup)
            elif isinstance(condition, SetCondition):
                return self._evaluate_set(condition, candidate, metadata_lookup)
            elif isinstance(condition, RangeCondition):
                return self._evaluate_range(condition, candidate, metadata_lookup)
            elif isinstance(condition, BooleanCondition):
                return self._evaluate_boolean(condition, candidate, metadata_lookup)
            elif isinstance(condition, CompoundCondition):
                return self._evaluate_compound(condition, candidate, metadata_lookup)
            else:
                logger.warning("Unrecognized filter condition type: %s", type(condition).__name__)
                if self._strict_mode:
                    raise FilterEvaluationError(f"Unrecognized condition type: {type(condition).__name__}")
                return False
        except Exception as exc:
            if isinstance(exc, FilterEvaluationError):
                raise
            logger.error("Error evaluating condition %s on chunk '%s': %s", condition, candidate.chunk_id, exc)
            if self._strict_mode:
                raise FilterEvaluationError(
                    f"Error evaluating condition {condition} on chunk '{candidate.chunk_id}': {exc}",
                    original_error=exc,
                ) from exc
            return False

    def _evaluate_exact(
        self,
        cond: ExactMatchCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Any],
    ) -> bool:
        """Evaluate exact match constraint."""
        exists, val = CandidateMetadataResolver.resolve_field(candidate, cond.field, metadata_lookup)
        if not exists or val is None:
            return False

        target = cond.value
        # If target is enum or val is enum, resolve representation
        val_cmp = val.value if hasattr(val, "value") else val
        target_cmp = target.value if hasattr(target, "value") else target

        # Strict boolean distinction
        if isinstance(val_cmp, bool) != isinstance(target_cmp, bool):
            return False

        return val_cmp == target_cmp

    def _evaluate_set(
        self,
        cond: SetCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Any],
    ) -> bool:
        """Evaluate multiple-value / set membership constraint (IN)."""
        exists, val = CandidateMetadataResolver.resolve_field(candidate, cond.field, metadata_lookup)
        if not exists or val is None:
            return False

        # Build normalized target lookup set
        target_set: set[Any] = set()
        for v in cond.values:
            target_set.add(v.value if hasattr(v, "value") else v)

        # If candidate field is a sequence/collection (e.g. section_path, tags)
        if isinstance(val, (list, tuple, set)) and not isinstance(val, (str, bytes)):
            # Matches if any element of the candidate collection is in the permitted set
            for item in val:
                item_cmp = item.value if hasattr(item, "value") else item
                if item_cmp in target_set:
                    return True
            return False

        # Scalar candidate value
        val_cmp = val.value if hasattr(val, "value") else val
        return val_cmp in target_set

    def _evaluate_range(
        self,
        cond: RangeCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Any],
    ) -> bool:
        """Evaluate numerical or chronological range constraint."""
        exists, val = CandidateMetadataResolver.resolve_field(candidate, cond.field, metadata_lookup)
        if not exists or val is None:
            return False

        if cond.gte is not None and not _compare_bounds(val, cond.gte, "gte"):
            return False
        if cond.lte is not None and not _compare_bounds(val, cond.lte, "lte"):
            return False
        if cond.gt is not None and not _compare_bounds(val, cond.gt, "gt"):
            return False
        if cond.lt is not None and not _compare_bounds(val, cond.lt, "lt"):
            return False

        return True

    def _evaluate_boolean(
        self,
        cond: BooleanCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Any],
    ) -> bool:
        """Evaluate strict boolean constraint."""
        exists, val = CandidateMetadataResolver.resolve_field(candidate, cond.field, metadata_lookup)
        if not exists or val is None:
            return False

        # Must be genuine boolean
        if not isinstance(val, bool):
            return False

        return val is cond.value

    def _evaluate_compound(
        self,
        cond: CompoundCondition,
        candidate: FusedCandidate,
        metadata_lookup: Optional[Any],
    ) -> bool:
        """Evaluate logical compound group with short-circuiting."""
        if cond.operator == LogicalOperator.AND:
            for sub in cond.conditions:
                if not self.evaluate_condition(sub, candidate, metadata_lookup):
                    return False
            return True
        elif cond.operator == LogicalOperator.OR:
            for sub in cond.conditions:
                if self.evaluate_condition(sub, candidate, metadata_lookup):
                    return True
            return False
        elif cond.operator == LogicalOperator.NOT:
            # Returns True only if none of the subconditions match
            for sub in cond.conditions:
                if self.evaluate_condition(sub, candidate, metadata_lookup):
                    return False
            return True
        return False

    def evaluate_filter(
        self,
        metadata_filter: MetadataFilter,
        candidate: FusedCandidate,
        project_id: Optional[str] = None,
        metadata_lookup: Optional[Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any]]] = None,
    ) -> bool:
        """Evaluate composite MetadataFilter against candidate, enforcing project isolation.

        Returns True if candidate satisfies all constraints; False otherwise.
        """
        # 1. Enforce Project Isolation Invariant
        if project_id is not None:
            clean_pid = project_id.strip()
            if candidate.project_id.strip() != clean_pid:
                logger.warning(
                    "Candidate '%s' failed project isolation in metadata filter (expected='%s', actual='%s')",
                    candidate.chunk_id,
                    clean_pid,
                    candidate.project_id,
                )
                return False

        # 2. Passthrough if empty filter
        if metadata_filter.is_empty:
            return True

        # 3. Must clauses (AND)
        for cond in metadata_filter.must:
            if not self.evaluate_condition(cond, candidate, metadata_lookup):
                return False

        # 4. Must Not clauses (NOT)
        for cond in metadata_filter.must_not:
            if self.evaluate_condition(cond, candidate, metadata_lookup):
                return False

        # 5. Should clauses (OR)
        if metadata_filter.should:
            matched_any = False
            for cond in metadata_filter.should:
                if self.evaluate_condition(cond, candidate, metadata_lookup):
                    matched_any = True
                    break
            if not matched_any:
                return False

        return True
