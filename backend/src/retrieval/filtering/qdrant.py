"""Native Qdrant filter translation for metadata filtering constraints."""

from typing import Any, Optional

from qdrant_client import models

from exceptions.retrieval import InvalidFilterError
from retrieval.filtering.models import (
    BaseFilterCondition,
    BooleanCondition,
    CompoundCondition,
    ExactMatchCondition,
    LogicalOperator,
    MetadataFilter,
    RangeCondition,
    SetCondition,
)


def to_qdrant_condition(condition: BaseFilterCondition) -> models.Condition:
    """Translate a domain BaseFilterCondition into a native Qdrant models.Condition."""
    if isinstance(condition, ExactMatchCondition):
        val = condition.value.value if hasattr(condition.value, "value") else condition.value
        return models.FieldCondition(
            key=condition.field,
            match=models.MatchValue(value=val),
        )
    elif isinstance(condition, SetCondition):
        vals = [v.value if hasattr(v, "value") else v for v in condition.values]
        return models.FieldCondition(
            key=condition.field,
            match=models.MatchAny(any=vals),
        )
    elif isinstance(condition, RangeCondition):
        return models.FieldCondition(
            key=condition.field,
            range=models.Range(
                gte=condition.gte,
                lte=condition.lte,
                gt=condition.gt,
                lt=condition.lt,
            ),
        )
    elif isinstance(condition, BooleanCondition):
        return models.FieldCondition(
            key=condition.field,
            match=models.MatchValue(value=condition.value),
        )
    elif isinstance(condition, CompoundCondition):
        sub_conditions = [to_qdrant_condition(c) for c in condition.conditions]
        if condition.operator == LogicalOperator.AND:
            return models.Filter(must=sub_conditions)
        elif condition.operator == LogicalOperator.OR:
            return models.Filter(should=sub_conditions)
        elif condition.operator == LogicalOperator.NOT:
            return models.Filter(must_not=sub_conditions)
        raise InvalidFilterError(f"Unsupported compound operator: {condition.operator}")

    raise InvalidFilterError(f"Unsupported condition type for Qdrant translation: {type(condition).__name__}")


def to_qdrant_filter(
    filter_spec: MetadataFilter | dict[str, Any] | None,
    project_id: str,
) -> models.Filter:
    """Compile a MetadataFilter or dict into a native Qdrant models.Filter with mandatory project isolation.

    Args:
        filter_spec: MetadataFilter instance, filter configuration dict, or None.
        project_id: Non-empty project ID strictly enforcing tenant isolation.

    Returns:
        models.Filter: Qdrant-compatible query filter.

    Raises:
        InvalidFilterError: If project_id is invalid or filter cannot be translated.
    """
    if not isinstance(project_id, str) or not project_id.strip():
        raise InvalidFilterError("project_id must be a non-empty string for Qdrant filter translation.")
    clean_project_id = project_id.strip()

    # Mandatory project isolation condition
    project_condition = models.FieldCondition(
        key="project_id",
        match=models.MatchValue(value=clean_project_id),
    )

    if filter_spec is None:
        return models.Filter(must=[project_condition])

    if isinstance(filter_spec, dict):
        parsed_filter = MetadataFilter.from_dict(filter_spec)
    elif isinstance(filter_spec, MetadataFilter):
        parsed_filter = filter_spec
    else:
        raise InvalidFilterError(
            f"Expected MetadataFilter or dict, got '{type(filter_spec).__name__}'."
        )

    must_clauses: list[models.Condition] = [project_condition]
    for cond in parsed_filter.must:
        must_clauses.append(to_qdrant_condition(cond))

    should_clauses: list[models.Condition] = []
    for cond in parsed_filter.should:
        should_clauses.append(to_qdrant_condition(cond))

    must_not_clauses: list[models.Condition] = []
    for cond in parsed_filter.must_not:
        must_not_clauses.append(to_qdrant_condition(cond))

    kwargs: dict[str, Any] = {"must": must_clauses}
    if should_clauses:
        kwargs["should"] = should_clauses
    if must_not_clauses:
        kwargs["must_not"] = must_not_clauses

    return models.Filter(**kwargs)
