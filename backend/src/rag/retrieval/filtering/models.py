"""Domain models, operators, and condition structures for retrieval metadata filtering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from exceptions.retrieval import InvalidFilterError


class ComparisonOperator(str, Enum):
    """Supported comparison operators for metadata filter conditions."""

    EQ = "eq"
    NE = "ne"
    IN = "in"
    NIN = "nin"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"


class LogicalOperator(str, Enum):
    """Supported logical grouping operators for compound filtering."""

    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True)
class BaseFilterCondition(ABC):
    """Abstract base class for all metadata filter conditions."""

    @property
    @abstractmethod
    def condition_type(self) -> str:
        """Return the condition type identifier."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize condition to a dictionary."""


@dataclass(frozen=True)
class ExactMatchCondition(BaseFilterCondition):
    """Constraint asserting that a metadata field matches a specific scalar value exactly.

    Attributes:
        field: Name of the metadata attribute.
        value: Target value to match.
    """

    field: str
    value: Any

    def __post_init__(self) -> None:
        """Validate exact match condition attributes."""
        if not isinstance(self.field, str) or not self.field.strip():
            raise InvalidFilterError("ExactMatchCondition field must be a non-empty string.")
        if self.value is None:
            raise InvalidFilterError(
                f"ExactMatchCondition for field '{self.field}' cannot have None value. "
                f"Missing metadata must never silently satisfy a constraint."
            )

    @property
    def condition_type(self) -> str:
        """Return condition type identifier."""
        return "exact"

    def to_dict(self) -> dict[str, Any]:
        """Serialize exact match condition."""
        return {
            "type": self.condition_type,
            "field": self.field,
            "operator": ComparisonOperator.EQ.value,
            "value": self.value,
        }


@dataclass(frozen=True)
class SetCondition(BaseFilterCondition):
    """Constraint asserting that a metadata field matches any value from a permitted set (IN).

    Attributes:
        field: Name of the metadata attribute.
        values: Tuple of permitted target values.
    """

    field: str
    values: tuple[Any, ...]

    def __init__(self, field: str, values: Sequence[Any] | set[Any]) -> None:
        """Initialize SetCondition, converting values to an immutable tuple."""
        if not isinstance(field, str) or not field.strip():
            raise InvalidFilterError("SetCondition field must be a non-empty string.")
        if not isinstance(values, (Sequence, set)) or isinstance(values, (str, bytes)):
            raise InvalidFilterError(
                f"SetCondition values for field '{field}' must be a sequence or set, got {type(values).__name__}."
            )
        clean_values = tuple(v for v in values if v is not None)
        if len(clean_values) == 0:
            raise InvalidFilterError(
                f"SetCondition values for field '{field}' cannot be empty or contain only None values."
            )
        object.__setattr__(self, "field", field.strip())
        object.__setattr__(self, "values", clean_values)

    @property
    def condition_type(self) -> str:
        """Return condition type identifier."""
        return "set"

    def to_dict(self) -> dict[str, Any]:
        """Serialize set condition."""
        return {
            "type": self.condition_type,
            "field": self.field,
            "operator": ComparisonOperator.IN.value,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class RangeCondition(BaseFilterCondition):
    """Constraint asserting that an ordered metadata field falls within numerical or date boundaries.

    Attributes:
        field: Name of the metadata attribute.
        gte: Optional inclusive lower bound (>=).
        lte: Optional inclusive upper bound (<=).
        gt: Optional exclusive lower bound (>).
        lt: Optional exclusive upper bound (<).
    """

    field: str
    gte: Any = None
    lte: Any = None
    gt: Any = None
    lt: Any = None

    def __post_init__(self) -> None:
        """Validate range condition attributes."""
        if not isinstance(self.field, str) or not self.field.strip():
            raise InvalidFilterError("RangeCondition field must be a non-empty string.")
        if all(b is None for b in (self.gte, self.lte, self.gt, self.lt)):
            raise InvalidFilterError(
                f"RangeCondition for field '{self.field}' must specify at least one boundary "
                f"(gte, lte, gt, or lt)."
            )

    @property
    def condition_type(self) -> str:
        """Return condition type identifier."""
        return "range"

    def to_dict(self) -> dict[str, Any]:
        """Serialize range condition."""
        data: dict[str, Any] = {
            "type": self.condition_type,
            "field": self.field,
        }
        if self.gte is not None:
            data["gte"] = self.gte
        if self.lte is not None:
            data["lte"] = self.lte
        if self.gt is not None:
            data["gt"] = self.gt
        if self.lt is not None:
            data["lt"] = self.lt
        return data


@dataclass(frozen=True)
class BooleanCondition(BaseFilterCondition):
    """Constraint asserting that a boolean metadata field strictly equals True or False.

    Attributes:
        field: Name of the metadata attribute.
        value: Boolean value (True or False).
    """

    field: str
    value: bool

    def __post_init__(self) -> None:
        """Validate boolean condition attributes."""
        if not isinstance(self.field, str) or not self.field.strip():
            raise InvalidFilterError("BooleanCondition field must be a non-empty string.")
        if not isinstance(self.value, bool):
            raise InvalidFilterError(
                f"BooleanCondition for field '{self.field}' requires a strict bool value, "
                f"got '{type(self.value).__name__}' ({self.value!r})."
            )

    @property
    def condition_type(self) -> str:
        """Return condition type identifier."""
        return "boolean"

    def to_dict(self) -> dict[str, Any]:
        """Serialize boolean condition."""
        return {
            "type": self.condition_type,
            "field": self.field,
            "operator": ComparisonOperator.EQ.value,
            "value": self.value,
        }


@dataclass(frozen=True)
class CompoundCondition(BaseFilterCondition):
    """Logical grouping combining multiple sub-conditions with AND, OR, or NOT semantics.

    Attributes:
        operator: LogicalOperator (AND, OR, NOT).
        conditions: Tuple of child BaseFilterCondition instances.
    """

    operator: LogicalOperator
    conditions: tuple[BaseFilterCondition, ...]

    def __init__(
        self,
        operator: LogicalOperator | str,
        conditions: Sequence[BaseFilterCondition],
    ) -> None:
        """Initialize CompoundCondition."""
        if isinstance(operator, str):
            try:
                op = LogicalOperator(operator.lower())
            except ValueError:
                raise InvalidFilterError(
                    f"Invalid logical operator '{operator}'. Must be one of: "
                    f"{', '.join(o.value for o in LogicalOperator)}."
                )
        elif isinstance(operator, LogicalOperator):
            op = operator
        else:
            raise InvalidFilterError(
                f"operator must be LogicalOperator or str, got {type(operator).__name__}."
            )

        if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
            raise InvalidFilterError(
                f"CompoundCondition conditions must be a sequence, got {type(conditions).__name__}."
            )

        cond_list: list[BaseFilterCondition] = []
        for i, c in enumerate(conditions):
            if not isinstance(c, BaseFilterCondition):
                raise InvalidFilterError(
                    f"Item at index {i} in CompoundCondition conditions is not a BaseFilterCondition "
                    f"instance, got {type(c).__name__}."
                )
            cond_list.append(c)

        if len(cond_list) == 0:
            raise InvalidFilterError(
                f"CompoundCondition with operator '{op.value}' must contain at least one sub-condition."
            )

        object.__setattr__(self, "operator", op)
        object.__setattr__(self, "conditions", tuple(cond_list))

    @property
    def condition_type(self) -> str:
        """Return condition type identifier."""
        return "compound"

    def to_dict(self) -> dict[str, Any]:
        """Serialize compound condition."""
        return {
            "type": self.condition_type,
            "operator": self.operator.value,
            "conditions": [c.to_dict() for c in self.conditions],
        }


@dataclass(frozen=True)
class MetadataFilter:
    """Structured container representing composite metadata filtering constraints for retrieval.

    Combines mandatory constraints (`must` / AND), optional alternatives (`should` / OR),
    and exclusion constraints (`must_not` / NOT).
    """

    must: tuple[BaseFilterCondition, ...] = ()
    should: tuple[BaseFilterCondition, ...] = ()
    must_not: tuple[BaseFilterCondition, ...] = ()

    def __init__(
        self,
        must: Sequence[BaseFilterCondition] = (),
        should: Sequence[BaseFilterCondition] = (),
        must_not: Sequence[BaseFilterCondition] = (),
    ) -> None:
        """Initialize MetadataFilter validating condition sequences."""
        object.__setattr__(self, "must", tuple(must) if must else ())
        object.__setattr__(self, "should", tuple(should) if should else ())
        object.__setattr__(self, "must_not", tuple(must_not) if must_not else ())

    @property
    def is_empty(self) -> bool:
        """Return True if no constraints are specified in this filter."""
        return len(self.must) == 0 and len(self.should) == 0 and len(self.must_not) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize filter container to a dictionary."""
        data: dict[str, Any] = {}
        if self.must:
            data["must"] = [c.to_dict() for c in self.must]
        if self.should:
            data["should"] = [c.to_dict() for c in self.should]
        if self.must_not:
            data["must_not"] = [c.to_dict() for c in self.must_not]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MetadataFilter":
        """Construct a MetadataFilter from a flexible configuration dictionary.

        Supports:
            1. Explicit structured syntax:
               {"must": [...], "should": [...], "must_not": [...]}
               {"and": [...], "or": [...], "not": [...]}
            2. Flat key-value constraints (implicitly ANDed):
               {"document_id": "D123", "has_code": True}
               {"document_type": ["req", "spec"]}  -> SetCondition
               {"character_count": {"gte": 100, "lte": 500}} -> RangeCondition
        """
        if data is None or len(data) == 0:
            return cls()

        if not isinstance(data, dict):
            raise InvalidFilterError(
                f"MetadataFilter.from_dict requires a dictionary, got {type(data).__name__}."
            )

        # Case 1: Structured boolean clauses
        has_must = "must" in data or "and" in data
        has_should = "should" in data or "or" in data
        has_must_not = "must_not" in data or "not" in data

        if has_must or has_should or has_must_not:
            must_raw = data.get("must", data.get("and", []))
            should_raw = data.get("should", data.get("or", []))
            must_not_raw = data.get("must_not", data.get("not", []))

            must_conds = cls._parse_conditions_list(must_raw, context="must")
            should_conds = cls._parse_conditions_list(should_raw, context="should")
            must_not_conds = cls._parse_conditions_list(must_not_raw, context="must_not")

            return cls(must=must_conds, should=should_conds, must_not=must_not_conds)

        # Case 2: Flat key-value constraint map
        must_list: list[BaseFilterCondition] = []
        for key, val in data.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidFilterError(f"Filter field key must be a non-empty string, got {key!r}.")
            cond = cls._parse_single_field_value(key.strip(), val)
            must_list.append(cond)

        return cls(must=must_list)

    @classmethod
    def _parse_conditions_list(
        cls,
        items: Any,
        context: str,
    ) -> list[BaseFilterCondition]:
        """Parse a list of condition specifications into BaseFilterCondition instances."""
        if items is None:
            return []
        if isinstance(items, BaseFilterCondition):
            return [items]
        if isinstance(items, dict):
            # Single dict mapping inside clause
            if any(k in items for k in ("field", "operator", "value", "type", "and", "or", "not", "must", "should", "must_not")):
                # Structured condition dict
                return [cls._parse_condition_dict(items)]
            # Flat dict of fields
            return [cls._parse_single_field_value(k, v) for k, v in items.items()]

        if not isinstance(items, (list, tuple)):
            raise InvalidFilterError(
                f"Filter clause '{context}' expects a list or tuple of conditions, got {type(items).__name__}."
            )

        parsed: list[BaseFilterCondition] = []
        for i, item in enumerate(items):
            if isinstance(item, BaseFilterCondition):
                parsed.append(item)
            elif isinstance(item, dict):
                parsed.append(cls._parse_condition_dict(item))
            else:
                raise InvalidFilterError(
                    f"Item {i} in '{context}' is not a valid condition or dict, got {type(item).__name__}."
                )
        return parsed

    @classmethod
    def _parse_condition_dict(cls, d: dict[str, Any]) -> BaseFilterCondition:
        """Parse a dictionary describing a condition or compound condition."""
        if "and" in d:
            sub = cls._parse_conditions_list(d["and"], context="and")
            return CompoundCondition(LogicalOperator.AND, sub)
        if "or" in d:
            sub = cls._parse_conditions_list(d["or"], context="or")
            return CompoundCondition(LogicalOperator.OR, sub)
        if "not" in d:
            sub = cls._parse_conditions_list(d["not"], context="not")
            return CompoundCondition(LogicalOperator.NOT, sub)

        field_name = d.get("field")
        if not field_name or not isinstance(field_name, str):
            # Could be a single-key dictionary like {"doc_type": "req"}
            if len(d) == 1:
                k, v = next(iter(d.items()))
                return cls._parse_single_field_value(k, v)
            raise InvalidFilterError(f"Condition dict missing 'field' key: {d}")

        op = d.get("operator", d.get("type", "exact"))
        if op in (ComparisonOperator.EQ.value, "exact"):
            if "value" not in d:
                raise InvalidFilterError(f"Exact match condition for '{field_name}' missing 'value'.")
            return ExactMatchCondition(field_name, d["value"])
        elif op in (ComparisonOperator.IN.value, "set"):
            vals = d.get("values", d.get("value"))
            return SetCondition(field_name, vals)
        elif op in ("range", ComparisonOperator.GTE.value, ComparisonOperator.LTE.value):
            return RangeCondition(
                field=field_name,
                gte=d.get("gte"),
                lte=d.get("lte"),
                gt=d.get("gt"),
                lt=d.get("lt"),
            )
        elif op in ("boolean", "bool"):
            if "value" not in d:
                raise InvalidFilterError(f"Boolean condition for '{field_name}' missing 'value'.")
            return BooleanCondition(field_name, d["value"])
        elif op in (ComparisonOperator.NE.value, "ne"):
            exact = ExactMatchCondition(field_name, d["value"])
            return CompoundCondition(LogicalOperator.NOT, [exact])
        else:
            raise InvalidFilterError(f"Unsupported filter condition operator or type: '{op}'.")

    @classmethod
    def _parse_single_field_value(cls, key: str, val: Any) -> BaseFilterCondition:
        """Heuristically translate a single field-value pair into the appropriate condition."""
        if isinstance(val, BaseFilterCondition):
            return val

        if isinstance(val, bool):
            return BooleanCondition(field=key, value=val)

        if isinstance(val, (list, tuple, set)):
            return SetCondition(field=key, values=val)

        if isinstance(val, dict):
            # Check for range operators
            range_keys = {"gte", "lte", "gt", "lt"}
            if any(rk in val for rk in range_keys):
                return RangeCondition(
                    field=key,
                    gte=val.get("gte"),
                    lte=val.get("lte"),
                    gt=val.get("gt"),
                    lt=val.get("lt"),
                )
            if "in" in val:
                return SetCondition(field=key, values=val["in"])
            if "nin" in val or "not_in" in val:
                set_cond = SetCondition(field=key, values=val.get("nin", val.get("not_in")))
                return CompoundCondition(LogicalOperator.NOT, [set_cond])
            if "eq" in val:
                return ExactMatchCondition(field=key, value=val["eq"])
            if "ne" in val:
                exact = ExactMatchCondition(field=key, value=val["ne"])
                return CompoundCondition(LogicalOperator.NOT, [exact])

        return ExactMatchCondition(field=key, value=val)
