"""Constraint IR models for functional dependencies, implications, and composite PKs."""

from typing import Literal, Any, List
from pydantic import BaseModel, Field


class AtomicCondition(BaseModel):
    """Atomic condition in a structured expression."""

    col: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "is_null", "not_null"]
    value: Any | None = None  # list for "in", single value for others


class ConditionExpr(BaseModel):
    """Structured condition expression (tree of atomic conditions)."""

    kind: Literal["atom", "and", "or", "not"]
    atom: AtomicCondition | None = None
    children: List["ConditionExpr"] = Field(default_factory=list)


class FDConstraint(BaseModel):
    """Functional dependency constraint."""

    table: str
    lhs: List[str]  # determinant columns
    rhs: List[str]  # dependent columns
    mode: Literal["intra_row", "lookup"] = "intra_row"


class ImplicationConstraint(BaseModel):
    """Implication constraint: if condition, then effect."""

    table: str
    condition: ConditionExpr
    effect: ConditionExpr


class CompositePKConstraint(BaseModel):
    """Composite primary key constraint."""

    table: str
    cols: List[str]


class ConstraintSpec(BaseModel):
    """Collection of all constraints for a schema."""

    fds: List[FDConstraint] = Field(default_factory=list)
    implications: List[ImplicationConstraint] = Field(default_factory=list)
    composite_pks: List[CompositePKConstraint] = Field(default_factory=list)

