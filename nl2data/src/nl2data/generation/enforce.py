"""Constraint enforcement during data generation."""

import pandas as pd
from typing import List
from nl2data.ir.constraint_ir import (
    FDConstraint,
    ImplicationConstraint,
    ConstraintSpec,
    ConditionExpr,
    AtomicCondition,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def eval_condition(df: pd.DataFrame, expr: ConditionExpr) -> pd.Series:
    """
    Evaluate structured condition expression on DataFrame.

    Args:
        df: DataFrame to evaluate condition on
        expr: ConditionExpr to evaluate

    Returns:
        Boolean Series indicating which rows match the condition
    """
    if expr.kind == "atom":
        c = expr.atom
        if c is None:
            raise ValueError("Atom condition must have atom field set")
        s = df[c.col]
        if c.op == "eq":
            return s == c.value
        if c.op == "ne":
            return s != c.value
        if c.op == "lt":
            return s < c.value
        if c.op == "le":
            return s <= c.value
        if c.op == "gt":
            return s > c.value
        if c.op == "ge":
            return s >= c.value
        if c.op == "in":
            if not isinstance(c.value, list):
                raise ValueError("'in' operator requires list value")
            return s.isin(c.value)
        if c.op == "is_null":
            return s.isna()
        if c.op == "not_null":
            return s.notna()
        raise ValueError(f"Unknown op: {c.op}")

    if expr.kind == "and":
        mask = pd.Series(True, index=df.index)
        for child in expr.children:
            mask &= eval_condition(df, child)
        return mask

    if expr.kind == "or":
        mask = pd.Series(False, index=df.index)
        for child in expr.children:
            mask |= eval_condition(df, child)
        return mask

    if expr.kind == "not":
        if not expr.children:
            raise ValueError("'not' condition must have at least one child")
        return ~eval_condition(df, expr.children[0])

    raise ValueError(f"Unknown kind: {expr.kind}")


def enforce_intra_fd(df: pd.DataFrame, fd: FDConstraint) -> pd.DataFrame:
    """
    Enforce intra-row functional dependency.

    Groups by LHS columns and ensures RHS values are unique per group.
    If violations exist, picks first value (or majority) and overwrites others.

    Args:
        df: DataFrame to enforce FD on
        fd: FDConstraint to enforce

    Returns:
        DataFrame with FD enforced
    """
    key = fd.lhs
    val = fd.rhs[0]  # Keep it simple: single RHS column

    # Check all columns exist
    for col in key + fd.rhs:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping FD enforcement")
            return df

    # Create mapping from lhs -> canonical rhs
    # Use drop_duplicates to get first occurrence of each key
    canon_df = df[key + fd.rhs].drop_duplicates(subset=key, keep="first")
    canon = canon_df.set_index(key)[val]

    # Map and overwrite
    if len(key) == 1:
        df[val] = df[key[0]].map(canon)
    else:
        # Multi-column key: create tuple index
        key_tuples = df[key].apply(tuple, axis=1)
        canon_tuples = canon_df[key].apply(tuple, axis=1)
        canon_dict = dict(zip(canon_tuples, canon_df[val]))
        df[val] = key_tuples.map(canon_dict)

    return df


def enforce_implication(df: pd.DataFrame, constraint: ImplicationConstraint) -> pd.DataFrame:
    """
    Enforce implication: if condition, then effect.

    When condition matches, enforce effect (set to NULL, set to value, etc.).

    Args:
        df: DataFrame to enforce implication on
        constraint: ImplicationConstraint to enforce

    Returns:
        DataFrame with implication enforced
    """
    mask_if = eval_condition(df, constraint.condition)

    # Apply effect when condition is true
    if constraint.effect.kind == "atom" and constraint.effect.atom:
        atom = constraint.effect.atom
        col = atom.col

        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping implication enforcement")
            return df

        if atom.op == "is_null":
            df.loc[mask_if, col] = None
        elif atom.op == "not_null":
            # Not null: ensure value is not null (could set to default)
            df.loc[mask_if & df[col].isna(), col] = atom.value if atom.value is not None else ""
        elif atom.op == "eq":
            df.loc[mask_if, col] = atom.value
        else:
            logger.warning(f"Unsupported effect op: {atom.op}, skipping")

    return df


def enforce_nullability(df: pd.DataFrame, table_spec) -> pd.DataFrame:
    """
    Enforce nullability constraints from table specification.

    Args:
        df: DataFrame to enforce nullability on
        table_spec: TableSpec with column definitions

    Returns:
        DataFrame with nullability enforced
    """
    for col_spec in table_spec.columns:
        if not col_spec.nullable and col_spec.name in df.columns:
            # Fill nulls with appropriate defaults based on type
            null_mask = df[col_spec.name].isna()
            if null_mask.any():
                if col_spec.sql_type == "INT":
                    df.loc[null_mask, col_spec.name] = 0
                elif col_spec.sql_type == "FLOAT":
                    df.loc[null_mask, col_spec.name] = 0.0
                elif col_spec.sql_type == "TEXT":
                    df.loc[null_mask, col_spec.name] = ""
                elif col_spec.sql_type == "BOOL":
                    df.loc[null_mask, col_spec.name] = False
                elif col_spec.sql_type in ("DATE", "DATETIME"):
                    # Use a default date (e.g., epoch)
                    df.loc[null_mask, col_spec.name] = pd.Timestamp("1970-01-01")
                logger.warning(
                    f"Filled {null_mask.sum()} null values in non-nullable column "
                    f"{col_spec.name}"
                )

    return df


def enforce_batch(
    df: pd.DataFrame,
    constraints: ConstraintSpec,
    table_spec=None,
) -> pd.DataFrame:
    """
    Apply all constraints to a batch of rows.

    Args:
        df: DataFrame to enforce constraints on
        constraints: ConstraintSpec with all constraints (FDs are now in table_spec.fds)
        table_spec: Optional TableSpec for nullability enforcement and FDs

    Returns:
        DataFrame with all constraints enforced
    """
    result = df.copy()

    # Enforce FDs from table_spec.fds (not from constraints.fds)
    if table_spec and hasattr(table_spec, 'fds'):
        from nl2data.ir.constraint_ir import FDConstraint
        for table_fd in table_spec.fds:
            # Convert TableFDConstraint to FDConstraint for enforce_intra_fd
            fd = FDConstraint(
                table=table_spec.name,
                lhs=table_fd.lhs,
                rhs=table_fd.rhs,
                mode=table_fd.mode
            )
            if fd.mode == "intra_row":
                result = enforce_intra_fd(result, fd)
            # mode="lookup" FDs are checked post-generation, not enforced here

    # Enforce implications
    for impl in constraints.implications:
        result = enforce_implication(result, impl)

    # Enforce nullability if table_spec provided
    if table_spec:
        result = enforce_nullability(result, table_spec)

    return result

