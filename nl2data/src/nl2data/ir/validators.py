"""Validators for IR models."""

from dataclasses import dataclass, field
from typing import Literal, Callable, List, TYPE_CHECKING
from pydantic import ValidationError
from .dataset import DatasetIR
from nl2data.config.logging import get_logger

if TYPE_CHECKING:
    from nl2data.agents.base import Blackboard

logger = get_logger(__name__)


@dataclass
class QaIssue:
    """QA issue found during validation."""

    stage: Literal["LogicalIR", "GenerationIR", "PostGen"]
    code: str  # e.g., "MISSING_PK", "FK_REF_INVALID"
    location: str  # e.g., "table_name" or "table_name.column_name"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class DatasetQaIssue(QaIssue):
    """QA issue specific to dataset-level validation."""

    pass


def validate_logical(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate logical schema constraints.

    Args:
        ir: DatasetIR to validate

    Returns:
        List of QaIssue objects (empty if validation passes)
    """
    tables = ir.logical.tables
    issues: List[QaIssue] = []

    for table_name, table in tables.items():
        # Check primary key exists
        if not table.primary_key:
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="MISSING_PK",
                    location=table_name,
                    message=f"{table_name}: missing primary key. ALL tables must have at least one primary key column.",
                    details={"table": table_name, "kind": table.kind},
                )
            )

        # Check primary key columns exist
        for pk_col in table.primary_key:
            if pk_col not in {c.name for c in table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="PK_COL_MISSING",
                        location=f"{table_name}.{pk_col}",
                        message=f"{table_name}: primary key column '{pk_col}' does not exist",
                        details={"table": table_name, "column": pk_col},
                    )
                )

        # Check foreign keys
        for fk in table.foreign_keys:
            # Check referenced table exists
            if fk.ref_table not in tables:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="FK_REF_TABLE_MISSING",
                        location=f"{table_name}.{fk.column}",
                        message=f"{table_name}: foreign key '{fk.column}' references "
                        f"missing table '{fk.ref_table}'",
                        details={
                            "table": table_name,
                            "fk_column": fk.column,
                            "ref_table": fk.ref_table,
                        },
                    )
                )
                continue

            # Check FK column exists
            if fk.column not in {c.name for c in table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="FK_COL_MISSING",
                        location=f"{table_name}.{fk.column}",
                        message=f"{table_name}: foreign key column '{fk.column}' does not exist",
                        details={"table": table_name, "column": fk.column},
                    )
                )
                continue

            # Check referenced column exists
            ref_table = tables[fk.ref_table]
            if fk.ref_column not in {c.name for c in ref_table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="FK_REF_COL_MISSING",
                        location=f"{table_name}.{fk.column}",
                        message=f"{table_name}: foreign key '{fk.column}' references "
                        f"'{fk.ref_table}.{fk.ref_column}' which does not exist",
                        details={
                            "table": table_name,
                            "fk_column": fk.column,
                            "ref_table": fk.ref_table,
                            "ref_column": fk.ref_column,
                        },
                    )
                )

    if issues:
        logger.warning(f"Logical validation found {len(issues)} issues")
    else:
        logger.info("Logical validation passed")

    return issues


def validate_generation(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate generation specifications.

    Args:
        ir: DatasetIR to validate

    Returns:
        List of QaIssue objects (empty if validation passes)
    """
    tables = ir.logical.tables
    issues: List[QaIssue] = []

    for cg in ir.generation.columns:
        # Check table exists
        if cg.table not in tables:
            issues.append(
                QaIssue(
                    stage="GenerationIR",
                    code="GEN_TABLE_MISSING",
                    location=cg.table,
                    message=f"Generation spec references unknown table '{cg.table}'",
                    details={"table": cg.table, "column": cg.column},
                )
            )
            continue

        # Check column exists
        table = tables[cg.table]
        if cg.column not in {c.name for c in table.columns}:
            issues.append(
                QaIssue(
                    stage="GenerationIR",
                    code="GEN_COL_MISSING",
                    location=f"{cg.table}.{cg.column}",
                    message=f"Generation spec references unknown column '{cg.table}.{cg.column}'",
                    details={"table": cg.table, "column": cg.column},
                )
            )

    if issues:
        logger.warning(f"Generation validation found {len(issues)} issues")
    else:
        logger.info("Generation validation passed")

    return issues


def validate_derived_columns(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate that derived columns are properly specified.
    
    Checks:
    1. Every column in logical schema has a generation spec
    2. Derived columns have valid expressions
    3. Derived column dependencies exist
    4. Dimension columns referenced in derived expressions exist
    
    Args:
        ir: DatasetIR to validate
    
    Returns:
        List of QaIssue objects
    """
    issues: List[QaIssue] = []
    tables = ir.logical.tables
    
    # Build set of columns with generation specs
    gen_specs = {(cg.table, cg.column) for cg in ir.generation.columns}
    
    # Check 1: Every logical column has a generation spec
    for table_name, table in tables.items():
        for col in table.columns:
            if (table_name, col.name) not in gen_specs:
                issues.append(
                    QaIssue(
                        stage="GenerationIR",
                        code="MISSING_GEN_SPEC",
                        location=f"{table_name}.{col.name}",
                        message=f"Column '{col.name}' in table '{table_name}' has no generation specification. "
                                f"It must be either sampled (with a distribution) or derived (with an expression).",
                        details={"table": table_name, "column": col.name},
                    )
                )
    
    # Check 2: Derived columns have valid expressions
    from nl2data.ir.generation import DistDerived
    from nl2data.generation.derived_program import compile_derived
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistDerived):
            dist = cg.distribution
            # Try to compile expression
            try:
                prog = compile_derived(dist.expression, dist.dtype)
            except Exception as e:
                issues.append(
                    QaIssue(
                        stage="GenerationIR",
                        code="INVALID_DERIVED_EXPR",
                        location=f"{cg.table}.{cg.column}",
                        message=f"Invalid derived expression for '{cg.table}.{cg.column}': {e}",
                        details={
                            "table": cg.table,
                            "column": cg.column,
                            "expression": dist.expression,
                            "error": str(e),
                        },
                    )
                )
                continue
            
            # Check 3: Dependencies exist
            table = tables.get(cg.table)
            if table:
                # Build map of all columns across all tables for better error messages
                all_table_cols = {}
                for tname, t in tables.items():
                    all_table_cols[tname] = {c.name for c in t.columns}
                
                for dep in prog.dependencies:
                    # Check if dependency is a column in the same table
                    if dep not in {c.name for c in table.columns}:
                        # Check dimension tables via foreign keys
                        found_in_dim = False
                        found_in_table = None
                        for fk in table.foreign_keys:
                            ref_table = tables.get(fk.ref_table)
                            if ref_table and dep in {c.name for c in ref_table.columns}:
                                found_in_dim = True
                                found_in_table = fk.ref_table
                                break
                        
                        # If not found in dimensions, check all other tables
                        if not found_in_dim:
                            for tname, cols in all_table_cols.items():
                                if dep in cols:
                                    found_in_table = tname
                                    break
                        
                        # Issue if: not in same table and not in dimension
                        # (Even if found in another table, it's still an issue if not accessible via FK)
                        if not found_in_dim:
                            # Build helpful error message with suggestions
                            available_cols = []
                            for tname, cols in all_table_cols.items():
                                available_cols.extend([f"{tname}.{c}" for c in cols])
                            
                            # Find similar column names (fuzzy match)
                            similar_cols = []
                            dep_lower = dep.lower()
                            for tname, cols in all_table_cols.items():
                                for col in cols:
                                    if dep_lower in col.lower() or col.lower() in dep_lower:
                                        similar_cols.append(f"{tname}.{col}")
                            
                            error_msg = (
                                f"Derived column '{cg.table}.{cg.column}' depends on '{dep}' "
                                f"which does not exist in table '{cg.table}' or its dimension tables."
                            )
                            
                            if similar_cols:
                                error_msg += f" Did you mean: {', '.join(similar_cols[:5])}?"
                            
                            # Add architectural limitation warnings
                            if found_in_table and found_in_table != cg.table:
                                # Column exists in another table
                                ref_table_obj = tables.get(found_in_table)
                                is_dimension = ref_table_obj and ref_table_obj.kind == "dimension"
                                if not is_dimension:
                                    # Cross-fact-table reference detected
                                    error_msg += (
                                        f" Note: Column '{dep}' exists in fact table '{found_in_table}' but cross-fact-table "
                                        f"references are not currently supported. Consider using a dimension table lookup instead."
                                    )
                                else:
                                    # It's a dimension but not joined via FK - suggest adding FK
                                    error_msg += (
                                        f" Note: Column '{dep}' exists in dimension table '{found_in_table}' but there's no "
                                        f"foreign key relationship. Ensure the fact table has a foreign key to '{found_in_table}'."
                                    )
                            elif found_in_table is None:
                                # Column doesn't exist anywhere
                                error_msg += " Column does not exist in any table."
                            
                            issues.append(
                                QaIssue(
                                    stage="GenerationIR",
                                    code="MISSING_DERIVED_DEP",
                                    location=f"{cg.table}.{cg.column}",
                                    message=error_msg,
                                    details={
                                        "table": cg.table,
                                        "column": cg.column,
                                        "dependency": dep,
                                        "expression": dist.expression,
                                        "available_columns": available_cols[:20],  # Limit to avoid huge messages
                                        "similar_columns": similar_cols[:5],
                                    },
                                )
                            )
    
    if issues:
        logger.warning(f"Derived column validation found {len(issues)} issues")
    else:
        logger.info("Derived column validation passed")
    
    return issues


def validate_dataset(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate complete dataset IR.

    Args:
        ir: DatasetIR to validate

    Returns:
        List of QaIssue objects (empty if validation passes)
    """
    issues: List[QaIssue] = []
    issues.extend(validate_logical(ir))
    issues.extend(validate_generation(ir))
    issues.extend(validate_derived_columns(ir))

    if issues:
        logger.warning(f"Dataset validation found {len(issues)} issues")
    else:
        logger.info("Dataset IR validation passed")

    return issues


def validate_logical_ir(board: "Blackboard") -> List[QaIssue]:
    """
    Validate LogicalIR from blackboard.

    Args:
        board: Blackboard containing LogicalIR

    Returns:
        List of QaIssue objects
    """
    if board.logical_ir is None:
        return []

    issues: List[QaIssue] = []
    tables = board.logical_ir.tables

    # Check schema_mode consistency
    if board.requirement_ir and board.logical_ir:
        req_mode = board.requirement_ir.schema_mode
        log_mode = board.logical_ir.schema_mode
        if req_mode != log_mode:
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="SCHEMA_MODE_MISMATCH",
                    location="schema_mode",
                    message=f"Schema mode mismatch: RequirementIR has '{req_mode}', "
                    f"LogicalIR has '{log_mode}'",
                    details={"requirement_mode": req_mode, "logical_mode": log_mode},
                )
            )

    # Check composite PK consistency
    for cpk in board.logical_ir.constraints.composite_pks:
        if cpk.table not in tables:
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="COMPOSITE_PK_TABLE_MISSING",
                    location=cpk.table,
                    message=f"Composite PK references unknown table '{cpk.table}'",
                    details={"table": cpk.table, "columns": cpk.cols},
                )
            )
            continue

        table = tables[cpk.table]
        # Check all composite PK columns exist
        for col in cpk.cols:
            if col not in {c.name for c in table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="COMPOSITE_PK_COL_MISSING",
                        location=f"{cpk.table}.{col}",
                        message=f"Composite PK column '{col}' does not exist in table '{cpk.table}'",
                        details={"table": cpk.table, "column": col, "composite_pk": cpk.cols},
                    )
                )

        # Check composite PK matches table's primary_key
        if set(cpk.cols) != set(table.primary_key):
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="COMPOSITE_PK_MISMATCH",
                    location=cpk.table,
                    message=f"Composite PK columns {cpk.cols} do not match "
                    f"table primary_key {table.primary_key}",
                    details={
                        "table": cpk.table,
                        "composite_pk": cpk.cols,
                        "table_pk": table.primary_key,
                    },
                )
            )

    # Check FD constraints
    for fd in board.logical_ir.constraints.fds:
        if fd.table not in tables:
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="FD_TABLE_MISSING",
                    location=fd.table,
                    message=f"FD constraint references unknown table '{fd.table}'",
                    details={"table": fd.table, "lhs": fd.lhs, "rhs": fd.rhs},
                )
            )
            continue

        table = tables[fd.table]
        # Check LHS columns exist
        for col in fd.lhs:
            if col not in {c.name for c in table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="FD_LHS_COL_MISSING",
                        location=f"{fd.table}.{col}",
                        message=f"FD LHS column '{col}' does not exist in table '{fd.table}'",
                        details={"table": fd.table, "column": col, "lhs": fd.lhs, "rhs": fd.rhs},
                    )
                )
        # Check RHS columns exist
        for col in fd.rhs:
            if col not in {c.name for c in table.columns}:
                issues.append(
                    QaIssue(
                        stage="LogicalIR",
                        code="FD_RHS_COL_MISSING",
                        location=f"{fd.table}.{col}",
                        message=f"FD RHS column '{col}' does not exist in table '{fd.table}'",
                        details={"table": fd.table, "column": col, "lhs": fd.lhs, "rhs": fd.rhs},
                    )
                )

    # Check implication constraints
    for impl in board.logical_ir.constraints.implications:
        if impl.table not in tables:
            issues.append(
                QaIssue(
                    stage="LogicalIR",
                    code="IMPLICATION_TABLE_MISSING",
                    location=impl.table,
                    message=f"Implication constraint references unknown table '{impl.table}'",
                    details={"table": impl.table},
                )
            )

    return issues


def collect_issues(
    validators: List[Callable[["Blackboard"], List[QaIssue]]],
    board: "Blackboard",
) -> List[QaIssue]:
    """
    Concatenate issues from all validators.

    Args:
        validators: List of validator functions
        board: Blackboard to validate

    Returns:
        Combined list of all QaIssue objects
    """
    all_issues: List[QaIssue] = []
    for v in validators:
        all_issues.extend(v(board))
    return all_issues

