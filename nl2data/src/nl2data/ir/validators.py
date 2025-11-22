"""Validators for IR models."""

from dataclasses import dataclass, field
from typing import Literal, Callable, List, Optional, TYPE_CHECKING
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
    from nl2data.ir.generation import DistDerived, DistWindow
    from nl2data.generation.derived_program import compile_derived
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistWindow):
            # Validate window specifications
            dist = cg.distribution
            table = tables.get(cg.table)
            if table:
                # Check order_by column exists
                if dist.order_by not in {c.name for c in table.columns}:
                    issues.append(
                        QaIssue(
                            stage="GenerationIR",
                            code="WINDOW_ORDER_BY_MISSING",
                            location=f"{cg.table}.{cg.column}",
                            message=f"Window column '{cg.table}.{cg.column}' references order_by column '{dist.order_by}' which does not exist in table '{cg.table}'.",
                            details={
                                "table": cg.table,
                                "column": cg.column,
                                "order_by": dist.order_by,
                            },
                        )
                    )
                
                # Check partition_by columns exist
                for part_col in dist.partition_by:
                    if part_col not in {c.name for c in table.columns}:
                        issues.append(
                            QaIssue(
                                stage="GenerationIR",
                                code="WINDOW_PARTITION_BY_MISSING",
                                location=f"{cg.table}.{cg.column}",
                                message=f"Window column '{cg.table}.{cg.column}' references partition_by column '{part_col}' which does not exist in table '{cg.table}'.",
                                details={
                                    "table": cg.table,
                                    "column": cg.column,
                                    "partition_by": part_col,
                                },
                            )
                        )
        elif isinstance(cg.distribution, DistDerived):
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


def check_nuance_coverage(nl_text: Optional[str], ir: DatasetIR) -> List[QaIssue]:
    """
    Check if NL requirements are reflected in the IR (nuance coverage).
    
    Parses NL text for keywords and verifies that corresponding IR constructs exist.
    
    Args:
        nl_text: Natural language description (optional)
        ir: DatasetIR to check
    
    Returns:
        List of QaIssue objects for missing nuances
    """
    issues: List[QaIssue] = []
    
    if not nl_text:
        return issues
    
    nl_lower = nl_text.lower()
    
    # Expanded keyword to IR construct mappings
    nuance_checks = {
        "rolling": {
            "keywords": ["rolling", "rolling mean", "rolling average", "rolling sum", "rolling count", "moving average", "moving window"],
            "check": lambda ir: _has_window_operations(ir),
            "construct": "window operations (DistWindow or rolling functions)"
        },
        "window": {
            "keywords": ["window", "sliding window", "moving window", "last n days", "last 7 days", "last 30 days"],
            "check": lambda ir: _has_window_operations(ir),
            "construct": "window operations (DistWindow)"
        },
        "lag": {
            "keywords": ["lag", "previous value", "prior value"],
            "check": lambda ir: _has_lag_lead(ir),
            "construct": "lag/lead functions"
        },
        "lead": {
            "keywords": ["lead", "next value", "following value"],
            "check": lambda ir: _has_lag_lead(ir),
            "construct": "lead functions"
        },
        "incident": {
            "keywords": ["incident", "event", "disruption", "outage", "failure", "storm", "campaign"],
            "check": lambda ir: _has_events(ir),
            "construct": "EventSpec"
        },
        "heavy_tail": {
            "keywords": ["heavy tail", "heavy-tailed", "heavy tailed", "right-skewed", "right skewed", "skewed"],
            "check": lambda ir: _has_heavy_tail_distribution(ir),
            "construct": "lognormal or pareto distribution"
        },
        "log-normal": {
            "keywords": ["log-normal", "lognormal", "log normal"],
            "check": lambda ir: _has_lognormal(ir),
            "construct": "lognormal distribution (DistLognormal)"
        },
        "pareto": {
            "keywords": ["pareto", "power-law", "power law", "80/20", "80-20", "80 20"],
            "check": lambda ir: _has_pareto(ir),
            "construct": "pareto distribution (DistPareto)"
        },
        "mixture": {
            "keywords": ["mixture", "multi-modal", "multimodal", "base traffic + spikes", "normal + incident"],
            "check": lambda ir: _has_mixture(ir),
            "construct": "mixture distribution (DistMixture)"
        },
        "zipf": {
            "keywords": ["zipf", "zipfian"],
            "check": lambda ir: _has_zipf(ir),
            "construct": "Zipf distribution"
        },
        "proration": {
            "keywords": ["proration", "prorate", "overlap", "interval"],
            "check": lambda ir: _has_proration(ir),
            "construct": "overlap_days() or interval functions"
        },
        "pay-day": {
            "keywords": ["pay-day", "payday", "pay day", "salary day", "15th", "1st of month"],
            "check": lambda ir: _has_seasonal_patterns(ir),
            "construct": "seasonal patterns or day-of-month logic"
        },
        "surge": {
            "keywords": ["surge", "peak", "spike", "burst", "traffic spike"],
            "check": lambda ir: _has_surge_pattern(ir),
            "construct": "mixture distribution or event-driven patterns"
        },
        "churn": {
            "keywords": ["churn", "cancellation", "attrition", "within 30 days"],
            "check": lambda ir: _has_churn_pattern(ir),
            "construct": "conditional logic or derived columns"
        },
        "fraud": {
            "keywords": ["fraud", "fraudulent", "anomaly", "anomalous"],
            "check": lambda ir: _has_fraud_pattern(ir),
            "construct": "mixture distribution or event-driven patterns"
        },
        "seasonal": {
            "keywords": ["seasonal", "seasonality", "holiday", "weekend"],
            "check": lambda ir: _has_seasonal_patterns(ir),
            "construct": "seasonal distribution or day-of-week logic"
        },
    }
    
    # Check each nuance
    for nuance_name, check_info in nuance_checks.items():
        # Check if any keyword is present in NL text
        keyword_found = any(kw in nl_lower for kw in check_info["keywords"])
        
        if keyword_found:
            # Check if corresponding construct exists in IR
            if not check_info["check"](ir):
                issues.append(
                    QaIssue(
                        stage="GenerationIR",
                        code="MISSING_NUANCE",
                        location="nuance_coverage",
                        message=f"NL mentions '{nuance_name}' but IR lacks {check_info['construct']}",
                        details={
                            "nuance": nuance_name,
                            "keywords_found": [kw for kw in check_info["keywords"] if kw in nl_lower],
                            "missing_construct": check_info["construct"]
                        },
                    )
                )
    
    return issues


def _has_window_operations(ir: DatasetIR) -> bool:
    """Check if IR has window operations."""
    from nl2data.ir.generation import DistWindow
    from nl2data.generation.derived_registry import build_derived_registry
    
    # Check for DistWindow in generation specs
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistWindow):
            return True
    
    # Check for window functions in derived expressions
    try:
        reg = build_derived_registry(ir)
        if reg.windows:
            return True
    except (ValueError, KeyError, AttributeError):
        # Registry build failed or no windows found - continue checking
        pass
    
    # Check for window function names in derived expressions
    for cg in ir.generation.columns:
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if any(func in expr for func in ["rolling_mean", "rolling_sum", "rolling_count", "lag(", "lead("]):
                return True
    
    return False


def _has_lag_lead(ir: DatasetIR) -> bool:
    """Check if IR has lag/lead functions."""
    for cg in ir.generation.columns:
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if "lag(" in expr or "lead(" in expr:
                return True
    return False


def _has_events(ir: DatasetIR) -> bool:
    """Check if IR has events."""
    return len(ir.generation.events) > 0


def _has_lognormal(ir: DatasetIR) -> bool:
    """Check if IR has lognormal distribution."""
    from nl2data.ir.generation import DistLognormal
    
    # Check for DistLognormal in generation specs
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistLognormal):
            return True
    
    # Check for lognormal in derived expressions
    for cg in ir.generation.columns:
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if "lognormal(" in expr:
                return True
    
    return False


def _has_pareto(ir: DatasetIR) -> bool:
    """Check if IR has pareto distribution."""
    from nl2data.ir.generation import DistPareto
    
    # Check for DistPareto in generation specs
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistPareto):
            return True
    
    # Check for pareto in derived expressions
    for cg in ir.generation.columns:
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if "pareto(" in expr:
                return True
    
    return False


def _has_mixture(ir: DatasetIR) -> bool:
    """Check if IR has mixture distribution."""
    from nl2data.ir.generation import DistMixture
    
    # Check for DistMixture in generation specs
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistMixture):
            return True
    
    return False


def _has_heavy_tail_distribution(ir: DatasetIR) -> bool:
    """Check if IR has any heavy-tail distribution (lognormal or pareto)."""
    return _has_lognormal(ir) or _has_pareto(ir)


def _has_surge_pattern(ir: DatasetIR) -> bool:
    """Check if IR has surge/spike patterns (mixture or events)."""
    return _has_mixture(ir) or _has_events(ir)


def _has_churn_pattern(ir: DatasetIR) -> bool:
    """Check if IR has churn patterns (derived columns with conditional logic)."""
    from nl2data.ir.generation import DistDerived
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistDerived):
            expr = str(cg.distribution.expression).lower()
            # Check for conditional logic that might indicate churn
            if "where(" in expr or "if" in expr or "days(" in expr:
                return True
    
    return False


def _has_fraud_pattern(ir: DatasetIR) -> bool:
    """Check if IR has fraud patterns (mixture or events)."""
    return _has_mixture(ir) or _has_events(ir)


def _has_zipf(ir: DatasetIR) -> bool:
    """Check if IR has Zipf distribution."""
    from nl2data.ir.generation import DistZipf
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistZipf):
            return True
    return False


def _has_proration(ir: DatasetIR) -> bool:
    """Check if IR has proration/overlap functions."""
    for cg in ir.generation.columns:
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if "overlap_days" in expr or "prorate" in expr or "interval" in expr:
                return True
    return False


def _has_seasonal_patterns(ir: DatasetIR) -> bool:
    """Check if IR has seasonal patterns."""
    from nl2data.ir.generation import DistSeasonal
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistSeasonal):
            return True
        
        # Check for day-of-month logic in derived expressions
        if hasattr(cg.distribution, 'expression'):
            expr = str(cg.distribution.expression).lower()
            if "day_of_month" in expr or "day_of_week" in expr:
                return True
    
    return False


def validate_dataset(ir: DatasetIR, nl_text: Optional[str] = None) -> List[QaIssue]:
    """
    Validate complete dataset IR.

    Args:
        ir: DatasetIR to validate
        nl_text: Optional natural language description for nuance coverage check

    Returns:
        List of QaIssue objects (empty if validation passes)
    """
    issues: List[QaIssue] = []
    issues.extend(validate_logical(ir))
    issues.extend(validate_generation(ir))
    issues.extend(validate_derived_columns(ir))
    
    # Check nuance coverage if NL text provided
    if nl_text:
        issues.extend(check_nuance_coverage(nl_text, ir))

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


def validate_logical_blackboard(board: "Blackboard") -> List[QaIssue]:
    """
    Validate logical IR from blackboard (for repair loop).
    
    Args:
        board: Blackboard with LogicalIR
        
    Returns:
        List of QaIssue objects
    """
    if board.logical_ir is None:
        return []
    
    # Create temporary DatasetIR for validation
    from .dataset import DatasetIR
    from .generation import GenerationIR
    temp_ir = DatasetIR(
        logical=board.logical_ir,
        generation=GenerationIR(columns=[])  # Empty for logical-only validation
    )
    return validate_logical(temp_ir)


def validate_generation_blackboard(board: "Blackboard") -> List[QaIssue]:
    """
    Validate generation IR from blackboard (for repair loop).
    
    Args:
        board: Blackboard with GenerationIR
        
    Returns:
        List of QaIssue objects
    """
    if board.generation_ir is None or board.logical_ir is None:
        return []
    
    # Create temporary DatasetIR for validation
    from .dataset import DatasetIR
    temp_ir = DatasetIR(
        logical=board.logical_ir,
        generation=board.generation_ir
    )
    return validate_generation(temp_ir)


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

