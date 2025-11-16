"""Schema validation checks."""

from typing import List
from nl2data.ir.dataset import DatasetIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def check_pk_fk(ir: DatasetIR) -> List[str]:
    """
    Check primary key and foreign key constraints.

    Args:
        ir: Dataset IR

    Returns:
        List of issue messages (empty if all checks pass)
    """
    issues = []
    tables = ir.logical.tables

    for table_name, table in tables.items():
        # Check primary key exists
        if not table.primary_key:
            issues.append(f"{table_name}: missing primary key")
            continue

        # Check primary key columns exist
        column_names = {c.name for c in table.columns}
        for pk_col in table.primary_key:
            if pk_col not in column_names:
                issues.append(
                    f"{table_name}: primary key column '{pk_col}' does not exist"
                )

        # Check foreign keys
        for fk in table.foreign_keys:
            # Check FK column exists
            if fk.column not in column_names:
                issues.append(
                    f"{table_name}: foreign key column '{fk.column}' does not exist"
                )
                continue

            # Check referenced table exists
            if fk.ref_table not in tables:
                issues.append(
                    f"{table_name}: foreign key '{fk.column}' references "
                    f"missing table '{fk.ref_table}'"
                )
                continue

            # Check referenced column exists
            ref_table = tables[fk.ref_table]
            ref_column_names = {c.name for c in ref_table.columns}
            if fk.ref_column not in ref_column_names:
                issues.append(
                    f"{table_name}: foreign key '{fk.column}' references "
                    f"'{fk.ref_table}.{fk.ref_column}' which does not exist"
                )

    if issues:
        logger.warning(f"Schema validation found {len(issues)} issues")
    else:
        logger.info("Schema validation passed")

    return issues

