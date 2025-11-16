"""Referential integrity checks."""

from typing import Dict
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def fk_coverage(
    ir: DatasetIR, dfs: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Check foreign key referential integrity coverage.

    Returns the fraction of FK values that exist in referenced tables.

    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame

    Returns:
        Dictionary mapping "table.column" -> coverage fraction (0.0 to 1.0)
    """
    coverage = {}

    for table_name, table in ir.logical.tables.items():
        if table_name not in dfs:
            logger.warning(f"Table '{table_name}' not found in dataframes")
            continue

        if not table.foreign_keys:
            continue

        fact_df = dfs[table_name]

        for fk in table.foreign_keys:
            if fk.column not in fact_df.columns:
                logger.warning(
                    f"FK column '{fk.column}' not found in table '{table_name}'"
                )
                continue

            if fk.ref_table not in dfs:
                logger.warning(
                    f"Referenced table '{fk.ref_table}' not found in dataframes"
                )
                continue

            dim_df = dfs[fk.ref_table]
            if fk.ref_column not in dim_df.columns:
                logger.warning(
                    f"Referenced column '{fk.ref_column}' not found in "
                    f"table '{fk.ref_table}'"
                )
                continue

            # Calculate coverage
            fk_values = fact_df[fk.column]
            ref_values = set(dim_df[fk.ref_column].unique())
            coverage_frac = fk_values.isin(ref_values).mean()
            key = f"{table_name}.{fk.column}"
            coverage[key] = float(coverage_frac)

            if coverage_frac < 0.999:
                logger.warning(
                    f"FK coverage for {key}: {coverage_frac:.4f} "
                    f"(expected >= 0.999)"
                )

    return coverage

