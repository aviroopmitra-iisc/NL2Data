"""Foreign key integrity metrics."""

from typing import Dict
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy import for DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None
    _MISSING_DUCKDB = ImportError(
        "DuckDB is not installed. Install with: pip install nl2data[eval]"
    )


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


def fk_coverage_duckdb(
    con,
    child_table: str,
    fk_column: str,
    parent_table: str,
    pk_column: str = "id",
) -> float:
    """
    Compute fraction of FK values that have valid references using DuckDB.
    
    Args:
        con: DuckDB connection
        child_table: Child table name
        fk_column: Foreign key column name in child table
        parent_table: Parent table name
        pk_column: Primary key column name in parent table (default: "id")
        
    Returns:
        Coverage fraction (0.0 to 1.0)
    """
    if not DUCKDB_AVAILABLE:
        raise RuntimeError(
            "fk_coverage_duckdb requires DuckDB. Install with: pip install nl2data[eval]"
        )
    
    try:
        # Total rows in child table
        tot_result = con.sql(f"SELECT COUNT(*) as cnt FROM {child_table}").fetchone()
        if tot_result is None:
            return 0.0
        tot = tot_result[0]
        
        if tot == 0:
            return 1.0
        
        # Valid FK references
        val_result = con.sql(f"""
            SELECT COUNT(*) as cnt
            FROM {child_table} c
            WHERE EXISTS (
                SELECT 1 FROM {parent_table} p
                WHERE p.{pk_column} = c.{fk_column}
            )
        """).fetchone()
        
        if val_result is None:
            return 0.0
        val = val_result[0]
        
        return float(val / tot)
    
    except Exception as e:
        logger.error(f"Error computing FK coverage: {e}")
        return 0.0
