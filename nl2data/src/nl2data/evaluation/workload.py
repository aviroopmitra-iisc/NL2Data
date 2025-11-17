"""Workload evaluation using DuckDB."""

from typing import List, Dict, Any, Optional
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.evaluation.stats import gini_coefficient, top_k_share
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


def run_workloads(
    ir: DatasetIR, dfs: Dict[str, pd.DataFrame]
) -> List[Dict[str, Any]]:
    """
    Run workload queries and collect metrics.

    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame

    Returns:
        List of workload results

    Raises:
        RuntimeError: If DuckDB is not installed
    """
    if not DUCKDB_AVAILABLE:
        raise RuntimeError(
            "run_workloads requires DuckDB. Install with: pip install nl2data[eval]"
        )

    if not ir.workload or not ir.workload.targets:
        logger.info("No workload specifications found")
        return []

    # Create DuckDB connection
    conn = duckdb.connect()
    logger.info("Initialized DuckDB connection")

    # Register DataFrames as tables
    for table_name, df in dfs.items():
        conn.register(table_name, df)
        logger.debug(f"Registered table '{table_name}' with {len(df)} rows")

    results = []

    for i, spec in enumerate(ir.workload.targets, 1):
        logger.info(f"Running workload {i}/{len(ir.workload.targets)}: {spec.type}")

        # Generate SQL query (simplified - in production, use spec.query_hint)
        sql = _generate_query(spec, ir)

        try:
            import time

            start_time = time.time()
            result_df = conn.execute(sql).df()
            elapsed_sec = time.time() - start_time

            # Calculate metrics
            group_gini = None
            top1_share = None

            if spec.type == "group_by" and len(result_df) > 0:
                # Assume first numeric column is the count/aggregate
                numeric_cols = result_df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    counts = result_df[numeric_cols[0]].values
                    group_gini = gini_coefficient(counts)
                    top1_share = top_k_share(counts, k=1)

            results.append(
                {
                    "sql": sql,
                    "type": spec.type,
                    "elapsed_sec": elapsed_sec,
                    "rows": len(result_df),
                    "group_gini": group_gini,
                    "top1_share": top1_share,
                }
            )

            logger.info(
                f"Workload {i} completed: {elapsed_sec:.3f}s, {len(result_df)} rows"
            )

        except Exception as e:
            logger.error(f"Workload {i} failed: {e}", exc_info=True)
            results.append(
                {
                    "sql": sql,
                    "type": spec.type,
                    "elapsed_sec": 0.0,
                    "rows": 0,
                    "error": str(e),
                }
            )

    conn.close()
    return results


def _generate_query(spec, ir: DatasetIR) -> str:
    """
    Generate SQL query from workload spec.

    This is a simplified implementation. In production, this would
    use spec.query_hint and spec.join_graph more intelligently.

    Args:
        spec: WorkloadSpec
        ir: Dataset IR

    Returns:
        SQL query string
    """
    if spec.query_hint:
        return spec.query_hint

    # Generate basic query based on type
    tables = list(ir.logical.tables.keys())
    fact_tables = [
        name
        for name, t in ir.logical.tables.items()
        if t.kind == "fact"
    ]
    dim_tables = [
        name
        for name, t in ir.logical.tables.items()
        if t.kind == "dimension"
    ]

    if spec.type == "group_by":
        if fact_tables:
            table = fact_tables[0]
            # Find a dimension FK column
            fact_table = ir.logical.tables[fact_tables[0]]
            fk_cols = [fk.column for fk in fact_table.foreign_keys]
            if fk_cols:
                group_col = fk_cols[0]
                return f"SELECT {group_col}, COUNT(*) as cnt FROM {table} GROUP BY {group_col}"
            else:
                # Fallback: group by first column
                first_col = fact_table.columns[0].name
                return f"SELECT {first_col}, COUNT(*) as cnt FROM {table} GROUP BY {first_col}"

    elif spec.type == "join":
        if fact_tables and dim_tables:
            fact = fact_tables[0]
            dim = dim_tables[0]
            fact_table = ir.logical.tables[fact]
            # Find FK to dim
            for fk in fact_table.foreign_keys:
                if fk.ref_table == dim:
                    return (
                        f"SELECT * FROM {fact} JOIN {dim} "
                        f"ON {fact}.{fk.column} = {dim}.{fk.ref_column} LIMIT 1000"
                    )
            # Fallback
            return f"SELECT * FROM {fact} JOIN {dim} LIMIT 1000"

    elif spec.type == "filter":
        if fact_tables:
            table = fact_tables[0]
            fact_table = ir.logical.tables[table]
            # Use first numeric column for filter
            numeric_cols = [
                c.name
                for c in fact_table.columns
                if c.sql_type in ("INT", "FLOAT")
            ]
            if numeric_cols:
                col = numeric_cols[0]
                return f"SELECT * FROM {table} WHERE {col} > 100 LIMIT 1000"

    # Fallback
    if tables:
        return f"SELECT * FROM {tables[0]} LIMIT 100"

    return "SELECT 1"

