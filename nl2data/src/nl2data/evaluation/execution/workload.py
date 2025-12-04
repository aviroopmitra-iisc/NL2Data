"""Workload evaluation using DuckDB."""

from typing import List, Dict, Any, Optional
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.evaluation.execution.stats import gini_coefficient, top_k_share
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


def _generate_query(spec, ir: DatasetIR) -> str:
    """
    Generate SQL query from workload specification.
    
    Args:
        spec: WorkloadSpec
        ir: DatasetIR
        
    Returns:
        SQL query string
    """
    # Simplified query generation
    # In production, use spec.query_hint or spec.query directly
    
    if hasattr(spec, "query") and spec.query:
        return spec.query
    
    if hasattr(spec, "query_hint") and spec.query_hint:
        return spec.query_hint
    
    # Fallback: generate basic query based on type
    if spec.type == "group_by":
        # Simple group by query
        table_name = list(ir.logical.tables.keys())[0] if ir.logical.tables else "table"
        return f"SELECT COUNT(*) as cnt FROM {table_name} GROUP BY 1"
    
    # Default: count query
    table_name = list(ir.logical.tables.keys())[0] if ir.logical.tables else "table"
    return f"SELECT COUNT(*) as cnt FROM {table_name}"


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
        
        # Generate SQL query
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
    
    # Close connection
    try:
        conn.close()
    except Exception:
        pass
    
    return results
