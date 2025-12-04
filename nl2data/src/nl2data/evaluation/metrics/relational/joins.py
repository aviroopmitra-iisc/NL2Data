"""Join selectivity metrics."""

from typing import Optional
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


def join_selectivity(
    con,
    query: str,
    base_rowcount: int,
) -> float:
    """
    Compute join selectivity (fraction of base rows that match).
    
    Args:
        con: DuckDB connection
        query: SQL query that performs the join
        base_rowcount: Number of rows in the base table
        
    Returns:
        Selectivity (0.0 to 1.0, or >1.0 if join produces more rows)
    """
    if not DUCKDB_AVAILABLE:
        raise RuntimeError(
            "join_selectivity requires DuckDB. Install with: pip install nl2data[eval]"
        )
    
    try:
        result = con.sql(f"SELECT COUNT(*) as cnt FROM ({query})").fetchone()
        if result is None:
            return 0.0
        
        join_count = result[0]
        if base_rowcount == 0:
            return 0.0
        
        return float(join_count / base_rowcount)
    
    except Exception as e:
        logger.error(f"Error computing join selectivity: {e}")
        return 0.0
