"""Relational evaluation metrics (FK coverage, degree distributions, join selectivity)."""

from typing import Dict, Tuple, Optional
import numpy as np
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

# Lazy import for SciPy (for Wasserstein on histograms)
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    _MISSING_SCIPY = ImportError(
        "SciPy is not installed. Install with: pip install nl2data[eval]"
    )


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


def degree_histogram(
    con,
    child_table: str,
    fk_column: str,
    parent_table: str,
    pk_column: str = "id",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of children per parent (degree distribution).

    Args:
        con: DuckDB connection
        child_table: Child table name
        fk_column: Foreign key column name in child table
        parent_table: Parent table name
        pk_column: Primary key column name in parent table

    Returns:
        Tuple of (histogram counts, bin edges)
    """
    if not DUCKDB_AVAILABLE:
        raise RuntimeError(
            "degree_histogram requires DuckDB. Install with: pip install nl2data[eval]"
        )

    try:
        # Compute degrees (children per parent)
        query = f"""
            SELECT p.{pk_column} as pid, COUNT(c.{fk_column}) as deg
            FROM {parent_table} p
            LEFT JOIN {child_table} c ON p.{pk_column} = c.{fk_column}
            GROUP BY p.{pk_column}
        """
        df = con.sql(query).df()

        if len(df) == 0:
            return np.array([]), np.array([])

        degrees = df["deg"].values

        # Compute histogram
        hist, bins = np.histogram(degrees, bins="auto")
        return hist, bins

    except Exception as e:
        logger.error(f"Error computing degree histogram: {e}")
        return np.array([]), np.array([])


def degree_distribution_divergence(
    real_hist: Tuple[np.ndarray, np.ndarray],
    synth_hist: Tuple[np.ndarray, np.ndarray],
) -> float:
    """
    Compute Wasserstein distance between degree distributions.

    Args:
        real_hist: Real degree histogram (counts, bins)
        synth_hist: Synthetic degree histogram (counts, bins)

    Returns:
        Wasserstein distance (lower is better)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "degree_distribution_divergence requires SciPy. Install with: pip install nl2data[eval]"
        )

    real_counts, real_bins = real_hist
    synth_counts, synth_bins = synth_hist

    if len(real_counts) == 0 or len(synth_counts) == 0:
        return 0.0

    # Convert histograms to samples for Wasserstein distance
    # Use bin centers as values
    real_centers = (real_bins[:-1] + real_bins[1:]) / 2
    synth_centers = (synth_bins[:-1] + synth_bins[1:]) / 2

    # Expand counts to samples
    real_samples = np.repeat(real_centers, real_counts.astype(int))
    synth_samples = np.repeat(synth_centers, synth_counts.astype(int))

    if len(real_samples) == 0 or len(synth_samples) == 0:
        return 0.0

    w1 = wasserstein_distance(real_samples, synth_samples)
    return float(w1)


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


def evaluate_relational_metrics(
    ir: DatasetIR,
    dfs: Dict[str, pd.DataFrame],
    con: Optional[any] = None,
) -> Dict[str, float]:
    """
    Evaluate all relational metrics for a dataset.

    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame
        con: Optional DuckDB connection (creates one if not provided)

    Returns:
        Dictionary of metric names -> values
    """
    if not DUCKDB_AVAILABLE:
        raise RuntimeError(
            "evaluate_relational_metrics requires DuckDB. Install with: pip install nl2data[eval]"
        )

    metrics = {}

    # Create DuckDB connection if not provided
    if con is None:
        con = duckdb.connect()

    # Register DataFrames as tables
    for table_name, df in dfs.items():
        con.register(table_name, df)

    # FK coverage for all foreign keys
    fk_coverages = []
    for table_name, table in ir.logical.tables.items():
        if table_name not in dfs:
            continue

        for fk in table.foreign_keys:
            if fk.ref_table not in dfs:
                continue

            # Get PK column from referenced table
            ref_table = ir.logical.tables[fk.ref_table]
            pk_col = ref_table.primary_key[0] if ref_table.primary_key else fk.ref_column

            coverage = fk_coverage_duckdb(
                con, table_name, fk.column, fk.ref_table, pk_col
            )
            fk_coverages.append(coverage)
            metrics[f"{table_name}.{fk.column}_coverage"] = coverage

    if fk_coverages:
        metrics["avg_fk_coverage"] = float(np.mean(fk_coverages))
        metrics["min_fk_coverage"] = float(np.min(fk_coverages))

    # Degree distributions for fact->dimension joins
    for table_name, table in ir.logical.tables.items():
        if table.kind != "fact" or table_name not in dfs:
            continue

        for fk in table.foreign_keys:
            if fk.ref_table not in dfs:
                continue

            ref_table = ir.logical.tables[fk.ref_table]
            pk_col = ref_table.primary_key[0] if ref_table.primary_key else fk.ref_column

            try:
                hist = degree_histogram(con, table_name, fk.column, fk.ref_table, pk_col)
                if len(hist[0]) > 0:
                    # Store histogram statistics
                    degrees = np.repeat(
                        (hist[1][:-1] + hist[1][1:]) / 2, hist[0].astype(int)
                    )
                    metrics[f"{table_name}.{fk.column}_avg_degree"] = float(np.mean(degrees))
                    metrics[f"{table_name}.{fk.column}_max_degree"] = float(np.max(degrees))
            except Exception as e:
                logger.warning(f"Error computing degree histogram for {table_name}.{fk.column}: {e}")

    # Close connection if we created it
    if con is not None:
        try:
            con.close()
        except:
            pass

    return metrics

