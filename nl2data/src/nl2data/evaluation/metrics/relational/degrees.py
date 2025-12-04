"""Degree distribution metrics."""

from typing import Tuple, Optional
import numpy as np
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
