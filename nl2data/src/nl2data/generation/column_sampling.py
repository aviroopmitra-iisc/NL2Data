"""Common column sampling utilities."""

import numpy as np
import pandas as pd
from typing import Optional
from nl2data.ir.logical import ColumnSpec
from nl2data.ir.generation import DistUniform
from nl2data.generation.constants import (
    DEFAULT_INT_RANGE,
    DEFAULT_DATE_RANGE_DAYS,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def sample_fallback_column(
    col: ColumnSpec,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample column values using SQL type-based fallback when no distribution is specified.
    
    Args:
        col: Column specification
        n: Number of samples
        rng: Random number generator
        
    Returns:
        Array of sampled values
    """
    if col.sql_type in ("INT32", "INT64"):
        return rng.integers(DEFAULT_INT_RANGE[0], DEFAULT_INT_RANGE[1], size=n, dtype=np.int64)
    if col.sql_type in ("FLOAT32", "FLOAT64"):
        return rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
    if col.sql_type == "TEXT":
        return np.array([f"{col.name}_{i}" for i in range(n)])
    if col.sql_type == "BOOL":
        return rng.choice([True, False], size=n)
    if col.sql_type in ("DATE", "DATETIME"):
        base = np.datetime64("2020-01-01")
        days = rng.integers(0, DEFAULT_DATE_RANGE_DAYS, size=n)
        return base + days.astype("timedelta64[D]")
    
    # Default fallback
    logger.warning(
        f"No fallback for type {col.sql_type}, using text fallback"
    )
    return np.array([f"{col.name}_{i}" for i in range(n)])


def sample_primary_key_column(
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample primary key column ensuring uniqueness.
    
    Args:
        col: Column specification (must be primary key)
        dist: Distribution specification (optional)
        n: Number of samples
        rng: Random number generator
        
    Returns:
        Array of unique primary key values
    """
    if col.sql_type not in ("INT32", "INT64"):
        # For non-integer PKs, use sequential IDs (fallback)
        # Note: Full sampling logic is in dim_generator to avoid circular imports
        return np.arange(1, n + 1, dtype=np.int64)
    
    # For integer PKs, generate unique sequential or random IDs
    if isinstance(dist, DistUniform) and dist.low is not None and dist.high is not None:
        # Use range-based unique IDs
        low = int(dist.low)
        high = int(dist.high)
        if high - low >= n:
            # Enough range - sample without replacement
            ids = rng.choice(range(low, high + 1), size=n, replace=False)
            return ids.astype(np.int64)
        else:
            # Not enough range - use sequential IDs starting from low
            ids = np.arange(low, low + n, dtype=np.int64)
            return ids
    
    # Fallback: sequential IDs
    return np.arange(1, n + 1, dtype=np.int64)

