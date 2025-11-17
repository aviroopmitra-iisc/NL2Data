"""Type enforcement utilities for column generation."""

import numpy as np
import pandas as pd
from typing import Optional
from nl2data.generation.constants import (
    RUSH_HOUR_MORNING,
    RUSH_HOUR_EVENING,
    DURATION_BASE_MIN,
    DURATION_BASE_MAX,
    DURATION_RUSH_MIN,
    DURATION_RUSH_MAX,
    DISTANCE_BASE_MIN,
    DISTANCE_BASE_MAX,
    DISTANCE_RUSH_MIN,
    DISTANCE_RUSH_MAX,
    SURGE_BASE_MIN,
    SURGE_BASE_MAX,
    SURGE_RUSH_MIN,
    SURGE_RUSH_MAX,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def _extract_datetime_to_int(arr: np.ndarray, col_name: Optional[str] = None) -> np.ndarray:
    """Extract integer value from datetime array based on column name."""
    if col_name and ('hour' in col_name.lower() or 'time' in col_name.lower()):
        return np.array([int(pd.Timestamp(d).hour) for d in arr], dtype=np.int64)
    # Default: use day of year
    return np.array([int(pd.Timestamp(d).timetuple().tm_yday) for d in arr], dtype=np.int64)


def _extract_datetime_to_float(
    arr: np.ndarray, 
    col_name: Optional[str] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Extract float value from datetime array based on column semantics."""
    if not col_name:
        # Default: extract hour as float
        return np.array([float(pd.Timestamp(d).hour) for d in arr], dtype=np.float64)
    
    col_lower = col_name.lower()
    hours = np.array([pd.Timestamp(d).hour for d in arr])
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Rush hours: morning and evening
    is_rush_hour = (
        ((hours >= RUSH_HOUR_MORNING[0]) & (hours <= RUSH_HOUR_MORNING[1])) |
        ((hours >= RUSH_HOUR_EVENING[0]) & (hours <= RUSH_HOUR_EVENING[1]))
    )
    
    if 'duration' in col_lower or 'minutes' in col_lower:
        base_duration = np.where(
            is_rush_hour,
            rng.uniform(DURATION_RUSH_MIN, DURATION_RUSH_MAX, size=len(arr)),
            rng.uniform(DURATION_BASE_MIN, DURATION_BASE_MAX, size=len(arr))
        )
        return base_duration.astype(np.float64)
    elif 'distance' in col_lower or 'km' in col_lower:
        base_distance = np.where(
            is_rush_hour,
            rng.uniform(DISTANCE_RUSH_MIN, DISTANCE_RUSH_MAX, size=len(arr)),
            rng.uniform(DISTANCE_BASE_MIN, DISTANCE_BASE_MAX, size=len(arr))
        )
        return base_distance.astype(np.float64)
    elif 'surge' in col_lower or 'multiplier' in col_lower:
        base_surge = np.where(
            is_rush_hour,
            rng.uniform(SURGE_RUSH_MIN, SURGE_RUSH_MAX, size=len(arr)),
            rng.uniform(SURGE_BASE_MIN, SURGE_BASE_MAX, size=len(arr))
        )
        return base_surge.astype(np.float64)
    
    # Default: extract hour as float
    return np.array([float(pd.Timestamp(d).hour) for d in arr], dtype=np.float64)


def enforce_column_type(
    arr: np.ndarray, 
    sql_type: str, 
    col_name: Optional[str] = None, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Enforce SQL type on sampled array.
    
    Args:
        arr: Sampled array
        sql_type: Target SQL type
        col_name: Optional column name for better error messages and semantic handling
        rng: Optional random number generator for semantic transformations
        
    Returns:
        Array with correct type
    """
    if sql_type == "INT":
        # Cast to integer (handles float IDs from uniform distributions)
        if arr.dtype.kind == 'f':  # float
            return arr.astype(np.int64)
        elif arr.dtype.kind == 'M':  # datetime64
            # If seasonal was used on an INT column, extract numeric value
            return _extract_datetime_to_int(arr, col_name)
        return arr.astype(np.int64)
    
    elif sql_type == "FLOAT":
        if arr.dtype.kind == 'M':  # datetime64
            # If seasonal was used on a FLOAT column, extract numeric value
            return _extract_datetime_to_float(arr, col_name, rng)
        return arr.astype(np.float64)
    
    elif sql_type in ("DATE", "DATETIME"):
        # Already datetime64, return as is
        return arr
    
    # For other types, return as is
    return arr

