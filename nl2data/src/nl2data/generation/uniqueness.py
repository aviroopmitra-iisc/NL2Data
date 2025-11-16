"""Uniqueness enforcement utilities for column generation."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from nl2data.ir.logical import ColumnSpec
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


# Patterns for identifying category/type columns that should be unique
CATEGORY_PATTERNS = [
    'type_name', 'type_', '_type',  # type_name, vehicle_type, etc.
    'category', '_category', 'category_',
    '_code', 'code_',  # status_code, etc.
    'zone_name', 'region_name',  # geographic identifiers
    'status', 'class', 'kind'  # classification columns
]

# Patterns for person name columns (these can have duplicates)
PERSON_NAME_PATTERNS = [
    'rider_name', 'driver_name', 'customer_name', 'user_name', 
    'person_name', 'employee_name', 'first_name', 'last_name'
]


def is_category_column(col_name: str) -> bool:
    """
    Check if a column name matches patterns for category/type columns.
    
    Args:
        col_name: Column name to check
        
    Returns:
        True if column appears to be a category/type identifier
    """
    col_lower = col_name.lower()
    return any(pattern in col_lower for pattern in CATEGORY_PATTERNS)


def is_person_name_column(col_name: str) -> bool:
    """
    Check if a column name matches patterns for person name columns.
    
    Args:
        col_name: Column name to check
        
    Returns:
        True if column appears to be a person name
    """
    col_lower = col_name.lower()
    return any(pattern in col_lower for pattern in PERSON_NAME_PATTERNS)


def enforce_unique_categorical_column(
    df: pd.DataFrame,
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, int]:
    """
    Enforce uniqueness on a categorical column with known domain.
    
    Args:
        df: DataFrame to modify
        col: Column specification
        dist: Distribution specification (should have domain attribute)
        n: Original row count
        rng: Random number generator
        
    Returns:
        Tuple of (modified DataFrame, new row count)
    """
    if not (dist and hasattr(dist, 'domain') and hasattr(dist.domain, 'values')):
        # No categorical domain - drop duplicates
        df = df.drop_duplicates(subset=[col.name], keep="first")
        if len(df) < n:
            logger.warning(
                f"Column {col.name} marked as unique but "
                f"only {len(df)} unique values generated. Table reduced to {len(df)} rows."
            )
        return df, len(df)
    
    # Categorical with known domain - sample without replacement
    all_values = list(dist.domain.values)
    if len(all_values) >= n:
        # We have enough unique values, sample n without replacement
        sampled = rng.choice(all_values, size=n, replace=False)
        df[col.name] = sampled
        return df, n
    else:
        # Not enough unique values in domain - keep only unique rows
        df = df.drop_duplicates(subset=[col.name], keep="first")
        logger.warning(
            f"Column {col.name} marked as unique but "
            f"only {len(all_values)} unique values in domain. "
            f"Reduced table to {len(df)} rows."
        )
        return df, len(df)


def enforce_unique_non_text_column(
    df: pd.DataFrame,
    col: ColumnSpec,
    n: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Enforce uniqueness on a non-text column.
    
    Args:
        df: DataFrame to modify
        col: Column specification
        n: Original row count
        
    Returns:
        Tuple of (modified DataFrame, new row count)
    """
    df = df.drop_duplicates(subset=[col.name], keep="first")
    if len(df) < n:
        logger.warning(
            f"Column {col.name} marked as unique but "
            f"only {len(df)} unique values generated. Table reduced to {len(df)} rows."
        )
    return df, len(df)

