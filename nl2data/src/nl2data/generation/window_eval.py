"""Window function evaluation for rolling aggregations."""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union
from nl2data.ir.generation import DistWindow, WindowFrame


def parse_window_size(size_str: str) -> Union[pd.Timedelta, int]:
    """
    Parse window size string to timedelta or integer.
    
    Args:
        size_str: String like "7d", "24h", "100" (for ROWS)
    
    Returns:
        pd.Timedelta for time-based windows, int for ROWS-based windows
    """
    size_str = size_str.strip().lower()
    
    # Try to parse as integer (for ROWS)
    try:
        return int(size_str)
    except ValueError:
        pass
    
    # Parse time-based strings
    if size_str.endswith('d') or size_str.endswith('days'):
        num = float(size_str.rstrip('d').rstrip('ays').rstrip('ay'))
        return pd.Timedelta(days=num)
    elif size_str.endswith('h') or size_str.endswith('hours'):
        num = float(size_str.rstrip('h').rstrip('ours'))
        return pd.Timedelta(hours=num)
    elif size_str.endswith('m') or size_str.endswith('minutes'):
        num = float(size_str.rstrip('m').rstrip('inutes'))
        return pd.Timedelta(minutes=num)
    elif size_str.endswith('s') or size_str.endswith('seconds'):
        num = float(size_str.rstrip('s').rstrip('econds'))
        return pd.Timedelta(seconds=num)
    else:
        raise ValueError(f"Unable to parse window size: {size_str}. Expected format: '7d', '24h', '100', etc.")


def eval_window_expression(
    df: pd.DataFrame,
    window_spec: DistWindow,
    expr_col: Optional[str] = None
) -> pd.Series:
    """
    Evaluate a window function on a DataFrame.
    
    Args:
        df: DataFrame to compute window on (will be sorted by order_by)
        window_spec: Window specification
        expr_col: Optional column name if expression is just a column reference
    
    Returns:
        Series with window function results
    """
    frame = window_spec.frame
    order_by = window_spec.order_by
    partition_by = window_spec.partition_by if window_spec.partition_by else []
    expression = window_spec.expression.strip()
    
    # Determine aggregation type and column from expression
    # Patterns: "mean(column)", "sum(column)", "count(*)", "column", "rolling_mean(column, '7d')"
    
    # Helper function to extract column name from function call
    def extract_col_name(expr: str) -> Optional[str]:
        """Extract column name from function call expression."""
        match = re.search(r'\((.*?)[,\)]', expr)
        if match:
            return match.group(1).strip().strip('"').strip("'")
        return None
    
    agg_type = "mean"  # default
    col_name = None
    
    if expression == "count(*)":
        agg_type = "count"
        col_name = None  # Count all rows
    
    if expression.startswith("mean(") or expression.startswith("rolling_mean("):
        agg_type = "mean"
        col_name = extract_col_name(expression)
    elif expression.startswith("sum(") or expression.startswith("rolling_sum("):
        agg_type = "sum"
        col_name = extract_col_name(expression)
    elif expression.startswith("count(") or expression.startswith("rolling_count("):
        agg_type = "count"
        col_name = extract_col_name(expression)
    elif expression.startswith("std(") or expression.startswith("rolling_std("):
        agg_type = "std"
        col_name = extract_col_name(expression)
    elif expression.startswith("lag("):
        agg_type = "lag"
        col_name = extract_col_name(expression)
    elif expression.startswith("lead("):
        agg_type = "lead"
        col_name = extract_col_name(expression)
    else:
        # Assume expression is a column name
        col_name = expression
    
    # Parse window size
    window_size = parse_window_size(frame.preceding)
    
    # Sort DataFrame by order_by (required for window functions)
    df_sorted = df.sort_values(by=order_by).copy()
    original_index = df.index
    df_sorted_index = df_sorted.index
    
    # Get the column to aggregate (after sorting)
    if col_name and col_name in df_sorted.columns:
        agg_col_name = col_name
    elif expr_col and expr_col in df_sorted.columns:
        agg_col_name = expr_col
    elif agg_type == "count" and col_name is None:
        # count(*) - add a dummy column of ones to the DataFrame
        df_sorted["__count__"] = 1
        agg_col_name = "__count__"
    else:
        raise ValueError(
            f"Window expression '{expression}' references column '{col_name}' "
            f"which does not exist. Available columns: {list(df_sorted.columns)}"
        )
    
    # Handle lag/lead specially (they don't use rolling windows)
    if agg_type == "lag" or agg_type == "lead":
        # lag/lead use shift() instead of rolling
        shift_amount = 1  # Default
        # Try to extract shift amount from expression if present
        if agg_type == "lag":
            match = re.search(r'lag\([^,]+,\s*(\d+)', expression)
            if match:
                shift_amount = int(match.group(1))
            shift_amount = -shift_amount  # lag shifts backward
        elif agg_type == "lead":
            match = re.search(r'lead\([^,]+,\s*(\d+)', expression)
            if match:
                shift_amount = int(match.group(1))
            # lead shifts forward (positive)
        
        if partition_by:
            gb = df_sorted.groupby(partition_by, sort=False)
            results = []
            for group_key, group in gb:
                group_sorted = group.sort_values(by=order_by)
                result = group_sorted[agg_col_name].shift(shift_amount)
                results.append(result)
            result_series = pd.concat(results).sort_index()
        else:
            group_sorted = df_sorted.sort_values(by=order_by)
            result_series = group_sorted[agg_col_name].shift(shift_amount)
        
        # Reindex to original DataFrame index
        result_series = result_series.reindex(df_sorted_index)
        result_series = result_series.reindex(original_index)
        return result_series
    
    # Apply window function based on frame type
    if frame.type == "RANGE":
        # RANGE window: time-based
        if not isinstance(window_size, pd.Timedelta):
            raise ValueError(f"RANGE window requires time-based size (e.g., '7d'), got: {frame.preceding}")
        
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[order_by]):
            raise ValueError(
                f"RANGE window requires datetime column for order_by '{order_by}', "
                f"got: {df_sorted[order_by].dtype}"
            )
        
        # Use rolling with time-based window
        if partition_by:
            gb = df_sorted.groupby(partition_by, sort=False)
            results = []
            for group_key, group in gb:
                # Sort by order_by within group (already sorted, but ensure)
                group_sorted = group.sort_values(by=order_by)
                
                # Create rolling window on DataFrame (required for time-based windows)
                rolling = group_sorted.rolling(
                    window=window_size,
                    on=order_by,
                    closed='both'  # Include both start and end of window
                )
                
                # Apply aggregation on the specific column
                if agg_type == "mean":
                    result = rolling[agg_col_name].mean()
                elif agg_type == "sum":
                    result = rolling[agg_col_name].sum()
                elif agg_type == "count":
                    result = rolling[agg_col_name].count()
                elif agg_type == "std":
                    result = rolling[agg_col_name].std()
                else:
                    result = rolling[agg_col_name].mean()
                
                results.append(result)
            
            # Concatenate results and reindex to sorted index
            result_series = pd.concat(results).sort_index()
        else:
            # No partition - apply to entire DataFrame
            group_sorted = df_sorted.sort_values(by=order_by)
            # For time-based rolling, need to call on DataFrame, not Series
            rolling = group_sorted.rolling(
                window=window_size,
                on=order_by,
                closed='both'
            )
            if agg_type == "mean":
                result_series = rolling[agg_col_name].mean()
            elif agg_type == "sum":
                result_series = rolling[agg_col_name].sum()
            elif agg_type == "count":
                result_series = rolling[agg_col_name].count()
            elif agg_type == "std":
                result_series = rolling[agg_col_name].std()
            else:
                result_series = rolling[agg_col_name].mean()
        
    elif frame.type == "ROWS":
        # ROWS window: row-based
        if not isinstance(window_size, int):
            raise ValueError(f"ROWS window requires integer size (e.g., '100'), got: {frame.preceding}")
        
        if partition_by:
            gb = df_sorted.groupby(partition_by, sort=False)
            results = []
            for group_key, group in gb:
                # Sort by order_by within group
                group_sorted = group.sort_values(by=order_by)
                
                # Create rolling window
                rolling = group_sorted[agg_col_name].rolling(
                    window=window_size,
                    min_periods=1
                )
                
                # Apply aggregation
                if agg_type == "mean":
                    result = rolling.mean()
                elif agg_type == "sum":
                    result = rolling.sum()
                elif agg_type == "count":
                    result = rolling.count()
                elif agg_type == "std":
                    result = rolling.std()
                else:
                    result = rolling.mean()
                
                results.append(result)
            
            # Concatenate results
            result_series = pd.concat(results).sort_index()
        else:
            # No partition - apply to entire DataFrame
            group_sorted = df_sorted.sort_values(by=order_by)
            rolling = group_sorted[agg_col_name].rolling(
                window=window_size,
                min_periods=1
            )
            if agg_type == "mean":
                result_series = rolling.mean()
            elif agg_type == "sum":
                result_series = rolling.sum()
            elif agg_type == "count":
                result_series = rolling.count()
            elif agg_type == "std":
                result_series = rolling.std()
            else:
                result_series = rolling.mean()
    else:
        raise ValueError(f"Unknown window frame type: {frame.type}")
    
    # Reindex to original DataFrame index (restore original order)
    result_series = result_series.reindex(df_sorted_index)
    result_series = result_series.reindex(original_index)
    
    return result_series


def compute_window_columns(
    df: pd.DataFrame,
    window_specs: Dict[str, DistWindow]
) -> pd.DataFrame:
    """
    Compute all window columns for a DataFrame.
    
    Args:
        df: DataFrame (should be fully materialized, not streaming)
        window_specs: Dictionary mapping column names to window specifications
    
    Returns:
        DataFrame with window columns added
    """
    result_df = df.copy()
    
    for col_name, window_spec in window_specs.items():
        try:
            result_df[col_name] = eval_window_expression(result_df, window_spec)
        except Exception as e:
            raise ValueError(
                f"Failed to compute window column '{col_name}': {e}"
            ) from e
    
    return result_df

