"""Functional dependency utilities for schema evaluation."""

from typing import Dict, Any, List, Set
import numpy as np
import pandas as pd
from nl2data.ir.logical import LogicalIR, ColumnSpec, TableSpec
from nl2data.ir.constraint_ir import ConstraintSpec, FDConstraint, TableFDConstraint


def compute_fd_counts(ir: LogicalIR) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Compute FD participation counts for each column.
    
    For each column, counts:
    - How many FDs have this column in the LHS (left-hand side)
    - How many FDs have this column in the RHS (right-hand side)
    
    Args:
        ir: LogicalIR schema
        
    Returns:
        Dictionary: {table_name: {column_name: {"lhs": count, "rhs": count}}}
        
    Example:
        If column "id" is in LHS of 2 FDs and RHS of 0 FDs:
        result["table"]["id"] = {"lhs": 2, "rhs": 0}
    """
    fd_counts = {}
    
    # Initialize counts for all columns
    for table_name, table_spec in ir.tables.items():
        fd_counts[table_name] = {}
        for col in table_spec.columns:
            fd_counts[table_name][col.name] = {"lhs": 0, "rhs": 0}
    
    # Count FD participation (read from table.fds)
    for table_name, table_spec in ir.tables.items():
        if table_name in fd_counts:
            for fd in table_spec.fds:
                # Count LHS participation
                for col_name in fd.lhs:
                    if col_name in fd_counts[table_name]:
                        fd_counts[table_name][col_name]["lhs"] += 1
                
                # Count RHS participation
                for col_name in fd.rhs:
                    if col_name in fd_counts[table_name]:
                        fd_counts[table_name][col_name]["rhs"] += 1
    
    return fd_counts


def compute_fd_signature(ir: LogicalIR, table_name: str, feature_length: int = 10) -> np.ndarray:
    """
    Compute FD signature feature vector phi(t) for a table.
    
    This creates a fixed-length numeric vector representing the FD structure
    of a table. The vector includes features like:
    - Number of FDs
    - Average LHS size
    - Average RHS size
    - Number of columns involved in FDs
    - etc.
    
    Args:
        ir: LogicalIR schema
        table_name: Name of the table
        feature_length: Fixed length of the feature vector (default: 10)
        
    Returns:
        Fixed-length numpy array representing FD signature
    """
    # Read FDs from table.fds instead of constraints.fds
    if table_name not in ir.tables:
        return np.zeros(feature_length, dtype=np.float32)
    
    table_fds = ir.tables[table_name].fds
    
    if not table_fds:
        # No FDs: return zero vector
        return np.zeros(feature_length, dtype=np.float32)
    
    # Compute features
    lhs_sizes = [len(fd.lhs) for fd in table_fds]
    rhs_sizes = [len(fd.rhs) for fd in table_fds]
    
    # Collect all columns involved in FDs
    all_fd_cols = set()
    for fd in table_fds:
        all_fd_cols.update(fd.lhs)
        all_fd_cols.update(fd.rhs)
    
    # Build feature vector
    features = [
        len(table_fds),  # Total FD count
        np.mean(lhs_sizes) if lhs_sizes else 0.0,  # Average LHS size
        np.mean(rhs_sizes) if rhs_sizes else 0.0,  # Average RHS size
        len(all_fd_cols),  # Unique columns in FDs
        np.max(lhs_sizes) if lhs_sizes else 0.0,  # Max LHS size
        np.max(rhs_sizes) if rhs_sizes else 0.0,  # Max RHS size
        np.min(lhs_sizes) if lhs_sizes else 0.0,  # Min LHS size
        np.min(rhs_sizes) if rhs_sizes else 0.0,  # Min RHS size
        len([fd for fd in table_fds if len(fd.lhs) == 1]),  # Single-column LHS count
        len([fd for fd in table_fds if len(fd.rhs) == 1]),  # Single-column RHS count
    ]
    
    # Pad or truncate to fixed length
    if len(features) < feature_length:
        features.extend([0.0] * (feature_length - len(features)))
    elif len(features) > feature_length:
        features = features[:feature_length]
    
    return np.array(features, dtype=np.float32)


def compute_column_summaries(
    ir: LogicalIR,
    dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute numeric and categorical summaries for all columns from actual data.
    
    NOTE: This function computes statistics from DataFrames, not from schema metadata.
    For synthetic schemas, the schema may not have precomputed min/max, so we must
    compute from the generated data.
    
    Args:
        ir: LogicalIR schema (used for column type information)
        dfs: Dictionary of table_name -> DataFrame with actual data
        
    Returns:
        Dictionary: {table_name: {column_name: summary_dict}}
        
    Summary dict structure:
    - For numeric columns:
      - "is_numeric": True
      - "is_categorical": False
      - "min": float
      - "max": float
    - For categorical columns:
      - "is_numeric": False
      - "is_categorical": True
      - "dom_set": set of distinct values (top-K, K=100)
      - "p_c": dict mapping value -> probability
    - For datetime columns:
      - Treated as numeric (converted to timestamp)
    """
    summaries = {}
    
    for table_name, table_spec in ir.tables.items():
        if table_name not in dfs:
            # Skip if DataFrame not available
            continue
        
        df = dfs[table_name]
        summaries[table_name] = {}
        
        for col_spec in table_spec.columns:
            if col_spec.name not in df.columns:
                # Skip if column not in DataFrame
                continue
            
            series = df[col_spec.name].dropna()
            
            # If no data, skip
            if len(series) == 0:
                continue
            
            summary = {}
            
            # Determine if numeric or categorical based on SQL type
            numeric_types = {"INT", "FLOAT", "INT32", "INT64", "FLOAT32", "FLOAT64"}
            datetime_types = {"DATE", "DATETIME"}
            is_numeric_type = col_spec.sql_type in numeric_types
            is_datetime_type = col_spec.sql_type in datetime_types
            
            if is_numeric_type:
                # Numeric summary - compute from actual data
                summary["is_numeric"] = True
                summary["is_categorical"] = False
                summary["min"] = float(series.min())
                summary["max"] = float(series.max())
            
            elif is_datetime_type:
                # Datetime: treat as numeric for range comparison
                # Convert to numeric (timestamp) for min/max
                if pd.api.types.is_datetime64_any_dtype(series):
                    numeric_series = series.astype('int64')  # Convert to nanoseconds
                    summary["is_numeric"] = True  # Treat as numeric for range
                    summary["is_categorical"] = False
                    summary["min"] = float(numeric_series.min())
                    summary["max"] = float(numeric_series.max())
                else:
                    # Fallback: treat as categorical
                    summary["is_numeric"] = False
                    summary["is_categorical"] = True
                    value_counts = series.value_counts()
                    K = min(100, len(value_counts))
                    top_values = value_counts.head(K)
                    summary["dom_set"] = set(top_values.index.tolist())
                    total_count = top_values.sum()
                    summary["p_c"] = {
                        str(val): float(count / total_count)
                        for val, count in top_values.items()
                    }
            
            else:
                # Categorical summary (TEXT, BOOL, etc.) - compute from actual data
                summary["is_numeric"] = False
                summary["is_categorical"] = True
                
                # Get top-K distinct values (K=100)
                value_counts = series.value_counts()
                K = min(100, len(value_counts))
                top_values = value_counts.head(K)
                
                summary["dom_set"] = set(str(v) for v in top_values.index.tolist())
                
                # Normalized probabilities
                total_count = top_values.sum()
                summary["p_c"] = {
                    str(val): float(count / total_count)
                    for val, count in top_values.items()
                }
            
            summaries[table_name][col_spec.name] = summary
    
    return summaries

