"""Single-table quality evaluation using SD Metrics."""

from typing import Dict, Any, Optional
import pandas as pd
from sdmetrics.reports.single_table import QualityReport


def evaluate_table_quality(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Evaluate quality of synthetic table data using SD Metrics.
    
    IMPORTANT: Both DataFrames must have the same column names and structure.
    Column renaming and dropping should be done BEFORE calling this function.
    
    Args:
        real_df: Real data DataFrame (with matched columns only, renamed to match)
        synth_df: Synthetic data DataFrame (with matched columns only, renamed to match)
        metadata: Optional SDV metadata dict (auto-inferred if None)
        
    Returns:
        Dictionary with quality scores and detailed metrics:
        - overall_score: float - Overall SD Metrics score [0,1]
        - column_scores: Dict[str, Dict] - Per-column scores with metric names
        - pair_scores: Dict[Tuple[str, str], Dict] - Per-column-pair scores
        - report: QualityReport object
    """
    # Generate quality report
    # SD Metrics requires metadata - create simple metadata if not provided
    if metadata is None:
        # Create minimal metadata from DataFrame dtypes
        metadata = {
            "columns": {}
        }
        for col in real_df.columns:
            dtype = real_df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "boolean"
            else:
                col_type = "categorical"
            
            metadata["columns"][col] = {
                "sdtype": col_type
            }
    
    report = QualityReport()
    report.generate(
        real_data=real_df,
        synthetic_data=synth_df,
        metadata=metadata
    )
    
    # Extract overall score
    # Based on SD Metrics API: https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report
    overall_score = report.get_score()
    
    # Extract detailed metrics from report
    # SD Metrics API: get_details() returns a pandas DataFrame
    # Based on: https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report/multi-table-api
    # Properties: 'Column Shapes', 'Column Pair Trends'
    column_scores = {}
    pair_scores = {}
    
    try:
        # Get column shape details
        # Returns DataFrame with columns: Table, Column, Metric, Score
        column_details_df = report.get_details('Column Shapes')
        if hasattr(column_details_df, 'iterrows'):  # Check if it's a DataFrame
            for _, row in column_details_df.iterrows():
                col_name = row.get('Column', '')
                col_score = row.get('Score', 0.0)
                metric_name = row.get('Metric', 'unknown')
                if col_name:
                    column_scores[col_name] = {
                        "score": float(col_score) if col_score is not None else 0.0,
                        "metric_name": str(metric_name)
                    }
        elif isinstance(column_details_df, dict):
            # Fallback: handle as dict if API changes
            for col_name, col_info in column_details_df.items():
                if isinstance(col_info, dict):
                    col_score = col_info.get('score', 0.0)
                    metric_name = col_info.get('metric', 'unknown')
                    column_scores[col_name] = {
                        "score": float(col_score) if col_score is not None else 0.0,
                        "metric_name": str(metric_name)
                    }
    except Exception as e:
        # If API structure is different, try alternative approach
        pass
    
    try:
        # Get column pair details
        # Returns DataFrame with columns: Table, Column 1, Column 2, Metric, Score
        pair_details_df = report.get_details('Column Pair Trends')
        if hasattr(pair_details_df, 'iterrows'):  # Check if it's a DataFrame
            for _, row in pair_details_df.iterrows():
                col1 = row.get('Column 1', '') or row.get('Column1', '')
                col2 = row.get('Column 2', '') or row.get('Column2', '')
                pair_score = row.get('Score', 0.0)
                metric_name = row.get('Metric', 'unknown')
                if col1 and col2:
                    pair_scores[(col1, col2)] = {
                        "score": float(pair_score) if pair_score is not None else 0.0,
                        "metric_name": str(metric_name)
                    }
        elif isinstance(pair_details_df, dict):
            # Fallback: handle as dict if API changes
            for key, pair_info in pair_details_df.items():
                if isinstance(pair_info, dict):
                    # Key might be tuple or string like "col1, col2"
                    if isinstance(key, tuple) and len(key) == 2:
                        col1, col2 = key
                    elif isinstance(key, str) and ',' in key:
                        col1, col2 = key.split(',', 1)
                        col1, col2 = col1.strip(), col2.strip()
                    else:
                        continue
                    
                    pair_score = pair_info.get('score', 0.0)
                    metric_name = pair_info.get('metric', 'unknown')
                    pair_scores[(col1, col2)] = {
                        "score": float(pair_score) if pair_score is not None else 0.0,
                        "metric_name": str(metric_name)
                    }
    except Exception:
        # If API structure is different, skip pair scores
        pass
    
    return {
        "overall_score": overall_score,
        "column_scores": column_scores,
        "pair_scores": pair_scores,
        "report": report
    }

