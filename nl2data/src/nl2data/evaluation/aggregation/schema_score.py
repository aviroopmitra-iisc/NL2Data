"""Schema score (S_schema) computation."""

from typing import Dict
import numpy as np
import pandas as pd
from nl2data.ir.logical import LogicalIR
from nl2data.evaluation.metrics.table.marginals import (
    numeric_marginals,
    categorical_marginals,
)
from nl2data.evaluation.metrics.schema.coverage import compute_coverage_factors
from nl2data.evaluation.utils.normalization import normalize_score
from nl2data.evaluation.matching.similarity import (
    name_similarity,
    type_compatibility,
    get_column_role,
    role_sim,
    fd_sim,
)
from nl2data.evaluation.utils.fd_utils import compute_fd_counts
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def compute_schema_score(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute schema score (S_schema) with coverage penalties.
    
    Process:
    1. For each matched table pair, build aligned DataFrames
    2. Compute column-shape scores per column
    3. Aggregate per-table scores
    4. Apply coverage penalties
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        S_schema score [0,1]
    """
    if not table_mapping:
        return 0.0
    
    table_scores = []
    
    # For each matched table pair
    for real_table_name, synth_table_name in table_mapping.items():
        if real_table_name not in real_dfs or synth_table_name not in synth_dfs:
            continue
        
        real_df = real_dfs[real_table_name]
        synth_df = synth_dfs[synth_table_name]
        real_table = real_ir.tables.get(real_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        
        if not real_table or not synth_table:
            continue
        
        column_matches = column_mappings.get(real_table_name, {})
        if not column_matches:
            continue
        
        # Compute per-column scores
        column_scores = []
        
        for real_col_name, synth_col_name in column_matches.items():
            if real_col_name not in real_df.columns or synth_col_name not in synth_df.columns:
                continue
            
            real_col = next((c for c in real_table.columns if c.name == real_col_name), None)
            if not real_col:
                continue
            
            real_series = real_df[real_col_name].dropna()
            synth_series = synth_df[synth_col_name].dropna()
            
            if len(real_series) == 0 or len(synth_series) == 0:
                continue
            
            # Compute marginal metrics
            is_numeric = real_col.sql_type in {
                "INT", "INT32", "INT64", "FLOAT", "FLOAT32", "FLOAT64", "NUMERIC"
            }
            
            try:
                if is_numeric:
                    metrics = numeric_marginals(real_series.values, synth_series.values)
                    # Convert to score: higher p-value = better, lower distance = better
                    ks_score = metrics.get("ks_pvalue", 0.0)
                    w1_score = 1.0 - min(1.0, metrics.get("wasserstein_distance", 0.0) / 10.0)
                    col_score = (ks_score + w1_score) / 2.0
                else:
                    metrics = categorical_marginals(real_series.values, synth_series.values)
                    # Higher p-value = better
                    col_score = metrics.get("chi2_pvalue", 0.0)
                
                column_scores.append(col_score)
            except Exception as e:
                logger.warning(
                    f"Error computing marginal for {real_table_name}.{real_col_name}: {e}"
                )
                continue
        
        # Per-table score: average of column scores
        if column_scores:
            table_score = np.mean(column_scores)
            table_scores.append(table_score)
    
    if not table_scores:
        return 0.0
    
    # Aggregate across tables (before penalties)
    S_schema_aligned = np.mean(table_scores)
    
    # Compute coverage factors
    coverage_factors = compute_coverage_factors(
        table_mapping, column_mappings, real_ir
    )
    
    table_coverage = coverage_factors["table_coverage"]
    column_coverage = coverage_factors["column_coverage"]
    
    # Average column coverage across matched tables
    matched_table_coverage = [
        column_coverage.get(real_table, 0.0)
        for real_table in table_mapping.keys()
    ]
    avg_column_coverage = np.mean(matched_table_coverage) if matched_table_coverage else 0.0
    
    # Coverage factor
    C_schema = table_coverage * avg_column_coverage
    
    # Final score with coverage penalty
    S_schema = S_schema_aligned * C_schema
    
    return float(max(0.0, min(1.0, S_schema)))


def compute_schema_score_schema_only(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute schema score (S_schema) using binary column matching.
    
    Since column matching is now binary (0 or 1), the schema score is simply
    the average column F1 score across all matched tables, weighted by coverage.
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        Schema score [0,1]
    """
    if not table_mapping:
        return 0.0
    
    # Calculate column F1 for each matched table
    from nl2data.evaluation.matching.column_matcher import _find_best_column_mapping_f1
    
    table_f1_scores = []
    
    for real_table_name, synth_table_name in table_mapping.items():
        real_table = real_ir.tables.get(real_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        
        if not real_table or not synth_table:
            continue
        
        # Get the column F1 score for this table pair
        _, column_f1 = _find_best_column_mapping_f1(real_table, synth_table)
        table_f1_scores.append(column_f1)
    
    if not table_f1_scores:
        return 0.0
    
    # Aggregate across tables (before penalties)
    S_schema_aligned = np.mean(table_f1_scores)
    
    # Compute coverage factors
    coverage_factors = compute_coverage_factors(
        table_mapping, column_mappings, real_ir
    )
    
    table_coverage = coverage_factors["table_coverage"]
    column_coverage = coverage_factors["column_coverage"]
    
    # Average column coverage across matched tables
    matched_table_coverage = [
        column_coverage.get(real_table, 0.0)
        for real_table in table_mapping.keys()
    ]
    avg_column_coverage = np.mean(matched_table_coverage) if matched_table_coverage else 0.0
    
    # Coverage factor
    C_schema = table_coverage * avg_column_coverage
    
    # Final score with coverage penalty
    S_schema = S_schema_aligned * C_schema
    
    return float(max(0.0, min(1.0, S_schema)))
