"""Multi-table quality evaluation using SD Metrics."""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport

from nl2data.ir.logical import LogicalIR
from nl2data.evaluation.models.multi_table import SchemaMatchResult


def evaluate_multi_table_quality(
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],  # real_table -> synth_table
    relationship_mapping: List[Tuple[str, str, str, str]]  # (parent_table, child_table, parent_col, child_col)
) -> Dict[str, Any]:
    """
    Evaluate quality of multi-table synthetic data.
    
    IMPORTANT: All DataFrames must have matched columns renamed to the same names.
    Unmatched columns should be dropped before calling this function.
    
    Args:
        real_dfs: Real DataFrames by table name (with matched columns only, renamed)
        synth_dfs: Synthetic DataFrames by table name (with matched columns only, renamed)
        table_mapping: Mapping from real to synthetic table names
        relationship_mapping: List of (parent_table, child_table, parent_col, child_col) tuples
        
    Returns:
        Dictionary with multi-table quality scores:
        - overall_score: float - Overall SD Metrics score [0,1]
        - relationship_scores: Dict[Tuple[str, str], Dict] - Per-relationship scores
        - report: MultiTableQualityReport object
    """
    # Prepare real and synthetic table dicts with matched names
    # Use real table names as keys for both (since columns are already renamed)
    real_tables = {real_name: real_dfs[real_name] 
                   for real_name in table_mapping.keys()}
    synth_tables = {real_name: synth_dfs[synth_name] 
                    for real_name, synth_name in table_mapping.items()}
    
    # Generate multi-table quality report
    report = MultiTableQualityReport()
    report.generate(
        real_data=real_tables,
        synthetic_data=synth_tables,
        relationships=relationship_mapping
    )
    
    overall_score = report.get_score()
    
    # Extract detailed metrics from report
    # Based on SD Metrics Multi Table API: https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report/multi-table-api
    # get_details(property_name, table_name) returns a pandas DataFrame
    relationship_scores = {}
    
    try:
        # Get Intertable Trends details
        # Returns DataFrame with columns: Table, Parent Table, Child Table, Metric, Score (or similar)
        intertable_details_df = report.get_details('Intertable Trends')
        if hasattr(intertable_details_df, 'iterrows'):  # Check if it's a DataFrame
            for _, row in intertable_details_df.iterrows():
                parent = row.get('Parent Table', '') or row.get('ParentTable', '')
                child = row.get('Child Table', '') or row.get('ChildTable', '')
                rel_score = row.get('Score', 0.0)
                metric_name = row.get('Metric', 'unknown')
                if parent and child:
                    relationship_scores[(parent, child)] = {
                        "score": float(rel_score) if rel_score is not None else 0.0,
                        "metric_name": str(metric_name)
                    }
    except Exception:
        # If API structure is different, skip relationship scores
        pass
    
    return {
        "overall_score": overall_score,
        "relationship_scores": relationship_scores,
        "report": report
    }


def extract_relationship_mappings(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    schema_match_result: SchemaMatchResult
) -> List[Tuple[str, str, str, str]]:
    """
    Extract relationship mappings for multi-table SD Metrics.
    
    Maps foreign key relationships from real schema to matched tables/columns
    in synthetic schema. Returns relationships using real table/column names
    (since columns are renamed to match real names before SD Metrics evaluation).
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        schema_match_result: Result from enhanced schema matching
        
    Returns:
        List of (parent_table, child_table, parent_col, child_col) tuples
        All names are from real schema (after column renaming)
    """
    relationships = []
    
    # Build table name mapping (real -> synth)
    table_mapping = {
        m.real_table: m.synth_table 
        for m in schema_match_result.table_matches
    }
    
    # Build column mappings per table (real_col -> synth_col)
    column_mappings = {}
    for real_table, matches in schema_match_result.column_matches.items():
        column_mappings[real_table] = {
            cm.real_column: cm.synth_column 
            for cm in matches
        }
    
    # Extract FK relationships from real schema
    for real_table_name, real_table in real_ir.tables.items():
        if real_table_name not in table_mapping:
            continue  # Skip unmatched tables
        
        for fk in real_table.foreign_keys:
            # Parse FK reference (format: "table.column" or just "column")
            if '.' in fk.references:
                parent_table, ref_col = fk.references.split('.', 1)
            else:
                # If no table specified, assume same table
                parent_table = real_table_name
                ref_col = fk.references
            
            # Check if parent table is matched
            if parent_table not in table_mapping:
                continue
            
            # Check if FK column and referenced column are both matched
            if (real_table_name in column_mappings and 
                fk.column in column_mappings[real_table_name] and
                parent_table in column_mappings and
                ref_col in column_mappings[parent_table]):
                # Add relationship using real column names
                # (columns will be renamed to match real names before SD Metrics)
                relationships.append((
                    parent_table,  # Parent table (real name)
                    real_table_name,  # Child table (real name)
                    ref_col,  # Parent column (real name)
                    fk.column  # Child column (real name)
                ))
    
    return relationships


def compute_quality_scores(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    schema_match_result: SchemaMatchResult
) -> Dict[str, Any]:
    """
    Compute SD Metrics quality scores for matched tables/columns.
    
    CRITICAL: This function prepares DataFrames for SD Metrics by:
    1. Selecting only matched columns
    2. Renaming synthetic columns to match real column names
    3. Dropping all unmatched columns
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        schema_match_result: Result from enhanced schema matching
        
    Returns:
        Dictionary with quality scores per table and overall:
        - overall_quality: float - Overall quality score (simple average)
        - table_quality: Dict[str, Dict] - Per-table quality scores
    """
    from .table_quality import evaluate_table_quality
    
    quality_results = {}
    
    # Process each matched table pair
    for table_match in schema_match_result.table_matches:
        real_table = table_match.real_table
        synth_table = table_match.synth_table
        
        # Get matched columns for this table pair
        column_matches = schema_match_result.column_matches.get(real_table, [])
        
        if not column_matches:
            continue
        
        # Extract matched column names
        real_cols = [cm.real_column for cm in column_matches]
        synth_cols = [cm.synth_column for cm in column_matches]
        
        # Get original DataFrames
        real_df = real_dfs.get(real_table, pd.DataFrame())
        synth_df = synth_dfs.get(synth_table, pd.DataFrame())
        
        if real_df.empty or synth_df.empty:
            continue
        
        # STEP 1: Select only matched columns from real DataFrame
        real_df_subset = real_df[real_cols].copy()
        
        # STEP 2: Select only matched columns from synthetic DataFrame
        synth_df_subset = synth_df[synth_cols].copy()
        
        # STEP 3: Rename synthetic columns to match real column names
        # This ensures SD Metrics can compare columns with the same names
        synth_df_aligned = synth_df_subset.copy()
        synth_df_aligned.columns = real_cols
        
        # STEP 4: Drop any unmatched columns (already done by column selection)
        # Both DataFrames now have the same column names and structure
        
        # Compute quality scores
        table_quality = evaluate_table_quality(
            real_df=real_df_subset,
            synth_df=synth_df_aligned
        )
        
        quality_results[real_table] = {
            "synth_table": synth_table,
            "overall_score": table_quality["overall_score"],
            "column_scores": table_quality["column_scores"],
            "pair_scores": table_quality["pair_scores"]
        }
    
    # Compute overall quality score (simple average across all tables)
    if quality_results:
        overall_quality = sum(
            result["overall_score"] 
            for result in quality_results.values()
        ) / len(quality_results)
    else:
        overall_quality = 0.0
    
    return {
        "overall_quality": overall_quality,
        "table_quality": quality_results
    }

