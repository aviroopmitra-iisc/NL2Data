"""Structure score (S_structure,intra and S_structure,inter) computation."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from nl2data.ir.logical import LogicalIR
from nl2data.evaluation.metrics.table.correlations import correlation_metrics
from nl2data.evaluation.metrics.relational.integrity import (
    fk_coverage,
    fk_coverage_duckdb,
)
from nl2data.evaluation.metrics.relational.degrees import (
    degree_histogram,
    degree_distribution_divergence,
)
from nl2data.evaluation.metrics.schema.coverage import compute_coverage_factors
from nl2data.evaluation.matching.similarity import name_similarity
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy import for DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


def compute_intra_structure_score(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
    marginal_weight: float = 0.7,
) -> float:
    """
    Compute intra-table structure score (S_structure,intra).
    
    For each matched table pair, compares pairwise relationships (correlations)
    between real and synthetic.
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        marginal_weight: Weight for marginal vs pairwise (not used here, for consistency)
        
    Returns:
        S_structure,intra score [0,1]
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
        column_matches = column_mappings.get(real_table_name, {})
        
        if len(column_matches) < 2:
            # Need at least 2 columns for pairwise metrics
            continue
        
        # Compute pairwise correlation metrics
        pairwise_scores = []
        matched_cols = list(column_matches.items())
        
        for i, (real_col1, synth_col1) in enumerate(matched_cols):
            for real_col2, synth_col2 in matched_cols[i + 1:]:
                if (
                    real_col1 not in real_df.columns
                    or real_col2 not in real_df.columns
                    or synth_col1 not in synth_df.columns
                    or synth_col2 not in synth_df.columns
                ):
                    continue
                
                try:
                    real_series1 = real_df[real_col1].dropna()
                    real_series2 = real_df[real_col2].dropna()
                    synth_series1 = synth_df[synth_col1].dropna()
                    synth_series2 = synth_df[synth_col2].dropna()
                    
                    # Align lengths
                    min_len = min(
                        len(real_series1),
                        len(real_series2),
                        len(synth_series1),
                        len(synth_series2),
                    )
                    if min_len < 2:
                        continue
                    
                    real_series1 = real_series1[:min_len]
                    real_series2 = real_series2[:min_len]
                    synth_series1 = synth_series1[:min_len]
                    synth_series2 = synth_series2[:min_len]
                    
                    # Compute correlation metrics
                    metrics = correlation_metrics(
                        real_series1.values,
                        real_series2.values,
                        synth_series1.values,
                        synth_series2.values,
                    )
                    
                    # Convert deltas to scores (lower delta = better)
                    pearson_score = 1.0 - min(1.0, metrics.get("pearson_delta", 1.0))
                    spearman_score = 1.0 - min(1.0, metrics.get("spearman_delta", 1.0))
                    pair_score = (pearson_score + spearman_score) / 2.0
                    pairwise_scores.append(pair_score)
                except Exception as e:
                    logger.warning(
                        f"Error computing correlation for {real_table_name}.{real_col1}-{real_col2}: {e}"
                    )
                    continue
        
        # Per-table score: average of pairwise scores
        if pairwise_scores:
            table_score = np.mean(pairwise_scores)
            table_scores.append(table_score)
    
    if not table_scores:
        return 0.0
    
    # Aggregate across tables (before penalties)
    S_structure_intra_aligned = np.mean(table_scores)
    
    # Apply coverage penalties (same as schema score)
    coverage_factors = compute_coverage_factors(
        table_mapping, column_mappings, real_ir
    )
    table_coverage = coverage_factors["table_coverage"]
    column_coverage = coverage_factors["column_coverage"]
    
    matched_table_coverage = [
        column_coverage.get(real_table, 0.0) for real_table in table_mapping.keys()
    ]
    avg_column_coverage = np.mean(matched_table_coverage) if matched_table_coverage else 0.0
    
    C_schema = table_coverage * avg_column_coverage
    
    # Final score with coverage penalty
    S_structure_intra = S_structure_intra_aligned * C_schema
    
    return float(max(0.0, min(1.0, S_structure_intra)))


def compute_intra_structure_score_schema_only(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute schema-only intra-table structure score.
    
    Compares FD patterns and constraint structures without requiring data.
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        S_structure,intra score [0,1]
    """
    if not table_mapping:
        return 0.0
    
    table_scores = []
    
    # For each matched table pair
    for real_table_name, synth_table_name in table_mapping.items():
        real_table = real_ir.tables.get(real_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        
        if not real_table or not synth_table:
            continue
        
        column_matches = column_mappings.get(real_table_name, {})
        if len(column_matches) < 2:
            continue
        
        # Compare FD patterns
        real_fds = real_table.fds
        synth_fds = synth_table.fds
        
        # FD pattern similarity: compare FD counts and structures
        fd_scores = []
        
        # Compare number of FDs
        if len(real_fds) == 0 and len(synth_fds) == 0:
            fd_scores.append(1.0)
        elif len(real_fds) > 0 and len(synth_fds) > 0:
            # Both have FDs - compare structures
            # Simple heuristic: if both have similar number of FDs, score is higher
            fd_count_sim = 1.0 / (1.0 + abs(len(real_fds) - len(synth_fds)))
            fd_scores.append(fd_count_sim)
            
            # Compare average LHS/RHS sizes
            if real_fds and synth_fds:
                real_avg_lhs = np.mean([len(fd.lhs) for fd in real_fds])
                synth_avg_lhs = np.mean([len(fd.lhs) for fd in synth_fds])
                real_avg_rhs = np.mean([len(fd.rhs) for fd in real_fds])
                synth_avg_rhs = np.mean([len(fd.rhs) for fd in synth_fds])
                
                lhs_sim = 1.0 / (1.0 + abs(real_avg_lhs - synth_avg_lhs))
                rhs_sim = 1.0 / (1.0 + abs(real_avg_rhs - synth_avg_rhs))
                fd_scores.append((lhs_sim + rhs_sim) / 2.0)
        else:
            # One has FDs, other doesn't
            fd_scores.append(0.0)
        
        # Compare PK structure
        pk_sim = 1.0 if real_table.primary_key == synth_table.primary_key else 0.5
        
        # Compare FK structure (count)
        fk_count_sim = 1.0 / (1.0 + abs(len(real_table.foreign_keys) - len(synth_table.foreign_keys)))
        
        # Combine structure scores
        if fd_scores:
            structure_score = (
                0.4 * np.mean(fd_scores) +
                0.3 * pk_sim +
                0.3 * fk_count_sim
            )
            table_scores.append(structure_score)
    
    if not table_scores:
        return 0.0
    
    # Aggregate across tables
    S_structure_intra_aligned = np.mean(table_scores)
    
    # Apply coverage penalties
    coverage_factors = compute_coverage_factors(
        table_mapping, column_mappings, real_ir
    )
    table_coverage = coverage_factors["table_coverage"]
    column_coverage = coverage_factors["column_coverage"]
    
    matched_table_coverage = [
        column_coverage.get(real_table, 0.0) for real_table in table_mapping.keys()
    ]
    avg_column_coverage = np.mean(matched_table_coverage) if matched_table_coverage else 0.0
    
    C_schema = table_coverage * avg_column_coverage
    
    # Final score with coverage penalty
    S_structure_intra = S_structure_intra_aligned * C_schema
    
    return float(max(0.0, min(1.0, S_structure_intra)))


def compute_inter_structure_score(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    table_mapping: Dict[str, str],
    ri_weight: float = 0.4,
    cardinality_weight: float = 0.3,
    trend_weight: float = 0.3,
) -> float:
    """
    Compute inter-table structure score (S_structure,inter).
    
    For each real FK relationship, computes:
    - Referential integrity (r_RI)
    - Cardinality similarity (r_card)
    - Trend similarity (r_trend)
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        table_mapping: Dictionary mapping real_table -> synth_table
        ri_weight: Weight for referential integrity (α)
        cardinality_weight: Weight for cardinality (β)
        trend_weight: Weight for trends (γ)
        
    Returns:
        S_structure,inter score [0,1]
    """
    if not DUCKDB_AVAILABLE:
        logger.warning("DuckDB not available, skipping inter-table structure scoring")
        return 0.0
    
    relationship_scores = []
    
    # Create DuckDB connections for real and synthetic
    real_con = duckdb.connect()
    synth_con = duckdb.connect()
    
    # Register DataFrames
    for table_name, df in real_dfs.items():
        real_con.register(table_name, df)
    for table_name, df in synth_dfs.items():
        synth_table_name = table_mapping.get(table_name, table_name)
        if synth_table_name in synth_dfs:
            synth_con.register(synth_table_name, synth_dfs[synth_table_name])
    
    # For each real FK relationship
    for real_table_name, real_table in real_ir.tables.items():
        if real_table_name not in table_mapping:
            continue
        
        synth_table_name = table_mapping[real_table_name]
        if synth_table_name not in synth_ir.tables:
            continue
        
        synth_table = synth_ir.tables[synth_table_name]
        
        for fk in real_table.foreign_keys:
            # Check if corresponding FK exists in synthetic
            synth_fk = None
            for sfk in synth_table.foreign_keys:
                if (
                    sfk.column == fk.column
                    and sfk.ref_table in table_mapping.values()
                ):
                    synth_fk = sfk
                    break
            
            if not synth_fk:
                # FK doesn't exist in synthetic - contribute 0
                continue
            
            # Get referenced table names
            real_ref_table = fk.ref_table
            synth_ref_table = table_mapping.get(real_ref_table)
            if not synth_ref_table:
                continue
            
            # Get PK column
            real_ref_table_spec = real_ir.tables.get(real_ref_table)
            synth_ref_table_spec = synth_ir.tables.get(synth_ref_table)
            if not real_ref_table_spec or not synth_ref_table_spec:
                continue
            
            pk_col = (
                real_ref_table_spec.primary_key[0]
                if real_ref_table_spec.primary_key
                else fk.ref_column
            )
            
            try:
                # 1. Referential Integrity (r_RI)
                real_ri = fk_coverage_duckdb(
                    real_con, real_table_name, fk.column, real_ref_table, pk_col
                )
                synth_ri = fk_coverage_duckdb(
                    synth_con, synth_table_name, synth_fk.column, synth_ref_table, pk_col
                )
                r_RI = min(real_ri, synth_ri)  # Use minimum (both should be high)
                
                # 2. Cardinality (r_card)
                real_hist = degree_histogram(
                    real_con, real_table_name, fk.column, real_ref_table, pk_col
                )
                synth_hist = degree_histogram(
                    synth_con, synth_table_name, synth_fk.column, synth_ref_table, pk_col
                )
                
                if len(real_hist[0]) > 0 and len(synth_hist[0]) > 0:
                    w1_dist = degree_distribution_divergence(real_hist, synth_hist)
                    # Convert distance to similarity (normalize)
                    r_card = 1.0 / (1.0 + w1_dist)  # Decay with distance
                else:
                    r_card = 0.0
                
                # 3. Trend similarity (r_trend) - simplified for now
                # Join tables and compare column-pair relationships
                # For now, use a simple heuristic: if RI and cardinality are good, trends are likely good
                r_trend = (r_RI + r_card) / 2.0
                
                # Combine relationship scores
                total_weight = ri_weight + cardinality_weight + trend_weight
                if total_weight > 0:
                    r_rel = (
                        ri_weight * r_RI
                        + cardinality_weight * r_card
                        + trend_weight * r_trend
                    ) / total_weight
                    relationship_scores.append(r_rel)
            
            except Exception as e:
                logger.warning(
                    f"Error computing inter-table structure for {real_table_name}.{fk.column}: {e}"
                )
                continue
    
    # Close connections
    try:
        real_con.close()
        synth_con.close()
    except Exception:
        pass
    
    if not relationship_scores:
        return 0.0
    
    # Average over all relationships
    S_structure_inter = np.mean(relationship_scores)
    
    return float(max(0.0, min(1.0, S_structure_inter)))


def compute_inter_structure_score_schema_only(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute schema-only inter-table structure score using F1 score on FK constraints.
    
    Collects all FK constraints from both schemas, maps column names using column_mappings,
    and calculates F1 score (precision, recall, F1) based on common constraints.
    
    Args:
        real_ir: Real LogicalIR (gold truth)
        synth_ir: Synthetic LogicalIR (generated)
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        F1 score [0,1] for FK constraint matching
    """
    if not table_mapping:
        return 0.0
    
    # Step 1: Collect all FK constraints from gold schema
    gold_fk_set = set()
    for real_table_name, real_table in real_ir.tables.items():
        for fk in real_table.foreign_keys:
            # Create normalized FK constraint tuple: (table, column, ref_table, ref_column)
            gold_fk_set.add((real_table_name, fk.column, fk.ref_table, fk.ref_column))
    
    # Step 2: Collect all FK constraints from generated schema
    synth_fk_set = set()
    for synth_table_name, synth_table in synth_ir.tables.items():
        for fk in synth_table.foreign_keys:
            synth_fk_set.add((synth_table_name, fk.column, fk.ref_table, fk.ref_column))
    
    # Step 3: Map synth FK constraints to gold space using table and column mappings
    mapped_synth_fk_set = set()
    for synth_table_name, synth_col, synth_ref_table, synth_ref_col in synth_fk_set:
        # Find corresponding gold table
        gold_table_name = None
        for gold_t, synth_t in table_mapping.items():
            if synth_t == synth_table_name:
                gold_table_name = gold_t
                break
        
        if not gold_table_name:
            continue  # Skip if table not mapped
        
        # Find corresponding gold column
        col_mapping = column_mappings.get(gold_table_name, {})
        # Reverse lookup: find gold_col such that col_mapping[gold_col] == synth_col
        gold_col = None
        for g_col, s_col in col_mapping.items():
            if s_col == synth_col:
                gold_col = g_col
                break
        if not gold_col:
            # If not found in mapping, use synth_col as-is (fallback)
            gold_col = synth_col
        
        # Find corresponding gold ref_table
        gold_ref_table = None
        for gold_t, synth_t in table_mapping.items():
            if synth_t == synth_ref_table:
                gold_ref_table = gold_t
                break
        if not gold_ref_table:
            continue  # Skip if ref_table not mapped
        
        # Find corresponding gold ref_column
        ref_col_mapping = column_mappings.get(gold_ref_table, {})
        # Reverse lookup: find gold_ref_col such that ref_col_mapping[gold_ref_col] == synth_ref_col
        gold_ref_col = None
        for g_col, s_col in ref_col_mapping.items():
            if s_col == synth_ref_col:
                gold_ref_col = g_col
                break
        if not gold_ref_col:
            # If not found in mapping, use synth_ref_col as-is (fallback)
            gold_ref_col = synth_ref_col
        
        mapped_synth_fk_set.add((gold_table_name, gold_col, gold_ref_table, gold_ref_col))
    
    # Step 4: Calculate F1 score
    intersection = gold_fk_set & mapped_synth_fk_set
    precision = len(intersection) / len(mapped_synth_fk_set) if mapped_synth_fk_set else 0.0
    recall = len(intersection) / len(gold_fk_set) if gold_fk_set else 0.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    logger.info(
        f"FK Constraint F1: precision={precision:.4f}, recall={recall:.4f}, "
        f"f1={f1_score:.4f} (gold={len(gold_fk_set)}, synth={len(synth_fk_set)}, "
        f"mapped_synth={len(mapped_synth_fk_set)}, intersection={len(intersection)})"
    )
    
    return float(max(0.0, min(1.0, f1_score)))


# Alias for backward compatibility and clarity
compute_foreign_key_score_schema_only = compute_inter_structure_score_schema_only


def compute_primary_key_score_schema_only(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute primary key constraint score using F1 score.
    
    For each matched table, checks if the counterpart table's PK matches
    the mapped version of the gold table's PK.
    
    Args:
        real_ir: Real LogicalIR (gold truth)
        synth_ir: Synthetic LogicalIR (generated)
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        F1 score [0,1] for PK constraint matching
    """
    if not table_mapping:
        return 0.0
    
    matched_pks = []
    total_gold_tables = 0
    
    # For each matched table pair
    for gold_table_name, synth_table_name in table_mapping.items():
        gold_table = real_ir.tables.get(gold_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        
        if not gold_table or not synth_table:
            continue
        
        total_gold_tables += 1
        
        # Get gold PK
        gold_pk = set(gold_table.primary_key)
        if not gold_pk:
            continue  # Skip if no PK
        
        # Get synth PK
        synth_pk = set(synth_table.primary_key)
        if not synth_pk:
            matched_pks.append(False)
            continue
        
        # Map synth PK columns to gold space
        col_mapping = column_mappings.get(gold_table_name, {})
        mapped_synth_pk = set()
        for synth_col in synth_pk:
            # Reverse lookup: find gold column that maps to this synth column
            gold_col = None
            for g_col, s_col in col_mapping.items():
                if s_col == synth_col:
                    gold_col = g_col
                    break
            if gold_col:
                mapped_synth_pk.add(gold_col)
            else:
                # If not found in mapping, try direct match
                if synth_col in gold_pk:
                    mapped_synth_pk.add(synth_col)
        
        # Check if mapped synth PK matches gold PK
        matched_pks.append(gold_pk == mapped_synth_pk)
    
    if total_gold_tables == 0:
        return 0.0
    
    # Calculate F1 score
    true_positives = sum(matched_pks)
    precision = true_positives / total_gold_tables if total_gold_tables > 0 else 0.0
    recall = true_positives / total_gold_tables if total_gold_tables > 0 else 0.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    logger.info(
        f"PK Constraint F1: precision={precision:.4f}, recall={recall:.4f}, "
        f"f1={f1_score:.4f} (matched={true_positives}/{total_gold_tables})"
    )
    
    return float(max(0.0, min(1.0, f1_score)))


def compute_functional_dependency_score_schema_only(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    table_mapping: Dict[str, str],
    column_mappings: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute functional dependency constraint score using F1 metric.
    
    Similar to FK evaluation: collects all FD constraints from real and generated schemas,
    maps generated FDs to the real schema space using table and column mappings,
    and calculates F1 score based on the intersection.
    
    Args:
        real_ir: Real LogicalIR (gold truth)
        synth_ir: Synthetic LogicalIR (generated)
        table_mapping: Dictionary mapping real_table -> synth_table
        column_mappings: Dictionary mapping table_name -> {real_col -> synth_col}
        
    Returns:
        F1 score [0,1] for FD constraint matching
    """
    if not table_mapping:
        return 0.0

    # Step 1: Extract all FD constraints from real IR
    gold_fd_set = set()
    for real_table_name, real_table in real_ir.tables.items():
        for fd in real_table.fds:
            # FD is represented as (table, lhs_cols, rhs_cols)
            # Convert to tuple for set operations
            lhs_tuple = tuple(sorted(fd.lhs))
            rhs_tuple = tuple(sorted(fd.rhs))
            gold_fd_set.add((real_table_name, lhs_tuple, rhs_tuple))

    # Step 2: Extract all FD constraints from synthetic IR
    synth_fd_set = set()
    for synth_table_name, synth_table in synth_ir.tables.items():
        for fd in synth_table.fds:
            lhs_tuple = tuple(sorted(fd.lhs))
            rhs_tuple = tuple(sorted(fd.rhs))
            synth_fd_set.add((synth_table_name, lhs_tuple, rhs_tuple))

    # Step 3: Map synth FD constraints to gold space using table and column mappings
    mapped_synth_fd_set = set()
    for synth_table_name, synth_lhs, synth_rhs in synth_fd_set:
        # Find corresponding gold table
        gold_table_name = None
        for gold_t, synth_t in table_mapping.items():
            if synth_t == synth_table_name:
                gold_table_name = gold_t
                break
        
        if not gold_table_name:
            continue  # Skip if table not mapped
        
        # Find corresponding gold columns for LHS
        col_mapping = column_mappings.get(gold_table_name, {})
        mapped_lhs = []
        for synth_col in synth_lhs:
            # Reverse lookup: find gold column that maps to this synth column
            gold_col = None
            for g_col, s_col in col_mapping.items():
                if s_col == synth_col:
                    gold_col = g_col
                    break
            if gold_col:
                mapped_lhs.append(gold_col)
            else:
                # If not found in mapping, try direct match
                gold_table = real_ir.tables.get(gold_table_name)
                if gold_table and synth_col in [c.name for c in gold_table.columns]:
                    mapped_lhs.append(synth_col)
        
        # Find corresponding gold columns for RHS
        mapped_rhs = []
        for synth_col in synth_rhs:
            gold_col = None
            for g_col, s_col in col_mapping.items():
                if s_col == synth_col:
                    gold_col = g_col
                    break
            if gold_col:
                mapped_rhs.append(gold_col)
            else:
                gold_table = real_ir.tables.get(gold_table_name)
                if gold_table and synth_col in [c.name for c in gold_table.columns]:
                    mapped_rhs.append(synth_col)
        
        if mapped_lhs and mapped_rhs:
            mapped_synth_fd_set.add((gold_table_name, tuple(sorted(mapped_lhs)), tuple(sorted(mapped_rhs))))
    
    # Step 4: Calculate F1 score
    intersection = gold_fd_set & mapped_synth_fd_set
    precision = len(intersection) / len(mapped_synth_fd_set) if mapped_synth_fd_set else 0.0
    recall = len(intersection) / len(gold_fd_set) if gold_fd_set else 0.0
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    logger.info(
        f"FD Constraint F1: precision={precision:.4f}, recall={recall:.4f}, "
        f"f1={f1_score:.4f} (gold={len(gold_fd_set)}, synth={len(synth_fd_set)}, "
        f"mapped_synth={len(mapped_synth_fd_set)}, intersection={len(intersection)})"
    )
    
    return float(max(0.0, min(1.0, f1_score)))
