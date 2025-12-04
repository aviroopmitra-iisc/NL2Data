"""Column-level schema matching."""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from nl2data.ir.logical import TableSpec, ColumnSpec, LogicalIR
from nl2data.config.logging import get_logger
from .similarity import (
    name_similarity, 
    type_compatibility, 
    distribution_similarity,
    get_column_role,
    role_sim,
    range_sim,
)
from nl2data.evaluation.utils.fd_utils import compute_fd_counts

logger = get_logger(__name__)

# Lazy import for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None


def compute_column_similarity_schema_only(
    real_col: ColumnSpec,
    synth_col: ColumnSpec,
    real_table: TableSpec,
    synth_table: TableSpec,
    real_fd_counts: Optional[Dict[str, Dict[str, int]]] = None,
    synth_fd_counts: Optional[Dict[str, Dict[str, int]]] = None,
    name_weight: float = 0.6,
    range_weight: float = 0.4,
) -> float:
    """
    Compute schema-only similarity between two columns (no data required).
    
    Uses:
    - Name similarity (weight: 0.6)
    - Range similarity from schema metadata (weight: 0.4)
    
    Note: Role (PK/FK) and FD participation are NOT included in column alignment.
    They are evaluated separately.
    
    Type compatibility is a HARD RULE: if types are incompatible, returns 0.0 immediately.
    This is checked before calling this function in match_columns().
    
    Args:
        real_col: Real column specification
        synth_col: Synthetic column specification
        real_table: Real table specification (unused, kept for compatibility)
        synth_table: Synthetic table specification (unused, kept for compatibility)
        real_fd_counts: Unused, kept for compatibility
        synth_fd_counts: Unused, kept for compatibility
        name_weight: Weight for name similarity (default: 0.6)
        range_weight: Weight for range similarity (default: 0.4)
        
    Returns:
        Similarity score [0,1]
    """
    # Hard rule: Type incompatibility is checked in match_columns() before calling this function
    # If we reach here, types are compatible (or at least not hard incompatible)
    
    # Name similarity (uses RSchema-style compatibility: if compatible, returns 1.0 automatically)
    name_sim = name_similarity(real_col.name, synth_col.name, use_rschema_compatibility=True)
    
    # Range similarity from schema metadata (simplified - use nullable/unique as proxy)
    range_sim_val = 1.0
    if real_col.nullable == synth_col.nullable:
        range_sim_val = 1.0
    elif real_col.nullable and not synth_col.nullable:
        range_sim_val = 0.8  # Real allows nulls, synth doesn't (compatible)
    elif not real_col.nullable and synth_col.nullable:
        range_sim_val = 0.8  # Real doesn't allow nulls, synth does (less compatible)
    else:
        range_sim_val = 0.6
    
    # Weighted average (type compatibility is a hard rule, not weighted)
    total_weight = name_weight + range_weight
    if total_weight == 0:
        return 0.0
    
    similarity = (
        name_weight * name_sim
        + range_weight * range_sim_val
    ) / total_weight
    
    # Log detailed breakdown for debugging (only for first few matches to avoid spam)
    if not hasattr(compute_column_similarity_schema_only, '_log_count'):
        compute_column_similarity_schema_only._log_count = 0
    
    if compute_column_similarity_schema_only._log_count < 5:
        logger.info(
            f"Column similarity breakdown: {real_col.name} vs {synth_col.name} = {similarity:.4f}\n"
            f"  Components: name={name_sim:.4f} (×{name_weight}), "
            f"range={range_sim_val:.4f} (×{range_weight}, nullable={real_col.nullable}/{synth_col.nullable})\n"
            f"  Type compatibility: {real_col.sql_type} vs {synth_col.sql_type} (hard rule - checked before)\n"
            f"  Calculation: {name_weight}×{name_sim:.4f} + {range_weight}×{range_sim_val:.4f} = {similarity:.4f}"
        )
        compute_column_similarity_schema_only._log_count += 1
    
    return float(max(0.0, min(1.0, similarity)))


def compute_column_similarity(
    real_col: ColumnSpec,
    synth_col: ColumnSpec,
    real_series: pd.Series,
    synth_series: pd.Series,
    name_weight: float = 0.1,  # Reduced from 0.4
    type_weight: float = 0.3,  # Increased from 0.2 to compensate
    distribution_weight: float = 0.6,  # Increased from 0.4 to compensate
) -> float:
    """
    Compute similarity score between two columns.
    
    Args:
        real_col: Real column specification
        synth_col: Synthetic column specification
        real_series: Real column data
        synth_series: Synthetic column data
        name_weight: Weight for name similarity
        type_weight: Weight for type compatibility
        distribution_weight: Weight for distribution similarity
        
    Returns:
        Similarity score [0,1]
    """
    # Name similarity (uses RSchema-style compatibility: if compatible, returns 1.0 automatically)
    name_sim = name_similarity(real_col.name, synth_col.name, use_rschema_compatibility=True)
    
    # Type compatibility
    type_sim = type_compatibility(real_col.sql_type, synth_col.sql_type)
    
    # Distribution similarity
    is_numeric = real_col.sql_type in {
        "INT",
        "INT32",
        "INT64",
        "FLOAT",
        "FLOAT32",
        "FLOAT64",
        "NUMERIC",
    }
    dist_sim = distribution_similarity(real_series, synth_series, is_numeric=is_numeric)
    
    # Weighted average
    total_weight = name_weight + type_weight + distribution_weight
    if total_weight == 0:
        return 0.0
    
    similarity = (
        name_weight * name_sim
        + type_weight * type_sim
        + distribution_weight * dist_sim
    ) / total_weight
    
    return float(max(0.0, min(1.0, similarity)))


def _is_column_matchable(
    real_col: ColumnSpec,
    synth_col: ColumnSpec,
    debug: bool = False,
) -> bool:
    """
    Check if two columns can be matched (binary: True or False).
    
    A column pair is matchable if:
    1. Types are compatible (not hard incompatible)
    2. Names are compatible via WordNet/semantic/string matching
    
    Args:
        real_col: Real column specification
        synth_col: Synthetic column specification
        debug: Whether to log debug information
        
    Returns:
        True if matchable, False otherwise
    """
    from .similarity import hard_incompatible_datatype, type_compatibility, name_compatible
    
    # Hard rule: Type incompatibility
    if hard_incompatible_datatype(real_col.sql_type, synth_col.sql_type):
        if debug:
            logger.info(f"  {real_col.name} vs {synth_col.name}: hard incompatible datatype ({real_col.sql_type} vs {synth_col.sql_type})")
        return False
    
    # Check type compatibility
    type_sim = type_compatibility(real_col.sql_type, synth_col.sql_type)
    if type_sim == 0.0:
        if debug:
            logger.info(f"  {real_col.name} vs {synth_col.name}: type compatibility = 0.0 ({real_col.sql_type} vs {synth_col.sql_type})")
        return False
    
    # Check name compatibility (WordNet/semantic/string matching)
    name_comp = name_compatible(real_col.name, synth_col.name, semantic_threshold=0.6, lcs_threshold=0.75)
    if debug:
        if name_comp:
            logger.info(f"  {real_col.name} vs {synth_col.name}: MATCH (types: {real_col.sql_type} vs {synth_col.sql_type})")
        else:
            logger.info(f"  {real_col.name} vs {synth_col.name}: name not compatible (types OK: {real_col.sql_type} vs {synth_col.sql_type})")
    if name_comp:
        return True
    
    return False


def _find_best_column_mapping_f1(
    real_table: TableSpec,
    synth_table: TableSpec,
) -> Tuple[Dict[str, str], float]:
    """
    Find the best column mapping between two tables using F1 score.
    
    Tries all possible valid mappings and selects the one with highest F1.
    
    Args:
        real_table: Real table specification
        synth_table: Synthetic table specification
        
    Returns:
        Tuple of (best_mapping_dict, best_f1_score) where:
        - best_mapping_dict: Dictionary mapping real_column -> synth_column
        - best_f1_score: F1 score of the best mapping
    """
    real_cols = {c.name: c for c in real_table.columns}
    synth_cols = {c.name: c for c in synth_table.columns}
    
    if not real_cols or not synth_cols:
        return {}, 0.0
    
    # Enable debug logging for specific table pairs
    debug = (real_table.name == "dim_user" and synth_table.name == "dim_user")
    
    if debug:
        logger.info(f"  DEBUG: Finding best column mapping for {real_table.name} vs {synth_table.name}")
        logger.info(f"  DEBUG: Expected columns: {list(real_cols.keys())}")
        logger.info(f"  DEBUG: Generated columns: {list(synth_cols.keys())}")
    
    # Build binary compatibility matrix
    real_col_list = list(real_cols.items())
    synth_col_list = list(synth_cols.items())
    
    # compatibility_matrix[i][j] = True if real_col[i] can match synth_col[j]
    compatibility_matrix = []
    for real_name, real_col in real_col_list:
        row = []
        for synth_name, synth_col in synth_col_list:
            row.append(_is_column_matchable(real_col, synth_col, debug=debug))
        compatibility_matrix.append(row)
    
    if debug:
        logger.info(f"  DEBUG: Compatibility matrix:")
        for i, (real_name, _) in enumerate(real_col_list):
            matches = [synth_col_list[j][0] for j, compatible in enumerate(compatibility_matrix[i]) if compatible]
            logger.info(f"  DEBUG:   {real_name} can match: {matches if matches else 'NONE'}")
    
    # Generate all possible valid mappings (one-to-one)
    # A valid mapping: each real column maps to at most one synth column, each synth column maps to at most one real column
    best_mapping = {}
    best_f1 = 0.0
    
    # For small problems, try all combinations
    # For larger problems, use greedy approach
    if len(real_col_list) <= 10 and len(synth_col_list) <= 10:
        # Try all possible one-to-one mappings
        # We need to try all combinations of which real columns to match with synth columns
        from itertools import combinations, permutations
        
        if debug:
            logger.info(f"  DEBUG: Trying all combinations (small problem: {len(real_col_list)}x{len(synth_col_list)})")
        
        # Try all ways to assign synth columns to any subset of real columns (one-to-one)
        # Try all possible sizes of mappings (from 1 to min(len(real), len(synth)))
        max_size = min(len(real_col_list), len(synth_col_list))
        for num_to_match in range(1, max_size + 1):
            # Try all combinations of which real columns to use
            for real_indices in combinations(range(len(real_col_list)), num_to_match):
                # For each combination of real columns, try all permutations of synth columns
                for perm in permutations(range(len(synth_col_list)), num_to_match):
                    mapping = {}
                    valid = True
                    for idx, (real_idx, synth_idx) in enumerate(zip(real_indices, perm)):
                        if compatibility_matrix[real_idx][synth_idx]:
                            real_name = real_col_list[real_idx][0]
                            synth_name = synth_col_list[synth_idx][0]
                            mapping[real_name] = synth_name
                        else:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate F1 for this mapping
                    matched_count = len(mapping)
                    precision = matched_count / len(synth_col_list) if synth_col_list else 0.0
                    recall = matched_count / len(real_col_list) if real_col_list else 0.0
                    
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)
                    
                    if debug and f1 > 0:
                        logger.info(f"  DEBUG: Found mapping with F1={f1:.4f}: {mapping} (precision={precision:.4f}, recall={recall:.4f}, size={num_to_match})")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_mapping = mapping
    else:
        # For larger problems, use greedy approach
        # Sort all compatible pairs and greedily assign
        compatible_pairs = []
        for i, (real_name, real_col) in enumerate(real_col_list):
            for j, (synth_name, synth_col) in enumerate(synth_col_list):
                if compatibility_matrix[i][j]:
                    compatible_pairs.append((i, j, real_name, synth_name))
        
        # Try different greedy strategies and pick best F1
        # Strategy: prioritize real columns with fewer options
        from collections import defaultdict
        real_options = defaultdict(list)
        for i, j, real_name, synth_name in compatible_pairs:
            real_options[i].append((j, real_name, synth_name))
        
        # Sort real columns by number of options (fewer options first)
        sorted_real_indices = sorted(real_options.keys(), key=lambda x: len(real_options[x]))
        
        mapping = {}
        used_synth = set()
        for i in sorted_real_indices:
            for j, real_name, synth_name in real_options[i]:
                if j not in used_synth:
                    mapping[real_name] = synth_name
                    used_synth.add(j)
                    break
        
        # Calculate F1
        matched_count = len(mapping)
        precision = matched_count / len(synth_col_list) if synth_col_list else 0.0
        recall = matched_count / len(real_col_list) if real_col_list else 0.0
        
        if precision + recall == 0:
            best_f1 = 0.0
        else:
            best_f1 = 2 * precision * recall / (precision + recall)
        best_mapping = mapping
    
    if debug:
        logger.info(f"  DEBUG: Best mapping F1={best_f1:.4f}: {best_mapping}")
        if best_f1 == 0.0:
            logger.warning(f"  DEBUG: WARNING - No columns matched between {real_table.name} and {synth_table.name}!")
    
    return best_mapping, best_f1


def match_columns(
    real_table: TableSpec,
    synth_table: TableSpec,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    threshold: float = 0.0,  # Unused, kept for compatibility
    use_hungarian: bool = True,  # Unused, kept for compatibility
    real_ir: Optional[LogicalIR] = None,  # Unused, kept for compatibility
    synth_ir: Optional[LogicalIR] = None,  # Unused, kept for compatibility
    real_table_name: Optional[str] = None,
    synth_table_name: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float]]:
    """
    Match columns between two tables using binary matching and F1 optimization.
    
    New logic:
    1. For each gold column, check if it can match any generated column:
       - Type must be compatible (hard rule)
       - Name must be compatible via WordNet/semantic/string matching
    2. If matchable → score = 1, otherwise → no match (score = 0)
    3. Find the column mapping that maximizes F1 score:
       - precision = matched_cols / generated_cols
       - recall = matched_cols / gold_cols
       - F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        real_table: Real table specification (gold truth)
        synth_table: Synthetic table specification (generated)
        real_df: Unused, kept for compatibility
        synth_df: Unused, kept for compatibility
        threshold: Unused, kept for compatibility
        use_hungarian: Unused, kept for compatibility
        real_ir: Unused, kept for compatibility
        synth_ir: Unused, kept for compatibility
        real_table_name: Real table name (for logging)
        synth_table_name: Synthetic table name (for logging)
        
    Returns:
        Tuple of (matches_dict, similarity_dict) where:
        - matches_dict: Dictionary mapping real_column -> synth_column
        - similarity_dict: Dictionary mapping (real_column, synth_column) -> 1.0 (all matches are binary)
    """
    logger.info(
        f"Matching columns: {real_table_name or 'real'} ({len(real_table.columns)} cols) vs "
        f"{synth_table_name or 'synth'} ({len(synth_table.columns)} cols) "
        f"(using binary matching + F1 optimization)"
    )
    
    # Find best mapping using F1 score
    best_mapping, best_f1 = _find_best_column_mapping_f1(real_table, synth_table)
    
    logger.info(
        f"Best column mapping F1: {best_f1:.4f} "
        f"(matched {len(best_mapping)}/{len(real_table.columns)} gold columns)"
    )
    
    # Build similarity dict (all matches are 1.0 since they're binary)
    similarities = {}
    for real_name, synth_name in best_mapping.items():
        similarities[(real_name, synth_name)] = 1.0
    
    return best_mapping, similarities

