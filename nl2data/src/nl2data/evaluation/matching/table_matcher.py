"""Table-level schema matching."""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger
from .similarity import name_similarity

logger = get_logger(__name__)

# Lazy import for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None


def compute_table_similarity(
    real_name: str,
    synth_name: str,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
) -> float:
    """
    Compute similarity score between two tables using column F1 score.
    
    For each table pair, calculates the column F1 score (from column matching).
    This is the only signal used for table similarity now.
    Cardinality scoring will be added later.
    
    Args:
        real_name: Real table name
        synth_name: Synthetic table name
        real_df: Real table DataFrame (unused for now)
        synth_df: Synthetic table DataFrame (unused for now)
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        
    Returns:
        Column F1 score [0,1] for this table pair
    """
    real_table = real_ir.tables.get(real_name)
    synth_table = synth_ir.tables.get(synth_name)
    
    if not real_table or not synth_table:
        return 0.0
    
    # Calculate column F1 score using the new binary matching logic
    from .column_matcher import _find_best_column_mapping_f1
    
    _, column_f1 = _find_best_column_mapping_f1(real_table, synth_table)
    
    return float(column_f1)


def solve_bipartite_matching(
    similarity_matrix: List[List[float]],
    threshold: float = 0.0,  # No threshold - Hungarian finds best alignment
    use_hungarian: bool = True,
) -> List[Tuple[int, int]]:
    """
    Solve bipartite matching to maximize total similarity.
    
    Uses Hungarian algorithm to find optimal alignment without threshold filtering.
    Only filters out matches with 0.0 similarity (hard incompatibility).
    
    Args:
        similarity_matrix: Matrix of similarities [real_idx][synth_idx]
        threshold: Ignored - kept for backward compatibility only
        use_hungarian: Whether to use Hungarian algorithm (else greedy)
        
    Returns:
        List of (real_idx, synth_idx) pairs
    """
    if not similarity_matrix or not similarity_matrix[0]:
        return []
    
    n_real = len(similarity_matrix)
    n_synth = len(similarity_matrix[0])
    
    if use_hungarian and SCIPY_AVAILABLE and n_real <= 20 and n_synth <= 20:
        # Use Hungarian algorithm for small problems
        try:
            # Convert to cost matrix (negate similarities for minimization)
            cost_matrix = [[-sim for sim in row] for row in similarity_matrix]
            real_indices, synth_indices = linear_sum_assignment(cost_matrix)
            
            # Only filter out hard incompatibilities (0.0 similarity)
            matches = [
                (int(r), int(s))
                for r, s in zip(real_indices, synth_indices)
                if similarity_matrix[r][s] > 0.0  # Only exclude hard incompatibilities
            ]
            return matches
        except Exception as e:
            logger.warning(f"Hungarian algorithm failed: {e}, falling back to greedy")
            use_hungarian = False
    
    # Greedy matching (fallback or for large problems)
    matches = []
    used_synth = set()
    
    # Sort all pairs by similarity (descending)
    all_pairs = []
    for r in range(n_real):
        for s in range(n_synth):
            sim = similarity_matrix[r][s]
            if sim > 0.0:  # Only exclude hard incompatibilities
                all_pairs.append((sim, r, s))
    
    all_pairs.sort(reverse=True)
    
    # Greedily assign
    for sim, r, s in all_pairs:
        if s not in used_synth:
            matches.append((r, s))
            used_synth.add(s)
    
    return matches


def match_tables(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    threshold: float = 0.0,  # No threshold - Hungarian finds best alignment
    use_hungarian: bool = True,
) -> Dict[str, str]:
    """
    Match tables between real and synthetic schemas.
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        threshold: Minimum similarity to match
        use_hungarian: Whether to use Hungarian algorithm
        
    Returns:
        Dictionary mapping real_table -> synth_table
    """
    real_tables = list(real_ir.tables.keys())
    synth_tables = list(synth_ir.tables.keys())
    
    if not real_tables or not synth_tables:
        return {}
    
    # Compute similarity matrix
    similarity_matrix = []
    logger.info(f"Computing table similarity matrix: {len(real_tables)} expected vs {len(synth_tables)} generated")
    for real_name in real_tables:
        row = []
        real_df = real_dfs.get(real_name, pd.DataFrame())
        for synth_name in synth_tables:
            synth_df = synth_dfs.get(synth_name, pd.DataFrame())
            sim = compute_table_similarity(
                real_name, synth_name, real_df, synth_df, real_ir, synth_ir
            )
            row.append(sim)
            # Log specific matches for debugging
            if real_name == synth_name:
                logger.info(f"  {real_name} vs {synth_name}: similarity = {sim:.4f}")
        similarity_matrix.append(row)
    
    # Solve bipartite matching
    matches = solve_bipartite_matching(
        similarity_matrix, threshold=threshold, use_hungarian=use_hungarian
    )
    
    # Build result dictionary
    result = {}
    for real_idx, synth_idx in matches:
        result[real_tables[real_idx]] = synth_tables[synth_idx]
    
    return result

