"""Enhanced schema matching algorithm for evaluation framework.

This module implements the complete enhanced matching algorithm as specified
in the evaluation framework plan, including:
- Column-column similarity matrix construction
- Column alignment score computation
- Table-table similarity matrix construction
- Global table matching via Hungarian algorithm
- Column matching within mapped table pairs
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec
from nl2data.evaluation.matching.similarity import (
    name_similarity,
    hard_incompatible_datatype,
    type_compatibility,
    range_sim,
    get_column_role,
    role_sim,
    fd_sim,
)
from nl2data.evaluation.utils.normalization import normalize_name, compute_name_embedding
from nl2data.evaluation.utils.fd_utils import (
    compute_fd_counts,
    compute_fd_signature,
    compute_column_summaries,
)
from nl2data.evaluation.config import MultiTableEvalConfig
from nl2data.evaluation.models.multi_table import (
    SchemaMatchResult,
    TableMatch,
    ColumnMatch,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_schemas(
    real_ir: LogicalIR,
    synth_ir: LogicalIR
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Preprocess schemas: sort tables and columns deterministically.
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        
    Returns:
        Tuple of:
        - sorted_real_tables: List of sorted table names from real_ir
        - sorted_synth_tables: List of sorted table names from synth_ir
        - real_table_columns: Dict mapping table_name -> sorted column names
        - synth_table_columns: Dict mapping table_name -> sorted column names
    """
    # Sort tables lexicographically
    sorted_real_tables = sorted(real_ir.tables.keys())
    sorted_synth_tables = sorted(synth_ir.tables.keys())
    
    # Sort columns for each table
    real_table_columns = {}
    for table_name in sorted_real_tables:
        table = real_ir.tables[table_name]
        real_table_columns[table_name] = sorted([c.name for c in table.columns])
    
    synth_table_columns = {}
    for table_name in sorted_synth_tables:
        table = synth_ir.tables[table_name]
        synth_table_columns[table_name] = sorted([c.name for c in table.columns])
    
    return sorted_real_tables, sorted_synth_tables, real_table_columns, synth_table_columns


def compute_embeddings(
    real_ir: LogicalIR,
    synth_ir: LogicalIR
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Compute deterministic embeddings for table and column names.
    
    Uses simple character n-gram embeddings (can be enhanced later).
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        
    Returns:
        Tuple of:
        - table_embeddings: Dict mapping table_name -> embedding set
        - column_embeddings: Dict mapping (table_name, col_name) -> embedding set
    """
    table_embeddings = {}
    column_embeddings = {}
    
    # Real schema embeddings
    for table_name, table in real_ir.tables.items():
        table_embeddings[table_name] = compute_name_embedding(table_name)
        for col in table.columns:
            column_embeddings[(table_name, col.name)] = compute_name_embedding(col.name)
    
    # Synthetic schema embeddings
    for table_name, table in synth_ir.tables.items():
        table_embeddings[table_name] = compute_name_embedding(table_name)
        for col in table.columns:
            column_embeddings[(table_name, col.name)] = compute_name_embedding(col.name)
    
    return table_embeddings, column_embeddings


def cosine_similarity_embeddings(emb1: Set[str], emb2: Set[str]) -> float:
    """
    Compute cosine similarity between two embedding sets.
    
    For n-gram embeddings, this is essentially Jaccard similarity.
    
    Args:
        emb1, emb2: Sets of n-grams
        
    Returns:
        Similarity score [0,1]
    """
    if not emb1 and not emb2:
        return 1.0
    if not emb1 or not emb2:
        return 0.0
    
    intersection = len(emb1 & emb2)
    union = len(emb1 | emb2)
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Column-Column Similarity Matrix Construction
# ============================================================================

def compute_column_similarity_matrix(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    synth_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    real_fd_counts: Dict[str, Dict[str, Dict[str, int]]],
    synth_fd_counts: Dict[str, Dict[str, Dict[str, int]]],
    column_embeddings: Dict[Tuple[str, str], Set[str]],
    config: MultiTableEvalConfig
) -> Tuple[Dict[Tuple[str, str, str, str], float], Dict[Tuple[str, str], Tuple[str, str]]]:
    """
    Compute column-column similarity matrix for all column pairs.
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        real_summaries: Column summaries for real schema
        synth_summaries: Column summaries for synthetic schema
        real_fd_counts: FD counts for real schema
        synth_fd_counts: FD counts for synthetic schema
        column_embeddings: Column name embeddings
        config: Evaluation configuration
        
    Returns:
        Tuple of:
        - S_col: Dict mapping (real_table, real_col, synth_table, synth_col) -> similarity score
        - column_index_map: Dict mapping (table, col) -> global index (for matrix access)
    """
    # Collect all columns with their table context
    all_real_cols = []  # List of (table_name, col_name, col_spec)
    all_synth_cols = []  # List of (table_name, col_name, col_spec)
    
    for table_name, table in real_ir.tables.items():
        for col in table.columns:
            all_real_cols.append((table_name, col.name, col, table))
    
    for table_name, table in synth_ir.tables.items():
        for col in table.columns:
            all_synth_cols.append((table_name, col.name, col, table))
    
    # Create index mappings
    real_col_to_idx = {(t, c): i for i, (t, c, _, _) in enumerate(all_real_cols)}
    synth_col_to_idx = {(t, c): i for i, (t, c, _, _) in enumerate(all_synth_cols)}
    column_index_map = {**real_col_to_idx, **synth_col_to_idx}
    
    # Compute similarity matrix
    S_col = {}
    
    for real_table, real_col, real_col_spec, real_table_spec in all_real_cols:
        for synth_table, synth_col, synth_col_spec, synth_table_spec in all_synth_cols:
            key = (real_table, real_col, synth_table, synth_col)
            
            # Check hard incompatibility
            if hard_incompatible_datatype(real_col_spec.sql_type, synth_col_spec.sql_type):
                S_col[key] = float('-inf')
                continue
            
            # Compute name similarity (using embeddings)
            real_emb = column_embeddings.get((real_table, real_col), set())
            synth_emb = column_embeddings.get((synth_table, synth_col), set())
            name_sim_val = max(cosine_similarity_embeddings(real_emb, synth_emb), 0.0)
            
            # Compute range similarity
            real_summary = real_summaries.get(real_table, {}).get(real_col, {})
            synth_summary = synth_summaries.get(synth_table, {}).get(synth_col, {})
            range_sim_val = range_sim(real_summary, synth_summary, config.matching.alpha_cat)
            
            # Compute role similarity
            real_role = get_column_role(real_col_spec, real_table_spec)
            synth_role = get_column_role(synth_col_spec, synth_table_spec)
            role_sim_val = role_sim(real_role, synth_role)
            
            # Compute FD similarity
            real_fd = real_fd_counts.get(real_table, {}).get(real_col, {"lhs": 0, "rhs": 0})
            synth_fd = synth_fd_counts.get(synth_table, {}).get(synth_col, {"lhs": 0, "rhs": 0})
            fd_sim_val = fd_sim(
                real_fd["lhs"], real_fd["rhs"],
                synth_fd["lhs"], synth_fd["rhs"],
                config.matching.lambda_fd,
                config.matching.mu_fd
            )
            
            # Combine into column similarity
            similarity = (
                config.matching.w_name_col * name_sim_val +
                config.matching.w_range_col * range_sim_val +
                config.matching.w_role_col * role_sim_val +
                config.matching.w_FD_col * fd_sim_val
            )
            
            S_col[key] = float(similarity)
    
    return S_col, column_index_map


# ============================================================================
# Column Alignment Score
# ============================================================================

def compute_column_alignment_score(
    real_table: str,
    synth_table: str,
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    S_col: Dict[Tuple[str, str, str, str], float],
    config: MultiTableEvalConfig
) -> float:
    """
    Compute column alignment score F_tilde(a,b) between two tables.
    
    Uses Hungarian algorithm to find optimal column matching.
    
    Args:
        real_table: Real table name
        synth_table: Synthetic table name
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        S_col: Column similarity matrix
        config: Evaluation configuration
        
    Returns:
        Normalized alignment score F_tilde(a,b) in [0,1]
    """
    real_table_spec = real_ir.tables.get(real_table)
    synth_table_spec = synth_ir.tables.get(synth_table)
    
    if not real_table_spec or not synth_table_spec:
        return 0.0
    
    # Get sorted columns
    colsA = sorted([c.name for c in real_table_spec.columns])
    colsB = sorted([c.name for c in synth_table_spec.columns])
    
    p = len(colsA)
    q = len(colsB)
    
    if p == 0 and q == 0:
        return 1.0  # Both empty, perfect match
    if p == 0 or q == 0:
        return 0.0  # One empty, no alignment
    
    # Build local similarity matrix M
    M = np.zeros((p, q))
    for u, col_a in enumerate(colsA):
        for v, col_b in enumerate(colsB):
            key = (real_table, col_a, synth_table, col_b)
            sim = S_col.get(key, 0.0)
            # Convert -inf to 0.0 for matching
            M[u, v] = max(sim, 0.0) if sim != float('-inf') else 0.0
    
    # Pad to square matrix
    s = max(p, q)
    M_padded = np.zeros((s, s))
    M_padded[:p, :q] = M
    
    # Find maximum value
    C_max_col = float(np.max(M_padded)) if s > 0 else 0.0
    
    # Construct cost matrix for Hungarian algorithm
    cost = C_max_col - M_padded
    
    # Run Hungarian algorithm
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception as e:
        logger.warning(f"Hungarian algorithm failed for {real_table} vs {synth_table}: {e}")
        return 0.0
    
    # Compute aggregated alignment score using all matches from Hungarian algorithm
    F_ab = 0.0
    for u, v in zip(row_ind, col_ind):
        if u < p and v < q:
            col_a = colsA[u]
            col_b = colsB[v]
            key = (real_table, col_a, synth_table, col_b)
            sim = S_col.get(key, 0.0)
            
            # Use all matches (only skip hard incompatibilities marked as -inf)
            if sim != float('-inf'):
                F_ab += sim
    
    # Normalize
    denom = max(p, q) if max(p, q) > 0 else 1
    F_tilde = F_ab / denom
    
    return float(max(0.0, min(1.0, F_tilde)))


# ============================================================================
# Table-Table Similarity Matrix Construction
# ============================================================================

def compute_table_similarity_matrix(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    real_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    synth_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    table_embeddings: Dict[str, Set[str]],
    column_alignment_scores: Dict[Tuple[str, str], float],
    config: MultiTableEvalConfig
) -> Dict[Tuple[str, str], float]:
    """
    Compute table-table similarity matrix.
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        real_summaries: Column summaries for real schema
        synth_summaries: Column summaries for synthetic schema
        table_embeddings: Table name embeddings
        column_alignment_scores: Precomputed column alignment scores F_tilde(a,b)
        config: Evaluation configuration
        
    Returns:
        Dict mapping (real_table, synth_table) -> similarity score
    """
    S_tab = {}
    
    for real_table_name in real_ir.tables.keys():
        for synth_table_name in synth_ir.tables.keys():
            # Compute table-name similarity
            real_emb = table_embeddings.get(real_table_name, set())
            synth_emb = table_embeddings.get(synth_table_name, set())
            name_sim_tab = max(cosine_similarity_embeddings(real_emb, synth_emb), 0.0)
            
            # Get column-alignment strength
            F_tilde = column_alignment_scores.get((real_table_name, synth_table_name), 0.0)
            
            # Compute cardinality similarity
            r_a = len(real_dfs.get(real_table_name, pd.DataFrame()))
            r_b = len(synth_dfs.get(synth_table_name, pd.DataFrame()))
            cardinality_sim = np.exp(
                -config.matching.eta * abs(np.log(r_a + 1) - np.log(r_b + 1))
            )
            
            # Compute FD signature similarity
            phi_a = compute_fd_signature(real_ir, real_table_name)
            phi_b = compute_fd_signature(synth_ir, synth_table_name)
            
            # Cosine similarity of FD signatures
            if np.linalg.norm(phi_a) == 0 and np.linalg.norm(phi_b) == 0:
                FD_sig_sim = 1.0  # Both have no FDs
            elif np.linalg.norm(phi_a) == 0 or np.linalg.norm(phi_b) == 0:
                FD_sig_sim = 0.0  # One has FDs, other doesn't
            else:
                try:
                    cos_sim = 1.0 - cosine(phi_a, phi_b)
                    FD_sig_sim = max(cos_sim, 0.0)
                except Exception:
                    FD_sig_sim = 0.0
            
            # Combine into table similarity
            similarity = (
                config.matching.alpha * name_sim_tab +
                config.matching.beta * F_tilde +
                config.matching.gamma * cardinality_sim +
                config.matching.delta * FD_sig_sim
            )
            
            S_tab[(real_table_name, synth_table_name)] = float(max(0.0, min(1.0, similarity)))
    
    return S_tab


# ============================================================================
# Global Table Matching via Hungarian
# ============================================================================

def match_tables_enhanced(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    S_tab: Dict[Tuple[str, str], float],
    config: MultiTableEvalConfig
) -> Tuple[List[Tuple[str, str, float]], List[str], List[str]]:
    """
    Match tables using Hungarian algorithm.
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        S_tab: Table similarity matrix
        config: Evaluation configuration
        
    Returns:
        Tuple of:
        - M_T: List of (real_table, synth_table, score) tuples
        - unmapped_real_tables: List of unmatched real table names
        - unmapped_synth_tables: List of unmatched synthetic table names
    """
    sorted_real_tables = sorted(real_ir.tables.keys())
    sorted_synth_tables = sorted(synth_ir.tables.keys())
    
    m = len(sorted_real_tables)
    n = len(sorted_synth_tables)
    
    if m == 0 or n == 0:
        return [], sorted_real_tables, sorted_synth_tables
    
    # Build similarity matrix T
    T = np.zeros((m, n))
    for x, real_table in enumerate(sorted_real_tables):
        for y, synth_table in enumerate(sorted_synth_tables):
            T[x, y] = S_tab.get((real_table, synth_table), 0.0)
    
    # Pad to square
    s = max(m, n)
    T_ext = np.zeros((s, s))
    T_ext[:m, :n] = T
    
    # Find maximum
    C_max_tab = float(np.max(T_ext)) if s > 0 else 0.0
    
    # Construct cost matrix
    cost_tab = C_max_tab - T_ext
    
    # Run Hungarian algorithm
    try:
        row_ind_tab, col_ind_tab = linear_sum_assignment(cost_tab)
    except Exception as e:
        logger.warning(f"Hungarian algorithm failed for table matching: {e}")
        return [], sorted_real_tables, sorted_synth_tables
    
    # Process results - accept all matches from Hungarian algorithm
    M_T = []
    mapped_real = set()
    mapped_synth = set()
    
    for x, y in zip(row_ind_tab, col_ind_tab):
        if x < m and y < n:
            real_table = sorted_real_tables[x]
            synth_table = sorted_synth_tables[y]
            score = S_tab.get((real_table, synth_table), 0.0)
            
            # Always accept the match found by Hungarian algorithm (optimal matching)
            M_T.append((real_table, synth_table, score))
            mapped_real.add(real_table)
            mapped_synth.add(synth_table)
    
    # Find unmapped tables
    unmapped_real_tables = [t for t in sorted_real_tables if t not in mapped_real]
    unmapped_synth_tables = [t for t in sorted_synth_tables if t not in mapped_synth]
    
    return M_T, unmapped_real_tables, unmapped_synth_tables


# ============================================================================
# Column Matching Within Mapped Table Pairs
# ============================================================================

def match_columns_enhanced(
    real_table: str,
    synth_table: str,
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    S_col: Dict[Tuple[str, str, str, str], float],
    config: MultiTableEvalConfig
) -> Tuple[List[Tuple[str, str, float]], List[str], List[str]]:
    """
    Match columns within a mapped table pair using Hungarian algorithm.
    
    Args:
        real_table: Real table name
        synth_table: Synthetic table name
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        S_col: Column similarity matrix
        config: Evaluation configuration
        
    Returns:
        Tuple of:
        - M_C_ab: List of (real_col, synth_col, score) tuples
        - unmapped_real_cols: List of unmatched real column names
        - unmapped_synth_cols: List of unmatched synthetic column names
    """
    real_table_spec = real_ir.tables.get(real_table)
    synth_table_spec = synth_ir.tables.get(synth_table)
    
    if not real_table_spec or not synth_table_spec:
        return [], [], []
    
    # Get sorted columns
    colsA = sorted([c.name for c in real_table_spec.columns])
    colsB = sorted([c.name for c in synth_table_spec.columns])
    
    p = len(colsA)
    q = len(colsB)
    
    if p == 0 or q == 0:
        return [], colsA, colsB
    
    # Build local similarity matrix
    M_loc = np.zeros((p, q))
    for u, col_a in enumerate(colsA):
        for v, col_b in enumerate(colsB):
            key = (real_table, col_a, synth_table, col_b)
            sim = S_col.get(key, 0.0)
            M_loc[u, v] = max(sim, 0.0) if sim != float('-inf') else 0.0
    
    # Pad to square
    s = max(p, q)
    M_loc_padded = np.zeros((s, s))
    M_loc_padded[:p, :q] = M_loc
    
    # Find maximum
    C_max_col_loc = float(np.max(M_loc_padded)) if s > 0 else 0.0
    
    # Construct cost matrix
    cost_loc = C_max_col_loc - M_loc_padded
    
    # Run Hungarian algorithm
    try:
        row_ind_loc, col_ind_loc = linear_sum_assignment(cost_loc)
    except Exception as e:
        logger.warning(f"Hungarian algorithm failed for columns in {real_table} vs {synth_table}: {e}")
        return [], colsA, colsB
    
    # Process results - accept all matches from Hungarian algorithm
    M_C_ab = []
    mapped_real_cols = set()
    mapped_synth_cols = set()
    
    for u, v in zip(row_ind_loc, col_ind_loc):
        if u < p and v < q:
            col_a = colsA[u]
            col_b = colsB[v]
            key = (real_table, col_a, synth_table, col_b)
            sim = S_col.get(key, 0.0)
            
            # Always accept the match found by Hungarian algorithm (optimal matching)
            # Only skip if similarity is explicitly -inf (hard incompatibility)
            if sim != float('-inf'):
                M_C_ab.append((col_a, col_b, sim))
                mapped_real_cols.add(col_a)
                mapped_synth_cols.add(col_b)
    
    # Find unmapped columns
    unmapped_real_cols = [c for c in colsA if c not in mapped_real_cols]
    unmapped_synth_cols = [c for c in colsB if c not in mapped_synth_cols]
    
    return M_C_ab, unmapped_real_cols, unmapped_synth_cols


# ============================================================================
# Main Enhanced Matching Function
# ============================================================================

def match_schemas_enhanced(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    config: MultiTableEvalConfig
) -> SchemaMatchResult:
    """
    Complete enhanced schema matching algorithm.
    
    This implements the full algorithm from the evaluation framework plan:
    1. Preprocessing (sorting, normalization)
    2. Compute column summaries
    3. Compute FD counts and signatures
    4. Compute column similarity matrix
    5. Compute column alignment scores
    6. Compute table similarity matrix
    7. Match tables via Hungarian algorithm
    8. Match columns within each mapped table pair
    
    Args:
        real_ir: Original LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        config: Evaluation configuration
        
    Returns:
        SchemaMatchResult with complete matching information
    """
    logger.info("Starting enhanced schema matching")
    
    # Step 1: Preprocessing
    logger.debug("Step 1: Preprocessing schemas...")
    sorted_real_tables, sorted_synth_tables, _, _ = preprocess_schemas(real_ir, synth_ir)
    
    # Step 2: Compute embeddings
    logger.debug("Step 2: Computing name embeddings...")
    table_embeddings, column_embeddings = compute_embeddings(real_ir, synth_ir)
    
    # Step 3: Compute column summaries
    logger.debug("Step 3: Computing column summaries...")
    real_summaries = compute_column_summaries(real_ir, real_dfs)
    synth_summaries = compute_column_summaries(synth_ir, synth_dfs)
    
    # Step 4: Compute FD counts
    logger.debug("Step 4: Computing FD counts...")
    real_fd_counts = compute_fd_counts(real_ir)
    synth_fd_counts = compute_fd_counts(synth_ir)
    
    # Step 5: Compute column similarity matrix
    logger.debug("Step 5: Computing column similarity matrix...")
    S_col, _ = compute_column_similarity_matrix(
        real_ir, synth_ir,
        real_summaries, synth_summaries,
        real_fd_counts, synth_fd_counts,
        column_embeddings,
        config
    )
    
    # Step 6: Compute column alignment scores for all table pairs
    logger.debug("Step 6: Computing column alignment scores...")
    column_alignment_scores = {}
    for real_table in sorted_real_tables:
        for synth_table in sorted_synth_tables:
            F_tilde = compute_column_alignment_score(
                real_table, synth_table,
                real_ir, synth_ir,
                S_col, config
            )
            column_alignment_scores[(real_table, synth_table)] = F_tilde
    
    # Step 7: Compute table similarity matrix
    logger.debug("Step 7: Computing table similarity matrix...")
    S_tab = compute_table_similarity_matrix(
        real_ir, synth_ir,
        real_dfs, synth_dfs,
        real_summaries, synth_summaries,
        table_embeddings,
        column_alignment_scores,
        config
    )
    
    # Step 8: Match tables via Hungarian
    logger.debug("Step 8: Matching tables via Hungarian algorithm...")
    M_T, unmapped_real_tables, unmapped_synth_tables = match_tables_enhanced(
        real_ir, synth_ir, S_tab, config
    )
    
    logger.info(f"Matched {len(M_T)} table pairs")
    
    # Step 9: Match columns within each mapped table pair
    logger.debug("Step 9: Matching columns within table pairs...")
    table_matches = []
    column_matches = {}
    unmatched_real_columns = {}
    unmatched_synth_columns = {}
    
    for real_table, synth_table, table_score in M_T:
        # Store table match
        table_matches.append(
            TableMatch(
                real_table=real_table,
                synth_table=synth_table,
                similarity=table_score
            )
        )
        
        # Match columns
        M_C_ab, unmapped_real_cols, unmapped_synth_cols = match_columns_enhanced(
            real_table, synth_table,
            real_ir, synth_ir,
            S_col, config
        )
        
        # Store column matches
        column_matches[real_table] = [
            ColumnMatch(real_column=r, synth_column=s, similarity=score)
            for r, s, score in M_C_ab
        ]
        
        # Store unmapped columns
        unmatched_real_columns[real_table] = unmapped_real_cols
        unmatched_synth_columns[synth_table] = unmapped_synth_cols
    
    # Compute coverage factors
    total_real_tables = len(real_ir.tables)
    table_coverage = len(M_T) / max(total_real_tables, 1) if total_real_tables > 0 else 0.0
    
    column_coverage = {}
    for real_table, synth_table, _ in M_T:
        real_table_spec = real_ir.tables.get(real_table)
        if real_table_spec:
            matched_cols = len(column_matches.get(real_table, []))
            total_cols = len(real_table_spec.columns)
            column_coverage[real_table] = matched_cols / max(total_cols, 1) if total_cols > 0 else 0.0
    
    # Build result
    result = SchemaMatchResult(
        table_matches=table_matches,
        column_matches=column_matches,
        unmatched_real_tables=unmapped_real_tables,
        unmatched_synth_tables=unmapped_synth_tables,
        unmatched_real_columns=unmatched_real_columns,
        unmatched_synth_columns=unmatched_synth_columns,
        table_coverage=table_coverage,
        column_coverage=column_coverage,
    )
    
    logger.info("Enhanced schema matching completed")
    return result

