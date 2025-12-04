"""Candidate key discovery from functional dependencies."""

from typing import Dict, List, Set, Tuple
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR
from nl2data.ir.constraint_ir import FDConstraint


def find_candidate_keys(
    df: pd.DataFrame,
    table_name: str,
    logical_ir: LogicalIR,
    discovered_fds: List[FDConstraint],
    support_confidence_map: Dict[tuple, tuple],
    min_support: float = 1.0,
    min_confidence: float = 1.0
) -> List[List[str]]:
    """
    Find candidate keys for a table.
    
    A candidate key is a set of columns (LHS) that functionally determines
    ALL other columns in the table with 1.0 confidence and support.
    
    Args:
        df: DataFrame with table data
        table_name: Name of the table
        logical_ir: LogicalIR with table schemas
        discovered_fds: List of discovered functional dependencies
        support_confidence_map: Dict mapping (table, tuple(lhs), rhs) -> (support, confidence)
        min_support: Minimum support for candidate keys (default 1.0)
        min_confidence: Minimum confidence for candidate keys (default 1.0)
        
    Returns:
        List of candidate keys, where each candidate key is a list of column names
    """
    if table_name not in logical_ir.tables:
        return []
    
    table = logical_ir.tables[table_name]
    all_columns = set(col.name for col in table.columns)
    
    # Filter columns that exist in DataFrame
    available_columns = [c for c in all_columns if c in df.columns]
    available_columns_set = set(available_columns)
    
    if len(available_columns) < 1:
        return []
    
    # Filter FDs for this table with perfect confidence and support
    table_fds = [
        fd for fd in discovered_fds
        if fd.table == table_name
        and len(fd.rhs) == 1  # Single RHS column
    ]
    
    # Build FD closure: for each LHS, what columns can it determine?
    # Use a more efficient closure computation algorithm
    
    # First, build a direct FD map: LHS -> set of RHS columns
    direct_fds: Dict[Tuple[str, ...], Set[str]] = {}
    
    for fd in table_fds:
        lhs_tuple = tuple(sorted(fd.lhs))
        rhs_col = fd.rhs[0]
        
        # Check support and confidence
        key = (table_name, tuple(fd.lhs), rhs_col)
        if key in support_confidence_map:
            support, confidence = support_confidence_map[key]
            if support >= min_support and confidence >= min_confidence:
                if lhs_tuple not in direct_fds:
                    direct_fds[lhs_tuple] = set()
                direct_fds[lhs_tuple].add(rhs_col)
    
    # Compute closure for each LHS using fixed-point iteration
    fd_closure: Dict[Tuple[str, ...], Set[str]] = {}
    
    # Initialize closure: each LHS determines itself
    for lhs_tuple in direct_fds.keys():
        fd_closure[lhs_tuple] = set(lhs_tuple)
    
    # Compute transitive closure
    changed = True
    max_iterations = len(available_columns)  # Prevent infinite loops
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for lhs_tuple in list(fd_closure.keys()):
            determined = fd_closure[lhs_tuple].copy()
            
            # For each subset of determined columns, check if it determines more columns
            # Check all FDs where LHS is a subset of determined columns
            for fd_lhs_tuple, rhs_cols in direct_fds.items():
                if set(fd_lhs_tuple).issubset(determined):
                    # This FD applies, add all RHS columns
                    for rhs_col in rhs_cols:
                        if rhs_col not in determined:
                            determined.add(rhs_col)
                            changed = True
            
            if determined != fd_closure[lhs_tuple]:
                fd_closure[lhs_tuple] = determined
                changed = True
    
    # Find candidate keys: LHS that determine ALL columns
    candidate_keys = []
    for lhs_tuple, determined in fd_closure.items():
        # Check if this LHS determines all available columns
        if determined.issuperset(available_columns_set):
            candidate_keys.append(list(lhs_tuple))
    
    # Also check single columns that might be candidate keys
    # (if they weren't found through FDs)
    for col in available_columns:
        # Check if this column is unique (potential candidate key)
        if df[col].nunique() == len(df[col].dropna()):
            # This column is unique, check if it's already in candidate_keys
            if [col] not in candidate_keys:
                # Verify it determines all columns through FDs or direct uniqueness
                col_tuple = (col,)
                if col_tuple in fd_closure:
                    if fd_closure[col_tuple].issuperset(available_columns_set):
                        if [col] not in candidate_keys:
                            candidate_keys.append([col])
                else:
                    # If no FDs found but column is unique, it's a candidate key
                    # (assuming it determines all columns by uniqueness)
                    if [col] not in candidate_keys:
                        candidate_keys.append([col])
    
    return candidate_keys


def find_minimal_candidate_keys(candidate_keys: List[List[str]]) -> List[List[str]]:
    """
    Find minimal candidate keys (no proper subset is also a candidate key).
    
    Args:
        candidate_keys: List of candidate keys (each is a list of column names)
        
    Returns:
        List of minimal candidate keys
    """
    if not candidate_keys:
        return []
    
    # Sort by size (smallest first)
    sorted_keys = sorted(candidate_keys, key=len)
    
    minimal_keys = []
    
    for candidate in sorted_keys:
        candidate_set = set(candidate)
        
        # Check if any existing minimal key is a proper subset of this candidate
        is_superset = False
        for minimal in minimal_keys:
            if set(minimal).issubset(candidate_set) and len(minimal) < len(candidate):
                is_superset = True
                break
        
        # If this candidate is not a superset of any minimal key, it's minimal
        if not is_superset:
            # Also check if we should remove any existing minimal keys that are supersets
            minimal_keys = [
                m for m in minimal_keys
                if not (candidate_set.issubset(set(m)) and len(candidate_set) < len(m))
            ]
            minimal_keys.append(candidate)
    
    return minimal_keys


def separate_candidate_key_fds(
    discovered_fds: List[FDConstraint],
    candidate_keys: List[List[str]],
    table_name: str
) -> Tuple[List[FDConstraint], List[FDConstraint]]:
    """
    Separate FDs into candidate-key FDs and regular FDs.
    
    Candidate-key FDs are those where the LHS is a candidate key.
    Regular FDs are all others.
    
    Args:
        discovered_fds: List of all discovered FDs
        candidate_keys: List of candidate keys for the table
        table_name: Name of the table
        
    Returns:
        Tuple of (candidate_key_fds, regular_fds)
    """
    candidate_key_sets = [set(ck) for ck in candidate_keys]
    
    candidate_key_fds = []
    regular_fds = []
    
    for fd in discovered_fds:
        if fd.table != table_name:
            regular_fds.append(fd)
            continue
        
        lhs_set = set(fd.lhs)
        
        # Check if LHS is a candidate key (exact match)
        is_candidate_key_fd = any(lhs_set == ck_set for ck_set in candidate_key_sets)
        
        if is_candidate_key_fd:
            candidate_key_fds.append(fd)
        else:
            regular_fds.append(fd)
    
    return candidate_key_fds, regular_fds

