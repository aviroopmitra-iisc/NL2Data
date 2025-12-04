"""Apriori-based functional dependency discovery from data."""

import itertools
from typing import Dict, List, Tuple, Set
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR
from nl2data.ir.constraint_ir import FDConstraint

# Sampling threshold: use sample for tables larger than this
SAMPLE_THRESHOLD = 100000  # 100K rows
SAMPLE_SIZE = 100000  # Sample 100K rows for FD discovery


def discover_functional_dependencies(
    dfs: Dict[str, pd.DataFrame],
    logical_ir: LogicalIR,
    min_support: float = 0.95,
    min_confidence: float = 0.95,
    max_lhs_size: int = 3
) -> Tuple[List[FDConstraint], Dict[tuple, tuple]]:
    """
    Discover functional dependencies using Apriori algorithm.
    
    Args:
        dfs: Dictionary of table_name -> DataFrame
        logical_ir: LogicalIR with table schemas
        min_support: Minimum support threshold (0.0 to 1.0)
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        max_lhs_size: Maximum number of columns in LHS
        
    Returns:
        Tuple of (discovered_fds, support_confidence_map)
        support_confidence_map: Dict mapping (table, tuple(lhs), rhs) -> (support, confidence)
    """
    discovered_fds = []
    support_confidence_map = {}
    
    total_tables = len([t for t in dfs.keys() if t in logical_ir.tables])
    current_table = 0
    
    for table_name, df in dfs.items():
        if table_name not in logical_ir.tables:
            continue
        
        current_table += 1
        original_row_count = len(df)
        
        # Apply sampling for large tables
        if original_row_count > SAMPLE_THRESHOLD:
            print(f"  [{current_table}/{total_tables}] {table_name}: {original_row_count:,} rows -> sampling {SAMPLE_SIZE:,} rows for FD discovery")
            df = df.sample(n=min(SAMPLE_SIZE, original_row_count), random_state=42).reset_index(drop=True)
        else:
            print(f"  [{current_table}/{total_tables}] {table_name}: {original_row_count:,} rows (using full dataset)")
        
        table = logical_ir.tables[table_name]
        columns = [col.name for col in table.columns]
        
        # Get primary key columns (from DDL) - exclude these from FD discovery
        primary_key_cols = set()
        if table.primary_key:
            primary_key_cols = set(table.primary_key)
        else:
            # Also check column roles for primary_key
            for col in table.columns:
                if col.role == "primary_key":
                    primary_key_cols.add(col.name)
        
        # Get foreign key columns (from DDL) - exclude these from FD discovery
        foreign_key_cols = set()
        for fk in table.foreign_keys:
            foreign_key_cols.add(fk.column)
        # Also check column roles for foreign_key
        for col in table.columns:
            if col.role == "foreign_key" or col.references:
                foreign_key_cols.add(col.name)
        
        # Filter columns that exist in DataFrame
        available_columns = [c for c in columns if c in df.columns]
        
        # Exclude primary key and foreign key columns from FD discovery
        # PK -> all columns is trivial, and FK -> referenced PK is already defined
        excluded_cols = primary_key_cols | foreign_key_cols
        non_key_columns = [c for c in available_columns if c not in excluded_cols]
        
        if len(non_key_columns) < 2:
            print(f"    Skipping: only {len(non_key_columns)} non-key columns (need at least 2)")
            continue
        
        print(f"    Discovering FDs: {len(non_key_columns)} non-key columns, max_lhs_size={max_lhs_size}")
        
        # Apriori: Find frequent itemsets (LHS candidates) - exclude PK and FK columns
        frequent_itemsets = apriori_frequent_itemsets(
            df, non_key_columns, min_support, max_lhs_size
        )
        
        print(f"    Found {len(frequent_itemsets)} frequent itemsets, checking FD candidates...")
        
        # For each frequent itemset (LHS), check all possible RHS (excluding PK and FK columns)
        total_checks = len(frequent_itemsets) * len(non_key_columns)
        checked = 0
        found_fds = 0
        
        for idx, lhs_cols in enumerate(frequent_itemsets):
            rhs_candidates = [c for c in non_key_columns if c not in lhs_cols]
            
            # Progress logging every 100 itemsets
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"      Processing itemset {idx + 1}/{len(frequent_itemsets)}: {lhs_cols}")
            
            for rhs_col in rhs_candidates:
                checked += 1
                confidence = compute_fd_confidence(df, list(lhs_cols), rhs_col)
                
                # Compute support for this specific FD
                support = compute_itemset_support(df, lhs_cols)
                
                if confidence >= min_confidence:
                    fd = FDConstraint(
                        table=table_name,
                        lhs=list(lhs_cols),
                        rhs=[rhs_col],
                        mode="intra_row"
                    )
                    discovered_fds.append(fd)
                    found_fds += 1
                    
                    # Store support and confidence
                    key = (table_name, tuple(lhs_cols), rhs_col)
                    support_confidence_map[key] = (support, confidence)
        
        print(f"    Completed: checked {checked:,} FD candidates, found {found_fds} FDs")
    
    return discovered_fds, support_confidence_map


def apriori_frequent_itemsets(
    df: pd.DataFrame,
    columns: List[str],
    min_support: float,
    max_size: int
) -> List[Tuple[str, ...]]:
    """
    Apriori algorithm to find frequent itemsets (potential LHS).
    
    Args:
        df: DataFrame with data
        columns: List of column names to consider
        min_support: Minimum support threshold
        max_size: Maximum itemset size
        
    Returns:
        List of column tuples that meet support threshold
    """
    # Clean data: remove nulls
    clean_df = df[columns].dropna()
    total_rows = len(clean_df)
    
    if total_rows == 0:
        return []
    
    # Level 1: Single columns
    frequent_1 = []
    for col in columns:
        support = compute_column_support(clean_df, col)
        if support >= min_support:
            frequent_1.append((col,))
    
    if not frequent_1:
        return []
    
    all_frequent = list(frequent_1)
    current_level = frequent_1
    
    # Generate k+1 itemsets from k itemsets
    for k in range(2, min(max_size + 1, len(columns) + 1)):
        # Generate candidates
        candidates = generate_candidates(current_level, k)
        
        if not candidates:
            break
        
        # Compute support and filter
        frequent_k = []
        total_candidates = len(candidates)
        for idx, candidate in enumerate(candidates):
            # Progress logging for large candidate sets
            if total_candidates > 1000 and (idx + 1) % 1000 == 0:
                print(f"        Level {k}: processed {idx + 1:,}/{total_candidates:,} candidates, found {len(frequent_k)} frequent so far")
            
            support = compute_itemset_support(clean_df, candidate)
            if support >= min_support:
                frequent_k.append(candidate)
        
        if not frequent_k:
            break
        
        print(f"        Level {k}: {len(frequent_k)}/{total_candidates} frequent itemsets")
        all_frequent.extend(frequent_k)
        current_level = frequent_k
    
    return all_frequent


def generate_candidates(
    frequent_k_minus_1: List[Tuple[str, ...]],
    k: int
) -> List[Tuple[str, ...]]:
    """
    Generate k-itemset candidates from (k-1)-itemsets.
    Join itemsets that share first (k-2) elements.
    
    Args:
        frequent_k_minus_1: List of (k-1)-itemsets
        k: Target itemset size
        
    Returns:
        List of k-itemset candidates
    """
    candidates = []
    
    for i in range(len(frequent_k_minus_1)):
        for j in range(i + 1, len(frequent_k_minus_1)):
            itemset1 = frequent_k_minus_1[i]
            itemset2 = frequent_k_minus_1[j]
            
            # Check if first k-2 elements are the same
            if k > 2 and itemset1[:k-2] == itemset2[:k-2]:
                # Join: combine both itemsets
                candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                if len(candidate) == k:
                    candidates.append(candidate)
            elif k == 2:
                # For k=2, just combine any two 1-itemsets
                candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                if len(candidate) == 2:
                    candidates.append(candidate)
    
    return candidates


def compute_column_support(df: pd.DataFrame, col: str) -> float:
    """
    Compute support for a single column (proportion of non-null unique values).
    
    Args:
        df: DataFrame
        col: Column name
        
    Returns:
        Support value (0.0 to 1.0)
    """
    unique_count = df[col].nunique()
    total_count = len(df)
    return unique_count / total_count if total_count > 0 else 0.0


def compute_itemset_support(df: pd.DataFrame, itemset: Tuple[str, ...]) -> float:
    """
    Compute support for an itemset (proportion of unique combinations).
    
    Args:
        df: DataFrame
        itemset: Tuple of column names
        
    Returns:
        Support value (0.0 to 1.0)
    """
    if len(itemset) == 0:
        return 0.0
    
    grouped = df[list(itemset)].groupby(list(itemset))
    unique_combinations = len(grouped)
    total_rows = len(df)
    return unique_combinations / total_rows if total_rows > 0 else 0.0


def compute_fd_confidence(
    df: pd.DataFrame,
    lhs_cols: List[str],
    rhs_col: str
) -> float:
    """
    Compute confidence for a functional dependency.
    
    Confidence = P(RHS | LHS) = proportion of rows where LHS uniquely determines RHS
    
    Args:
        df: DataFrame
        lhs_cols: List of LHS column names
        rhs_col: RHS column name
        
    Returns:
        Confidence value (0.0 to 1.0)
    """
    # Remove nulls in LHS or RHS
    clean_df = df[lhs_cols + [rhs_col]].dropna()
    
    if len(clean_df) == 0:
        return 0.0
    
    # Group by LHS
    grouped = clean_df.groupby(lhs_cols)
    
    # Confidence: proportion of rows in groups with unique RHS
    rows_with_unique_rhs = 0
    total_rows = len(clean_df)
    
    for name, group in grouped:
        if group[rhs_col].nunique() == 1:
            rows_with_unique_rhs += len(group)
    
    confidence = rows_with_unique_rhs / total_rows if total_rows > 0 else 0.0
    
    return confidence

