"""Name normalization utilities for deterministic preprocessing."""

import re
from typing import Dict, Set
import numpy as np


def normalize_name(name: str) -> str:
    """
    Normalize table/column name deterministically.
    
    This implements the deterministic preprocessing pipeline from the evaluation framework:
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Replace sequences of non-alphanumeric characters with a single underscore
    4. Collapse multiple underscores to one
    
    Args:
        name: Original name string
        
    Returns:
        Normalized name string
        
    Examples:
        >>> normalize_name("Product_Type")
        'product_type'
        >>> normalize_name("bw%2Fme")
        'bw_me'
        >>> normalize_name("  Customer Name  ")
        'customer_name'
        >>> normalize_name("table___name---here")
        'table_name_here'
    """
    if not name:
        return ""
    
    # 1. Lowercase
    normalized = name.lower()
    
    # 2. Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    # 3. Replace sequences of non-alphanumeric characters with a single underscore
    normalized = re.sub(r'[^a-z0-9]+', '_', normalized)
    
    # 4. Collapse multiple underscores to one
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores that might result
    normalized = normalized.strip('_')
    
    return normalized


def compute_name_embedding(name: str, n: int = 3) -> Set[str]:
    """
    Compute character n-gram embedding for a name.
    
    This is a simple embedding approach that can be used for name similarity.
    More sophisticated embeddings can be added later.
    
    Args:
        name: Name string
        n: N-gram size (default: 3 for trigrams)
        
    Returns:
        Set of character n-grams
        
    Examples:
        >>> compute_name_embedding("product")
        {'pro', 'rod', 'odu', 'duc', 'uct'}
    """
    normalized = normalize_name(name)
    
    if len(normalized) < n:
        return {normalized} if normalized else set()
    
    ngrams = {normalized[i:i+n] for i in range(len(normalized) - n + 1)}
    return ngrams


def normalize_schema_names(
    table_names: Set[str],
    column_names: Dict[str, Set[str]]
) -> Dict[str, Dict[str, str]]:
    """
    Normalize all table and column names in a schema.
    
    Args:
        table_names: Set of table names
        column_names: Dict mapping table_name -> set of column names
        
    Returns:
        Dict with:
        - 'tables': mapping original_table_name -> normalized_table_name
        - 'columns': mapping (table_name, col_name) -> normalized_col_name
    """
    result = {
        'tables': {},
        'columns': {}
    }
    
    # Normalize table names
    for table_name in table_names:
        result['tables'][table_name] = normalize_name(table_name)
    
    # Normalize column names
    for table_name, cols in column_names.items():
        for col_name in cols:
            key = (table_name, col_name)
            result['columns'][key] = normalize_name(col_name)
    
    return result


# ============================================================================
# Score Normalization (for backward compatibility)
# ============================================================================

def clip_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clip a score to valid range.
    
    Args:
        score: Score to clip
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 1.0)
        
    Returns:
        Clipped score
    """
    return float(max(min_val, min(max_val, score)))


def normalize_score(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    invert: bool = False,
) -> float:
    """
    Normalize a score to [0,1] range.
    
    Args:
        score: Score to normalize
        min_score: Minimum possible score
        max_score: Maximum possible score
        invert: If True, invert the score (1 - normalized)
        
    Returns:
        Normalized score in [0,1]
    """
    if max_score == min_score:
        normalized = 0.5  # Default to middle if range is zero
    else:
        normalized = (score - min_score) / (max_score - min_score)
        normalized = clip_score(normalized)
    
    if invert:
        normalized = 1.0 - normalized
    
    return float(normalized)
