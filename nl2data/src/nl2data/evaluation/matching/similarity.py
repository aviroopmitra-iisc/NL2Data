"""Similarity computation utilities for schema matching.

This module implements all similarity functions needed for the enhanced evaluation framework.
"""

from typing import Dict, Any, Set, Optional
import difflib
import numpy as np
import pandas as pd
from nl2data.ir.logical import ColumnSpec, TableSpec

# Try to import optional dependencies for name compatibility
try:
    from nltk.corpus import wordnet as wn
    from nltk import download as nltk_download
    WORDNET_AVAILABLE = True
    try:
        wn.synsets('test')
    except LookupError:
        try:
            nltk_download('wordnet', quiet=True)
            nltk_download('omw-1.4', quiet=True)
        except:
            WORDNET_AVAILABLE = False
except ImportError:
    WORDNET_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Global semantic model (lazy loaded)
_semantic_model = None


# ============================================================================
# Name Compatibility (RSchema-style)
# ============================================================================

def get_semantic_model():
    """Lazy load semantic similarity model."""
    global _semantic_model
    if _semantic_model is None and SEMANTIC_AVAILABLE:
        try:
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            return None
    return _semantic_model


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    return name.lower().strip().replace("_", "").replace(" ", "")


def get_wordnet_synonyms(word: str) -> Set[str]:
    """Get WordNet synonyms for a word."""
    if not WORDNET_AVAILABLE:
        return set()
    
    synonyms = set()
    word_lower = word.lower()
    
    try:
        # Get synsets for the word
        for syn in wn.synsets(word_lower):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ').lower()
                synonyms.add(lemma_name)
                synonyms.add(lemma_name.replace(' ', ''))
    except:
        pass
    
    # Also add the original word
    synonyms.add(word_lower)
    synonyms.add(word_lower.replace(' ', ''))
    
    return synonyms


def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using sentence-transformers."""
    model = get_semantic_model()
    if model is None:
        return 0.0
    
    try:
        embeddings = model.encode([text1, text2])
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(similarity)
    except Exception:
        return 0.0


def longest_common_substring(s1: str, s2: str) -> int:
    """Calculate length of longest common substring."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    
    m, n = len(s1_lower), len(s2_lower)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1_lower[i-1] == s2_lower[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_len


def lcs_ratio(s1: str, s2: str, threshold: float = 0.75) -> bool:
    """Check if LCS ratio exceeds threshold."""
    lcs_len = longest_common_substring(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return False
    ratio = lcs_len / max_len
    return ratio >= threshold


def name_compatible(
    name1: str,
    name2: str,
    use_wordnet: bool = True,
    semantic_threshold: float = 0.6,
    lcs_threshold: float = 0.75
) -> bool:
    """
    Check if two names are compatible using RSchema-style matching.
    
    Uses three methods in order:
    1. WordNet synonym matching
    2. Semantic similarity (if >= threshold)
    3. LCS matching (if >= threshold)
    
    Args:
        name1: First name
        name2: Second name
        use_wordnet: Whether to use WordNet (default: True)
        semantic_threshold: Threshold for semantic similarity (default: 0.6)
        lcs_threshold: Threshold for LCS ratio (default: 0.75)
        
    Returns:
        True if names are compatible, False otherwise
    """
    # Exact match (after normalization)
    if normalize_name(name1) == normalize_name(name2):
        return True
    
    # Step 1: WordNet synonym matching
    if use_wordnet and WORDNET_AVAILABLE:
        name1_synonyms = get_wordnet_synonyms(name1)
        name2_normalized = normalize_name(name2)
        name1_normalized = normalize_name(name1)
        
        if name2_normalized in name1_synonyms or name1_normalized == name2_normalized:
            return True
    
    # Step 2: Semantic similarity
    if SEMANTIC_AVAILABLE:
        sem_sim = semantic_similarity(name1, name2)
        if sem_sim >= semantic_threshold:
            return True
    
    # Step 3: LCS matching
    if lcs_ratio(name1, name2, lcs_threshold):
        return True
    
    return False


# ============================================================================
# Name Similarity
# ============================================================================

def name_similarity(
    name1: str,
    name2: str,
    use_rschema_compatibility: bool = True,
    semantic_threshold: float = 0.6,
    lcs_threshold: float = 0.75
) -> float:
    """
    Compute name similarity between two strings.
    
    First checks RSchema-style name compatibility. If compatible, returns 1.0.
    Otherwise, uses multiple methods and combines them:
    - Tokenized Jaccard similarity
    - Edit distance (normalized)
    - Sequence matching ratio
    
    Args:
        name1: First name
        name2: Second name
        use_rschema_compatibility: Whether to use RSchema compatibility check first (default: True)
        semantic_threshold: Threshold for semantic similarity in compatibility check (default: 0.6)
        lcs_threshold: Threshold for LCS ratio in compatibility check (default: 0.75)
        
    Returns:
        Similarity score [0,1] where 1.0 is identical or compatible
    """
    # First check: RSchema-style name compatibility
    if use_rschema_compatibility:
        if name_compatible(name1, name2, semantic_threshold=semantic_threshold, lcs_threshold=lcs_threshold):
            return 1.0
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    name1_lower = name1.lower()
    name2_lower = name2.lower()
    
    # Tokenized Jaccard similarity
    tokens1 = set(name1_lower.replace("_", " ").replace("-", " ").split())
    tokens2 = set(name2_lower.replace("_", " ").replace("-", " ").split())
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Edit distance (normalized)
    max_len = max(len(name1_lower), len(name2_lower))
    if max_len == 0:
        return 1.0
    
    edit_ratio = difflib.SequenceMatcher(None, name1_lower, name2_lower).ratio()
    
    # Combine: weighted average
    similarity = 0.4 * jaccard + 0.6 * edit_ratio
    
    return float(similarity)


# ============================================================================
# Type Compatibility
# ============================================================================

def hard_incompatible_datatype(type1: str, type2: str) -> bool:
    """
    Determine if two SQL types are hard incompatible.
    
    Compatible pairs:
    - INT ↔ FLOAT (and vice versa)
    - DATE ↔ DATETIME (and vice versa)
    - Same type (e.g., INT ↔ INT)
    
    All other combinations are incompatible.
    
    Args:
        type1, type2: SQLType strings
        
    Returns:
        True if types should never be matched
    """
    # Exact match is always compatible
    if type1 == type2:
        return False
    
    # Define compatible type groups
    numeric_group = {"INT", "FLOAT", "INT32", "INT64", "FLOAT32", "FLOAT64", "NUMERIC"}
    datetime_group = {"DATE", "DATETIME", "TIMESTAMP"}
    
    # Check if both are in the same compatible group
    if type1 in numeric_group and type2 in numeric_group:
        return False  # INT and FLOAT are compatible
    
    if type1 in datetime_group and type2 in datetime_group:
        return False  # DATE and DATETIME are compatible
    
    # All other combinations are incompatible
    return True


def type_compatibility(type1: str, type2: str) -> float:
    """
    Compute type compatibility score between two SQL types.
    
    Returns:
        - 1.0 if exact match
        - 0.8 if compatible (INT↔FLOAT, DATE↔DATETIME)
        - 0.0 if incompatible
    """
    # Exact match
    if type1 == type2:
        return 1.0
    
    # Compatible groups
    numeric_group = {"INT", "FLOAT", "INT32", "INT64", "FLOAT32", "FLOAT64", "NUMERIC"}
    datetime_group = {"DATE", "DATETIME", "TIMESTAMP"}
    
    # Check compatible groups
    if type1 in numeric_group and type2 in numeric_group:
        return 0.8  # INT and FLOAT are compatible
    
    if type1 in datetime_group and type2 in datetime_group:
        return 0.8  # DATE and DATETIME are compatible
    
    # Incompatible
    return 0.0


def is_numeric_type(sql_type: str) -> bool:
    """Check if SQL type is numeric."""
    numeric_types = {"INT", "FLOAT", "INT32", "INT64", "FLOAT32", "FLOAT64", "NUMERIC"}
    return sql_type in numeric_types


def is_datetime_type(sql_type: str) -> bool:
    """Check if SQL type is datetime."""
    datetime_types = {"DATE", "DATETIME", "TIMESTAMP"}
    return sql_type in datetime_types


def is_categorical_type(sql_type: str) -> bool:
    """Check if SQL type is categorical."""
    categorical_types = {"TEXT", "VARCHAR", "STRING"}
    return sql_type in categorical_types


# ============================================================================
# Range Similarity
# ============================================================================

def range_sim_num(
    summary_i: Dict[str, Any],
    summary_j: Dict[str, Any]
) -> float:
    """
    Compute numeric range similarity between two column summaries.
    
    Args:
        summary_i, summary_j: Column summary dicts with "min" and "max" keys
        
    Returns:
        Similarity score [0,1]
    """
    # Input validation
    if not summary_i.get("is_numeric") or not summary_j.get("is_numeric"):
        return 0.0
    
    if "min" not in summary_i or "max" not in summary_i:
        return 0.0
    if "min" not in summary_j or "max" not in summary_j:
        return 0.0
    
    l_i = summary_i["min"]
    u_i = summary_i["max"]
    l_j = summary_j["min"]
    u_j = summary_j["max"]
    
    # Compute interval overlap
    overlap_len = max(0, min(u_i, u_j) - max(l_i, l_j))
    union_len = max(u_i, u_j) - min(l_i, l_j)
    
    # Define numeric range similarity
    if union_len <= 0:
        return 0.0
    
    similarity = overlap_len / union_len
    return float(max(0.0, min(1.0, similarity)))  # Clamp to [0,1]


def range_sim_cat(
    summary_i: Dict[str, Any],
    summary_j: Dict[str, Any],
    alpha_cat: float = 0.5
) -> float:
    """
    Compute categorical range similarity between two column summaries.
    
    Args:
        summary_i, summary_j: Column summary dicts with "dom_set" and "p_c" keys
        alpha_cat: Weight for set-overlap vs frequency-overlap
        
    Returns:
        Similarity score [0,1]
    """
    if not summary_i.get("is_categorical") or not summary_j.get("is_categorical"):
        return 0.0
    
    D_i = summary_i.get("dom_set", set())
    D_j = summary_j.get("dom_set", set())
    p_i = summary_i.get("p_c", {})
    p_j = summary_j.get("p_c", {})
    
    # Set-level overlap (Jaccard)
    D_union = D_i | D_j
    if not D_union:
        J_set = 0.0
    else:
        intersection_size = len(D_i & D_j)
        union_size = len(D_union)
        J_set = intersection_size / union_size if union_size > 0 else 0.0
    
    # Frequency-level overlap
    J_freq = 0.0
    for v in D_union:
        # Convert to string for dict lookup
        v_str = str(v)
        p_i_val = p_i.get(v_str, 0.0)
        p_j_val = p_j.get(v_str, 0.0)
        J_freq += min(p_i_val, p_j_val)
    
    # Combine set- and frequency-level overlap
    range_sim_cat_val = alpha_cat * J_set + (1 - alpha_cat) * J_freq
    return float(max(0.0, min(1.0, range_sim_cat_val)))  # Clamp to [0,1]


def range_sim(
    summary_i: Dict[str, Any],
    summary_j: Dict[str, Any],
    alpha_cat: float = 0.5
) -> float:
    """
    Unified range similarity function.
    
    Automatically selects numeric or categorical similarity based on column types.
    
    Args:
        summary_i, summary_j: Column summary dicts
        alpha_cat: Weight for categorical set-overlap vs frequency-overlap
        
    Returns:
        Similarity score [0,1]
    """
    is_numeric_i = summary_i.get("is_numeric", False)
    is_numeric_j = summary_j.get("is_numeric", False)
    is_categorical_i = summary_i.get("is_categorical", False)
    is_categorical_j = summary_j.get("is_categorical", False)
    
    if is_numeric_i and is_numeric_j:
        return range_sim_num(summary_i, summary_j)
    elif is_categorical_i and is_categorical_j:
        return range_sim_cat(summary_i, summary_j, alpha_cat)
    else:
        return 0.0


# ============================================================================
# Role Similarity
# ============================================================================

def get_column_role(column: ColumnSpec, table: TableSpec) -> str:
    """
    Determine column role: PK, FK, or NORMAL.
    
    Args:
        column: ColumnSpec instance
        table: TableSpec instance containing the column
        
    Returns:
        "PK", "FK", or "NORMAL"
    """
    # Check if primary key
    if column.name in table.primary_key:
        return "PK"
    
    # Check if foreign key (via role or foreign_keys list)
    if column.role == "foreign_key" or column.references is not None:
        return "FK"
    
    # Check if in foreign_keys list
    for fk in table.foreign_keys:
        if fk.column == column.name:
            return "FK"
    
    return "NORMAL"


def role_sim(r1: str, r2: str) -> float:
    """
    Compute role similarity between two role strings.
    
    Args:
        r1, r2: Role strings ("PK", "FK", or "NORMAL")
        
    Returns:
        Similarity score [0,1]
    """
    if r1 == "PK" and r2 == "PK":
        return 1.0
    elif r1 == "FK" and r2 == "FK":
        return 0.8
    elif r1 == "NORMAL" and r2 == "NORMAL":
        return 0.5
    else:
        return 0.0


# ============================================================================
# FD Participation Similarity
# ============================================================================

def fd_sim(
    lhs_i: int,
    rhs_i: int,
    lhs_j: int,
    rhs_j: int,
    lambda_fd: float = 0.5,
    mu_fd: float = 0.5
) -> float:
    """
    Compute FD participation similarity between two columns.
    
    Uses exponential decay based on differences in FD participation counts.
    
    Args:
        lhs_i: LHS count for column i
        rhs_i: RHS count for column i
        lhs_j: LHS count for column j
        rhs_j: RHS count for column j
        lambda_fd: Decay coefficient for LHS differences
        mu_fd: Decay coefficient for RHS differences
        
    Returns:
        Similarity score in (0,1]
    """
    # Compute exponential decay
    similarity = np.exp(
        -lambda_fd * abs(lhs_i - lhs_j) - mu_fd * abs(rhs_i - rhs_j)
    )
    
    return float(max(0.0, min(1.0, similarity)))


# ============================================================================
# Legacy/Compatibility Functions
# ============================================================================

def distribution_similarity(
    real_series: pd.Series,
    synth_series: pd.Series,
    is_numeric: bool = True,
) -> float:
    """
    Compute distribution similarity between two data series.
    
    This is a legacy function kept for backward compatibility.
    For new code, use range_sim() with column summaries instead.
    
    For numeric: range overlap, basic statistics
    For categorical: Jaccard of unique values, frequency overlap
    
    Args:
        real_series: Real data series
        synth_series: Synthetic data series
        is_numeric: Whether the data is numeric
        
    Returns:
        Similarity score [0,1]
    """
    if len(real_series) == 0 or len(synth_series) == 0:
        return 0.0
    
    if is_numeric:
        # Range overlap
        real_min, real_max = float(real_series.min()), float(real_series.max())
        synth_min, synth_max = float(synth_series.min()), float(synth_series.max())
        
        if real_max == real_min and synth_max == synth_min:
            # Both constant
            return 1.0 if real_min == synth_min else 0.0
        
        # Overlap ratio
        overlap_min = max(real_min, synth_min)
        overlap_max = min(real_max, synth_max)
        overlap = max(0, overlap_max - overlap_min)
        total_range = max(real_max, synth_max) - min(real_min, synth_min)
        
        range_similarity = overlap / total_range if total_range > 0 else 0.0
        
        # Basic statistics similarity (mean, std)
        real_mean, real_std = float(real_series.mean()), float(real_series.std())
        synth_mean, synth_std = float(synth_series.mean()), float(synth_series.std())
        
        mean_similarity = 1.0 - min(
            1.0, abs(real_mean - synth_mean) / (abs(real_mean) + 1e-6)
        )
        std_similarity = 1.0 - min(
            1.0, abs(real_std - synth_std) / (abs(real_std) + 1e-6)
        )
        
        # Combine
        similarity = 0.4 * range_similarity + 0.3 * mean_similarity + 0.3 * std_similarity
        return float(max(0.0, min(1.0, similarity)))
    
    else:
        # Categorical: Jaccard of unique values
        real_values = set(real_series.dropna().unique())
        synth_values = set(synth_series.dropna().unique())
        
        if not real_values and not synth_values:
            return 1.0
        if not real_values or not synth_values:
            return 0.0
        
        intersection = len(real_values & synth_values)
        union = len(real_values | synth_values)
        jaccard = intersection / union if union > 0 else 0.0
        
        return float(jaccard)
