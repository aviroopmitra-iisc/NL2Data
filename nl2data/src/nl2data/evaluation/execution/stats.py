"""Statistical utility functions for evaluation."""

from typing import Tuple, Optional
import numpy as np
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for SciPy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    _MISSING_SCIPY = ImportError(
        "SciPy is not installed. Install with: pip install nl2data[eval]"
    )


def zipf_fit(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Zipf distribution to values and return R² and exponent.
    
    Args:
        values: Array of values to fit
        
    Returns:
        Tuple of (R², exponent s)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "zipf_fit requires SciPy. Install with: pip install nl2data[eval]"
        )
    
    if len(values) == 0:
        return 0.0, 0.0
    
    # Count frequencies
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) < 2:
        return 0.0, 0.0
    
    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    frequencies = counts[sorted_indices]
    ranks = np.arange(1, len(frequencies) + 1)
    
    # Fit log-log linear regression: log(freq) ~ log(rank)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    
    # Remove zeros/infs
    valid = np.isfinite(log_ranks) & np.isfinite(log_freqs)
    if np.sum(valid) < 2:
        return 0.0, 0.0
    
    # Use scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks[valid], log_freqs[valid]
    )
    s = -slope  # Zipf exponent (negate because slope is negative)
    
    return float(r_value ** 2), float(s)  # R² is correlation coefficient squared


def chi_square_test(
    observed: np.ndarray, expected: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Perform chi-square goodness-of-fit test.
    
    If expected is None, tests against uniform distribution.
    
    Args:
        observed: Observed frequencies
        expected: Expected frequencies (None for uniform)
        
    Returns:
        Tuple of (chi² statistic, p-value)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "chi_square_test requires SciPy. Install with: pip install nl2data[eval]"
        )
    
    if expected is None:
        expected = np.full_like(observed, observed.sum() / len(observed), dtype=float)
    
    # Remove zeros
    mask = (observed > 0) | (expected > 0)
    obs = observed[mask].astype(float)
    exp = expected[mask].astype(float)
    
    if len(obs) == 0:
        return 0.0, 1.0
    
    chi2, p_value = stats.chisquare(obs, exp)
    return float(chi2), float(p_value)


def ks_test(values: np.ndarray, dist_name: str = "norm", **dist_params) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test against a distribution.
    
    Args:
        values: Array of values to test
        dist_name: Distribution name (default: "norm")
        **dist_params: Distribution parameters
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "ks_test requires SciPy. Install with: pip install nl2data[eval]"
        )
    
    if len(values) == 0:
        return 0.0, 1.0
    
    # Get distribution from scipy.stats
    dist = getattr(stats, dist_name, None)
    if dist is None:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
    # Fit distribution if params not provided
    if not dist_params:
        dist_params = dist.fit(values)
    
    # Convert tuple params to dict if needed, or use as positional args
    if isinstance(dist_params, tuple):
        # Use positional arguments
        ks_stat, p_value = stats.kstest(values, lambda x: dist.cdf(x, *dist_params))
    else:
        # Use keyword arguments
        ks_stat, p_value = stats.kstest(values, lambda x: dist.cdf(x, **dist_params))
    
    return float(ks_stat), float(p_value)


def wasserstein_distance_metric(
    values1: np.ndarray, values2: np.ndarray
) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two distributions.
    
    Args:
        values1: First distribution
        values2: Second distribution
        
    Returns:
        Wasserstein distance
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "wasserstein_distance_metric requires SciPy. Install with: pip install nl2data[eval]"
        )
    
    if len(values1) == 0 or len(values2) == 0:
        return 0.0
    
    try:
        from scipy.stats import wasserstein_distance
        w1 = wasserstein_distance(values1, values2)
        return float(w1)
    except ImportError:
        # Fallback: manual computation (simplified)
        sorted1 = np.sort(values1)
        sorted2 = np.sort(values2)
        # Approximate with sorted values
        min_len = min(len(sorted1), len(sorted2))
        return float(np.mean(np.abs(sorted1[:min_len] - sorted2[:min_len])))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity [0,1]
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    if len(vec1) != len(vec2):
        # Pad shorter vector with zeros
        max_len = max(len(vec1), len(vec2))
        v1 = np.pad(vec1, (0, max_len - len(vec1)), mode="constant")
        v2 = np.pad(vec2, (0, max_len - len(vec2)), mode="constant")
    else:
        v1, v2 = vec1, vec2
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(max(0.0, min(1.0, similarity)))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient for measuring inequality.
    
    Args:
        values: Array of values
        
    Returns:
        Gini coefficient (0 to 1, higher = more unequal)
    """
    if len(values) == 0:
        return 0.0
    
    # Remove zeros and negatives
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    
    numerator = 2 * np.sum(index * sorted_values)
    denominator = n * np.sum(sorted_values)
    
    if denominator == 0:
        return 0.0
    
    gini = (numerator / denominator) - (n + 1) / n
    return float(max(0.0, min(1.0, gini)))


def top_k_share(values: np.ndarray, k: int = 1) -> float:
    """
    Calculate top-k share (concentration measure).
    
    Args:
        values: Array of values
        k: Number of top elements to consider
        
    Returns:
        Fraction of total value held by top k elements
    """
    if len(values) == 0:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0
    
    sorted_values = np.sort(values)[::-1]  # Descending
    top_k_sum = np.sum(sorted_values[:k])
    
    return float(top_k_sum / total)
