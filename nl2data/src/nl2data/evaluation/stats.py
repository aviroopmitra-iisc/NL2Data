"""Statistical evaluation functions."""

import numpy as np
from typing import List, Tuple, Optional
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for SciPy
try:
    from scipy import stats
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    wasserstein_distance = None
    _MISSING_SCIPY = ImportError(
        "SciPy is not installed. Install with: pip install nl2data[eval]"
    )


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
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (
        n + 1
    ) / n


def zipf_fit(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Zipf distribution to values and return R² and exponent.

    Args:
        values: Array of integer values (e.g., IDs)

    Returns:
        Tuple of (R², exponent s)
    """
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

    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, cannot compute Zipf fit")
        return 0.0, 0.0

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks[valid], log_freqs[valid]
    )
    s = -slope  # Zipf exponent

    return r_value ** 2, s


def chi_square_test(
    observed: np.ndarray, expected: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Perform chi-square goodness-of-fit test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (if None, assumes uniform)

    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    if expected is None:
        expected = np.full_like(observed, observed.sum() / len(observed))

    # Remove zeros
    mask = (observed > 0) | (expected > 0)
    obs = observed[mask]
    exp = expected[mask]

    if len(obs) == 0:
        return 0.0, 1.0

    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, cannot compute chi-square test")
        return 0.0, 1.0

    chi2, p_value = stats.chisquare(obs, exp)
    return float(chi2), float(p_value)


def ks_test(
    sample1: np.ndarray, sample2: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test.

    Args:
        sample1: First sample
        sample2: Second sample (if None, tests against uniform)

    Returns:
        Tuple of (KS statistic, p-value)
    """
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, cannot compute KS test")
        return 0.0, 1.0

    if sample2 is None:
        # Test against uniform distribution
        sample2 = np.linspace(sample1.min(), sample1.max(), len(sample1))

    statistic, p_value = stats.ks_2samp(sample1, sample2)
    return float(statistic), float(p_value)


def wasserstein_distance_metric(
    sample1: np.ndarray, sample2: np.ndarray
) -> float:
    """
    Calculate Wasserstein distance between two samples.

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        Wasserstein distance
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "wasserstein_distance_metric requires SciPy. Install with: pip install nl2data[eval]"
        )
    return float(wasserstein_distance(sample1, sample2))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1, typically 0 to 1 for positive vectors)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def top_k_share(values: np.ndarray, k: int = 1) -> float:
    """
    Calculate the share of top-k values.

    Args:
        values: Array of values (e.g., group counts)
        k: Number of top values to consider

    Returns:
        Share of top-k values (0 to 1)
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)[::-1]
    top_k_sum = np.sum(sorted_values[:k])
    total_sum = np.sum(sorted_values)

    if total_sum == 0:
        return 0.0

    return float(top_k_sum / total_sum)

