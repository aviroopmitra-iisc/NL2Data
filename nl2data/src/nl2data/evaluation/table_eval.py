"""Data-level (per table) evaluation metrics."""

import numpy as np
from typing import Dict, Tuple, Optional
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for SciPy
try:
    from scipy.stats import ks_2samp, wasserstein_distance, chisquare, spearmanr, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    _MISSING_SCIPY = ImportError(
        "SciPy is not installed. Install with: pip install nl2data[eval]"
    )

# Lazy imports for scikit-learn
try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    _MISSING_SKLEARN = ImportError(
        "scikit-learn is not installed. Install with: pip install nl2data[eval]"
    )


def numeric_marginals(
    real: np.ndarray,
    synth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute marginal distribution metrics for numeric columns.

    Args:
        real: Real data array
        synth: Synthetic data array

    Returns:
        Dictionary with KS statistic and Wasserstein distance
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "numeric_marginals requires SciPy. Install with: pip install nl2data[eval]"
        )

    if len(real) == 0 or len(synth) == 0:
        return {"ks_statistic": 0.0, "wasserstein_distance": 0.0}

    # Kolmogorov-Smirnov test
    ks_result = ks_2samp(real, synth)
    ks_stat = float(ks_result.statistic)

    # Wasserstein distance (Earth Mover's Distance)
    w1_dist = float(wasserstein_distance(real, synth))

    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": float(ks_result.pvalue),
        "wasserstein_distance": w1_dist,
    }


def categorical_marginals(
    real: np.ndarray,
    synth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute marginal distribution metrics for categorical columns.

    Args:
        real: Real data array (categorical)
        synth: Synthetic data array (categorical)

    Returns:
        Dictionary with chi-square statistic and p-value
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "categorical_marginals requires SciPy. Install with: pip install nl2data[eval]"
        )

    if len(real) == 0 or len(synth) == 0:
        return {"chi2_statistic": 0.0, "chi2_pvalue": 1.0}

    # Get unique values from both
    all_values = np.unique(np.concatenate([real, synth]))
    if len(all_values) == 0:
        return {"chi2_statistic": 0.0, "chi2_pvalue": 1.0}

    # Count frequencies
    real_counts = np.array([np.sum(real == val) for val in all_values])
    synth_counts = np.array([np.sum(synth == val) for val in all_values])

    # Normalize to expected frequencies (use real as expected)
    total_real = real_counts.sum()
    total_synth = synth_counts.sum()
    if total_real == 0 or total_synth == 0:
        return {"chi2_statistic": 0.0, "chi2_pvalue": 1.0}

    # Expected frequencies based on real distribution
    expected = (real_counts / total_real) * total_synth

    # Chi-square test
    chi2_result = chisquare(synth_counts, f_exp=expected)
    chi2_stat = float(chi2_result.statistic)
    chi2_pvalue = float(chi2_result.pvalue)

    return {
        "chi2_statistic": chi2_stat,
        "chi2_pvalue": chi2_pvalue,
    }


def correlation_metrics(
    real_col1: np.ndarray,
    real_col2: np.ndarray,
    synth_col1: np.ndarray,
    synth_col2: np.ndarray,
) -> Dict[str, float]:
    """
    Compute correlation metrics between two columns.

    Args:
        real_col1: First real column
        real_col2: Second real column
        synth_col1: First synthetic column
        synth_col2: Second synthetic column

    Returns:
        Dictionary with Pearson and Spearman correlation deltas
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "correlation_metrics requires SciPy. Install with: pip install nl2data[eval]"
        )

    if (
        len(real_col1) == 0
        or len(real_col2) == 0
        or len(synth_col1) == 0
        or len(synth_col2) == 0
    ):
        return {"pearson_delta": 0.0, "spearman_delta": 0.0}

    # Pearson correlation
    real_pearson, _ = pearsonr(real_col1, real_col2)
    synth_pearson, _ = pearsonr(synth_col1, synth_col2)
    pearson_delta = abs(float(real_pearson - synth_pearson))

    # Spearman correlation
    real_spearman, _ = spearmanr(real_col1, real_col2)
    synth_spearman, _ = spearmanr(synth_col1, synth_col2)
    spearman_delta = abs(float(real_spearman - synth_spearman))

    return {
        "pearson_delta": pearson_delta,
        "spearman_delta": spearman_delta,
    }


def mutual_information(
    cat_col1: np.ndarray,
    cat_col2: np.ndarray,
) -> float:
    """
    Compute mutual information between two categorical columns.

    Args:
        cat_col1: First categorical column
        cat_col2: Second categorical column

    Returns:
        Mutual information value
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(
            "mutual_information requires scikit-learn. Install with: pip install nl2data[eval]"
        )

    if len(cat_col1) == 0 or len(cat_col2) == 0:
        return 0.0

    # Encode categorical values to integers
    le1 = LabelEncoder()
    le2 = LabelEncoder()

    encoded1 = le1.fit_transform(cat_col1.ravel())
    encoded2 = le2.fit_transform(cat_col2.ravel())

    # Compute mutual information
    # Reshape for sklearn
    X = encoded1.reshape(-1, 1)
    y = encoded2

    mi = mutual_info_classif(X, y, random_state=42, discrete_features=True)
    return float(mi[0])


def table_fidelity_score(
    marginal_scores: Dict[str, float],
    pairwise_scores: Dict[str, float],
    marginal_weight: float = 0.7,
) -> float:
    """
    Compute aggregate table fidelity score.

    Args:
        marginal_scores: Dictionary of marginal metric scores (normalized 0-1)
        pairwise_scores: Dictionary of pairwise metric scores (normalized 0-1)
        marginal_weight: Weight for marginal scores (default: 0.7)

    Returns:
        Fidelity score (0-1, higher is better)
    """
    # Normalize scores (assuming lower is better for distances)
    # For now, simple average - in practice, you'd normalize each metric
    marginal_avg = np.mean(list(marginal_scores.values())) if marginal_scores else 0.0
    pairwise_avg = np.mean(list(pairwise_scores.values())) if pairwise_scores else 0.0

    fidelity = marginal_weight * marginal_avg + (1 - marginal_weight) * pairwise_avg
    return float(fidelity)

