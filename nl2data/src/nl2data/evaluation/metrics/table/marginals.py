"""Marginal distribution metrics."""

from typing import Dict
import numpy as np
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for SciPy
try:
    from scipy.stats import ks_2samp, wasserstein_distance, chisquare
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    _MISSING_SCIPY = ImportError(
        "SciPy is not installed. Install with: pip install nl2data[eval]"
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
        Dictionary with KS statistic, p-value, and Wasserstein distance
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "numeric_marginals requires SciPy. Install with: pip install nl2data[eval]"
        )
    
    if len(real) == 0 or len(synth) == 0:
        return {
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "wasserstein_distance": 0.0,
        }
    
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
