"""Column pair correlation metrics."""

from typing import Dict
import numpy as np
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for SciPy
try:
    from scipy.stats import spearmanr, pearsonr
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
    try:
        real_pearson, _ = pearsonr(real_col1, real_col2)
        synth_pearson, _ = pearsonr(synth_col1, synth_col2)
        pearson_delta = abs(float(real_pearson - synth_pearson))
    except Exception:
        pearson_delta = 0.0
    
    # Spearman correlation
    try:
        real_spearman, _ = spearmanr(real_col1, real_col2)
        synth_spearman, _ = spearmanr(synth_col1, synth_col2)
        spearman_delta = abs(float(real_spearman - synth_spearman))
    except Exception:
        spearman_delta = 0.0
    
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
