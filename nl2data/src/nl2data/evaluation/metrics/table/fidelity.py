"""Table fidelity scoring."""

from typing import Dict
import numpy as np


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
    return float(max(0.0, min(1.0, fidelity)))
