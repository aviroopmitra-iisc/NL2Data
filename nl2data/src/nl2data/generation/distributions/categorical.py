"""Categorical distribution sampler."""

import numpy as np
from typing import List, Optional
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class CategoricalSampler(BaseSampler):
    """Categorical distribution sampler."""

    def __init__(self, values: List[str], probs: Optional[List[float]] = None):
        """
        Initialize categorical sampler.

        Args:
            values: List of categorical values
            probs: Optional probability distribution (will be normalized to sum to 1)
        """
        self.values = values
        if probs is not None:
            if len(probs) != len(values):
                raise ValueError("Probabilities length must match values length")
            # Normalize probabilities to sum to 1.0 (handles floating point precision issues)
            total = sum(probs)
            if total <= 0:
                raise ValueError("Probabilities must be positive")
            self.probs = [p / total for p in probs]
        else:
            # Uniform distribution
            self.probs = [1.0 / len(values)] * len(values)

        logger.debug(
            f"Initialized CategoricalSampler with {len(values)} categories"
        )

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n categorical samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        indices = rng.choice(len(self.values), size=n, p=self.probs)
        return np.array([self.values[i] for i in indices])

