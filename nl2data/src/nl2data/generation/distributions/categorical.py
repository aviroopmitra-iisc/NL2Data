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
            probs: Optional probability distribution (must sum to 1)
        """
        self.values = values
        if probs is not None:
            if abs(sum(probs) - 1.0) > 1e-6:
                raise ValueError("Probabilities must sum to 1.0")
            if len(probs) != len(values):
                raise ValueError("Probabilities length must match values length")
            self.probs = probs
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

