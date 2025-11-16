"""Numeric distribution samplers."""

import numpy as np
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class UniformSampler(BaseSampler):
    """Uniform distribution sampler."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        """
        Initialize uniform sampler.

        Args:
            low: Lower bound
            high: Upper bound
        """
        self.low = low
        self.high = high
        logger.debug(f"Initialized UniformSampler: [{low}, {high})")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n uniform samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.uniform(self.low, self.high, size=n)


class NormalSampler(BaseSampler):
    """Normal (Gaussian) distribution sampler."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize normal sampler.

        Args:
            mean: Mean of the distribution
            std: Standard deviation
        """
        self.mean = mean
        self.std = std
        logger.debug(f"Initialized NormalSampler: μ={mean}, σ={std}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n normal samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.normal(self.mean, self.std, size=n)

