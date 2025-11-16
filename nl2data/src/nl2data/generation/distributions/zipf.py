"""Zipf distribution sampler."""

import numpy as np
from typing import Optional
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class ZipfSampler(BaseSampler):
    """Zipf distribution sampler for generating skewed distributions."""

    def __init__(self, s: float = 1.2, n: Optional[int] = None):
        """
        Initialize Zipf sampler.

        Args:
            s: Zipf exponent (higher = more skewed)
            n: Domain size (number of distinct values)
        """
        self.s = s
        self.n = n
        logger.debug(f"Initialized ZipfSampler: s={s}, n={n}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """
        Generate n samples from Zipf distribution.

        Args:
            n: Number of samples to generate
            **kwargs:
                - rng: Random number generator
                - support: Array of values to sample from (for FK sampling)
                - n_items: Domain size if not set in __init__

        Returns:
            Array of sampled values
        """
        rng = kwargs.get("rng", np.random.default_rng())
        support = kwargs.get("support", None)
        n_items = self.n or kwargs.get("n_items", None)

        if support is not None:
            # Sample from provided support (e.g., FK values)
            n_items = len(support)
            # Generate Zipf probabilities
            ranks = np.arange(1, n_items + 1)
            probs = 1.0 / (ranks ** self.s)
            probs = probs / probs.sum()

            # Sample indices
            indices = rng.choice(n_items, size=n, p=probs)
            return support[indices]
        elif n_items is not None:
            # Generate integer IDs following Zipf distribution
            ranks = np.arange(1, n_items + 1)
            probs = 1.0 / (ranks ** self.s)
            probs = probs / probs.sum()
            indices = rng.choice(n_items, size=n, p=probs)
            return indices + 1  # 1-indexed IDs
        else:
            raise ValueError(
                "ZipfSampler requires either 'support' or 'n_items' in kwargs"
            )

