"""Base sampler interface."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseSampler(ABC):
    """Base class for all distribution samplers."""

    @abstractmethod
    def sample(self, n: int, **kwargs: Any) -> np.ndarray:
        """
        Generate n samples from the distribution.

        Args:
            n: Number of samples to generate
            **kwargs: Additional parameters specific to sampler

        Returns:
            Array of samples
        """
        pass

