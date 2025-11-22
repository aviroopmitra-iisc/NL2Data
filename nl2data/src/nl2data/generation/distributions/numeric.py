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


class LognormalSampler(BaseSampler):
    """Log-normal distribution sampler."""

    def __init__(self, mean: float, sigma: float):
        """
        Initialize log-normal sampler.

        Args:
            mean: Mean of the underlying normal distribution
            sigma: Standard deviation of the underlying normal distribution (must be > 0)
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mean = mean
        self.sigma = sigma
        logger.debug(f"Initialized LognormalSampler: μ={mean}, σ={sigma}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n log-normal samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        # numpy's lognormal uses mean and sigma of the underlying normal distribution
        return rng.lognormal(self.mean, self.sigma, size=n)


class ParetoSampler(BaseSampler):
    """Pareto distribution sampler."""

    def __init__(self, alpha: float, xm: float = 1.0):
        """
        Initialize Pareto sampler.

        Args:
            alpha: Shape parameter (must be > 0)
            xm: Scale parameter / minimum value (must be > 0)
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if xm <= 0:
            raise ValueError("xm must be positive")
        self.alpha = alpha
        self.xm = xm
        logger.debug(f"Initialized ParetoSampler: α={alpha}, xm={xm}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n Pareto samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        # numpy's pareto uses shape parameter a, where a = alpha
        # The distribution is xm * (1 - U)^(-1/alpha) where U is uniform(0,1)
        # numpy.pareto returns values >= 1, so we scale by xm
        return self.xm * (1 - rng.uniform(0, 1, size=n)) ** (-1.0 / self.alpha)


class PoissonSampler(BaseSampler):
    """Poisson distribution sampler."""

    def __init__(self, lam: float):
        """
        Initialize Poisson sampler.

        Args:
            lam: Lambda parameter (rate, must be > 0)
        """
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        logger.debug(f"Initialized PoissonSampler: λ={lam}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n Poisson samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.poisson(self.lam, size=n)


class ExponentialSampler(BaseSampler):
    """Exponential distribution sampler."""

    def __init__(self, scale: float):
        """
        Initialize Exponential sampler.

        Args:
            scale: Scale parameter (1/lambda, must be > 0)
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        logger.debug(f"Initialized ExponentialSampler: scale={scale}")

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n exponential samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.exponential(self.scale, size=n)