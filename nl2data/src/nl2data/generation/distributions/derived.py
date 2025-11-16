"""Derived column sampler (DEPRECATED - use derived expression engine instead)."""

import numpy as np
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class DerivedSampler(BaseSampler):
    """
    Derived column sampler (DEPRECATED).

    This class is deprecated. DistDerived is now handled by the derived
    expression engine (derived_program.py, derived_eval.py) which computes
    derived columns after base columns are generated.

    This class exists only for backwards compatibility and should not be
    instantiated.
    """

    def __init__(self, expression: str):
        """
        DEPRECATED: DerivedSampler should not be used.

        Args:
            expression: Expression string (ignored)
        """
        raise RuntimeError(
            "DerivedSampler is deprecated. DistDerived is handled by "
            "the derived expression engine (derived_program.py, derived_eval.py), "
            "not via samplers. Derived columns are computed after base columns "
            "are generated using vectorized DataFrame operations."
        )

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """
        DEPRECATED: This method should not be called.
        """
        raise RuntimeError("DerivedSampler.sample() should not be called")

