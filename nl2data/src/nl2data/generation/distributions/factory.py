"""Factory for creating distribution samplers."""

from typing import Any, Dict
import numpy as np
from nl2data.ir.generation import (
    Distribution,
    DistUniform,
    DistNormal,
    DistZipf,
    DistSeasonal,
    DistCategorical,
    DistDerived,
)
from .numeric import UniformSampler, NormalSampler
from .categorical import CategoricalSampler
from .zipf import ZipfSampler
from .seasonal import SeasonalDateSampler
from .derived import DerivedSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def get_sampler(dist: Distribution, **ctx: Any) -> Any:
    """
    Create a sampler instance for a distribution specification.

    Args:
        dist: Distribution specification from IR
        **ctx: Context parameters (rng, support, n_items, etc.)

    Returns:
        Sampler instance

    Raises:
        NotImplementedError: If distribution type is not supported
    """
    if isinstance(dist, DistUniform):
        return UniformSampler(dist.low, dist.high)

    if isinstance(dist, DistNormal):
        return NormalSampler(dist.mean, dist.std)

    if isinstance(dist, DistZipf):
        n = dist.n or ctx.get("n_items")
        sampler = ZipfSampler(dist.s, n)
        return sampler

    if isinstance(dist, DistSeasonal):
        return SeasonalDateSampler(dist.weights, dist.granularity)

    if isinstance(dist, DistCategorical):
        return CategoricalSampler(dist.domain.values, dist.domain.probs)

    if isinstance(dist, DistDerived):
        # In the new design, DistDerived is handled at the DataFrame-level
        # after base columns are generated. It should not be used via
        # get_sampler. Raise to catch misuse.
        raise RuntimeError(
            "DistDerived should not be passed to get_sampler(); "
            "derived columns are computed by the derived expression engine."
        )

    raise NotImplementedError(f"Distribution type {type(dist)} not supported")

