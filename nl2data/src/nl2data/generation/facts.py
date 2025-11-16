"""Utilities for generating fact tables with composite PKs."""

import pandas as pd
import numpy as np
from typing import Callable
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def spawn_children(
    parent_ids: pd.Series,
    child_count_dist: Callable[[int], np.ndarray],
    parent_fk_name: str,
    child_seq_name: str = "line_no",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate child rows with composite PK (parent_fk, sequence).

    This is useful for generating order_items from orders, where each order
    has multiple line items with a composite PK (order_id, line_no).

    Args:
        parent_ids: Series of parent PKs (e.g., orders.order_id)
        child_count_dist: Function that takes n_parents and returns array of
                         child counts per parent (e.g., Poisson distribution)
        parent_fk_name: Column name for parent FK in child table
        child_seq_name: Column name for sequence number (default: "line_no")
        rng: Optional random number generator (for deterministic results)

    Returns:
        DataFrame with columns [parent_fk_name, child_seq_name]
    """
    n = len(parent_ids)
    if n == 0:
        return pd.DataFrame(columns=[parent_fk_name, child_seq_name])

    # Generate child counts per parent
    counts = child_count_dist(n)
    if rng is not None:
        # If rng is provided, use it for deterministic results
        # (assuming child_count_dist uses rng internally)
        pass

    # Build rows: for each parent, create sequence 1..count
    rows = []
    for pid, count in zip(parent_ids, counts):
        count_int = int(count)
        if count_int > 0:
            for seq in range(1, count_int + 1):
                rows.append({parent_fk_name: pid, child_seq_name: seq})

    if not rows:
        return pd.DataFrame(columns=[parent_fk_name, child_seq_name])

    result = pd.DataFrame(rows)
    logger.debug(
        f"Generated {len(result)} child rows from {n} parents "
        f"(avg {len(result)/n:.2f} children per parent)"
    )
    return result


def poisson_child_counts(
    n_parents: int,
    mean: float = 3.0,
    min_count: int = 1,
    max_count: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate child counts using Poisson distribution.

    This is a helper function for spawn_children that generates realistic
    child counts (e.g., number of line items per order).

    Args:
        n_parents: Number of parents
        mean: Mean of Poisson distribution (default: 3.0)
        min_count: Minimum child count (default: 1)
        max_count: Maximum child count (None = no limit)
        rng: Random number generator

    Returns:
        Array of child counts (one per parent)
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = rng.poisson(mean, size=n_parents)
    counts = np.maximum(counts, min_count)

    if max_count is not None:
        counts = np.minimum(counts, max_count)

    return counts


def uniform_child_counts(
    n_parents: int,
    min_count: int = 1,
    max_count: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate child counts using uniform distribution.

    Args:
        n_parents: Number of parents
        min_count: Minimum child count
        max_count: Maximum child count
        rng: Random number generator

    Returns:
        Array of child counts (one per parent)
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = rng.integers(min_count, max_count + 1, size=n_parents)
    return counts

