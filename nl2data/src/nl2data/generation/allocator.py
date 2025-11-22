"""Memory-safe FK allocation with guaranteed coverage and Zipf skew."""

import numpy as np
from typing import Iterator, Tuple
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def zipf_probs(K: int, alpha: float) -> np.ndarray:
    """
    Compute normalized Zipf probabilities for K items.

    Args:
        K: Number of items
        alpha: Zipf exponent (higher = more skew)

    Returns:
        Normalized probability array of length K
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    ranks = np.arange(1, K + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    probs = weights / weights.sum()
    return probs


def fk_assignments(
    pk_ids: np.ndarray,
    n_rows: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    batch: int = 5_000_000,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generate FK assignments with guaranteed coverage and target skew.

    Guarantees that each PK gets at least one child (coverage), then allocates
    remaining rows by Zipf probabilities. Streams (pk_id, count) pairs without
    materializing a giant fk_pool.

    Args:
        pk_ids: Sorted array of parent PK values
        n_rows: Total number of fact rows to generate
        probs: Zipf probabilities (from zipf_probs())
        rng: Random number generator
        batch: Batch size hint for streaming (not used directly, but influences chunking)

    Yields:
        Tuples of (pk_id, count) where count is the number of fact rows for this PK
    """
    K = len(pk_ids)
    if K == 0:
        return

    if n_rows < K:
        raise ValueError(
            f"n_rows ({n_rows}) must be >= number of PKs ({K}) "
            f"to guarantee coverage"
        )

    # Guarantee coverage: each PK gets at least 1
    base = np.ones(K, dtype=np.int64)
    leftover = n_rows - K

    if leftover < 0:
        raise ValueError(f"n_rows ({n_rows}) < number of PKs ({K})")

    # Allocate remaining rows by Zipf probabilities
    if leftover > 0:
        alloc = rng.multinomial(leftover, probs, size=1).ravel()
    else:
        alloc = np.zeros(K, dtype=np.int64)

    counts = base + alloc

    # Stream out (pk_id, count) pairs
    # Process in chunks to avoid memory issues with very large K
    chunk_size = max(1, min(batch // max(1, n_rows // K + 1), K))
    if chunk_size == 0:
        chunk_size = K

    for start in range(0, K, chunk_size):
        end = min(K, start + chunk_size)
        for i in range(start, end):
            c = counts[i]
            if c > 0:
                yield pk_ids[i], int(c)


