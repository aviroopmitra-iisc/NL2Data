"""Fact table generator with streaming support."""

import numpy as np
import pandas as pd
from typing import Dict, Iterator
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.logical import TableSpec, ColumnSpec
from nl2data.ir.generation import DistZipf, DistDerived
from nl2data.generation.distributions import get_sampler
from nl2data.generation.derived_registry import DerivedRegistry
from nl2data.generation.derived_eval import eval_derived
from nl2data.generation.allocator import (
    zipf_probs,
    fk_assignments,
    clip_alpha_for_max_share,
)
from nl2data.generation.constants import (
    DEFAULT_FACT_ROWS,
    PROGRESS_LOG_INTERVAL_SECONDS,
    LARGE_TABLE_THRESHOLD,
)
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def sample_fact_column(
    table: TableSpec,
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
    dims: Dict[str, pd.DataFrame],
) -> np.ndarray:
    """
    Sample values for a fact table column.

    Handles foreign keys specially to maintain referential integrity.

    Args:
        table: Table specification
        col: Column specification
        dist: Distribution specification (or None)
        n: Number of samples
        rng: Random number generator
        dims: Dictionary of dimension DataFrames

    Returns:
        Array of sampled values
    """
    # Handle foreign keys with memory-safe allocation
    if col.role == "foreign_key" and col.references:
        ref_table, ref_col = col.references.split(".")
        if ref_table not in dims:
            raise ValueError(
                f"Referenced dimension table '{ref_table}' not found"
            )
        support = dims[ref_table][ref_col].to_numpy(copy=False)

        if isinstance(dist, DistZipf):
            # Use memory-safe FK allocation with guaranteed coverage
            # Sort PKs deterministically
            pk_ids = np.sort(support)
            K = len(pk_ids)

            # Compute Zipf probabilities
            alpha = dist.s
            # Clip alpha if max_top1_share is specified (future enhancement)
            # For now, use alpha directly
            probs = zipf_probs(K, alpha)

            # Generate FK assignments with guaranteed coverage
            fk_values = []
            for pk_id, count in fk_assignments(pk_ids, n, probs, rng):
                fk_values.extend([pk_id] * count)

            # Shuffle to avoid clustering
            result = np.array(fk_values, dtype=support.dtype)
            rng.shuffle(result)
            return result
        else:
            # Fallback: uniform sampling from FK values (old method)
            # For large dimensions, consider using fk_assignments with uniform probs
            idx = rng.integers(0, len(support), size=n)
            return support[idx]

    # Non-FK: reuse dimension sampling logic
    from .dim_generator import sample_column

    result = sample_column(table, col, dist, n, rng)
    # Type enforcement is already done in sample_column
    return result


def generate_fact_stream(
    table: TableSpec,
    ir: DatasetIR,
    dims: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    chunk_rows: int,
    derived_reg: DerivedRegistry,
) -> Iterator[pd.DataFrame]:
    """
    Generate fact table in chunks (streaming) with two-phase generation (base → derived).

    Args:
        table: Table specification
        ir: Dataset IR
        dims: Dictionary of dimension DataFrames
        rng: Random number generator
        chunk_rows: Number of rows per chunk
        derived_reg: Derived registry with compiled expressions

    Yields:
        DataFrame chunks (with derived columns computed)
    """
    n_total = table.row_count or DEFAULT_FACT_ROWS
    logger.info(
        f"Generating fact table '{table.name}' with {n_total} rows "
        f"(chunk size: {chunk_rows})"
    )

    # Build distribution map
    from nl2data.generation.ir_helpers import build_distribution_map
    gen_map = build_distribution_map(ir)

    # Get derived column order for this table
    derived_cols = derived_reg.order.get(table.name, [])

    produced = 0
    chunk_num = 0
    import time
    start_time = time.time()
    last_progress_time = start_time

    while produced < n_total:
        m = min(chunk_rows, n_total - produced)
        
        # Phase 1: Generate base columns (non-derived)
        base_block = {}
        for col in table.columns:
            dist = gen_map.get((table.name, col.name))
            if isinstance(dist, DistDerived):
                continue  # Skip derived columns in first phase
            base_block[col.name] = sample_fact_column(
                table, col, dist, m, rng, dims
            )

        df_chunk = pd.DataFrame(base_block)

        # Phase 1.5: Join dimension tables (for derived column lookups)
        # This allows derived expressions to reference dimension columns
        for fk in table.foreign_keys:
            ref_table_name = fk.ref_table
            if ref_table_name in dims:
                dim_df = dims[ref_table_name]
                # Join on foreign key
                # Use left join to preserve all fact rows
                df_chunk = df_chunk.merge(
                    dim_df,
                    how="left",
                    left_on=fk.column,
                    right_on=fk.ref_column,
                    suffixes=("", f"_{ref_table_name}")
                )
                logger.debug(
                    f"Joined dimension '{ref_table_name}' to fact table '{table.name}' "
                    f"on {fk.column} = {fk.ref_column}"
                )
            else:
                logger.warning(
                    f"Dimension table '{ref_table_name}' not found for join. "
                    f"Derived expressions referencing this dimension will fail."
                )

        # Phase 2: Compute derived columns in dependency order
        for col_name in derived_cols:
            key = (table.name, col_name)
            if key in derived_reg.programs:
                prog = derived_reg.programs[key]
                try:
                    df_chunk[col_name] = eval_derived(prog, df_chunk, rng=rng)
                except KeyError as e:
                    # Column not found - provide helpful error
                    missing_col = str(e).strip("'")
                    logger.error(
                        f"Failed to compute derived column '{col_name}' in table '{table.name}': "
                        f"Missing column '{missing_col}'. "
                        f"Available columns: {list(df_chunk.columns)}. "
                        f"Expression: {prog.expr}. "
                        f"Dependencies: {prog.dependencies}."
                    )
                    raise ValueError(
                        f"Derived column '{col_name}' in table '{table.name}' depends on "
                        f"column '{missing_col}' which is not available. "
                        f"Ensure the column exists in the table or is joined from a dimension."
                    ) from e
                except Exception as e:
                    logger.error(
                        f"Failed to compute derived column '{col_name}' in chunk "
                        f"{chunk_num + 1} of table '{table.name}': {e}. "
                        f"Expression: {prog.expr}. "
                        f"Dependencies: {prog.dependencies}."
                    )
                    raise

        produced += m
        chunk_num += 1
        
        # Progress tracking for large tables
        current_time = time.time()
        elapsed = current_time - start_time
        progress_pct = (produced / n_total * 100) if n_total > 0 else 0
        
        # Log progress periodically for large tables
        if n_total > LARGE_TABLE_THRESHOLD and (current_time - last_progress_time) >= PROGRESS_LOG_INTERVAL_SECONDS:
            rate = produced / elapsed if elapsed > 0 else 0
            remaining = (n_total - produced) / rate if rate > 0 else 0
            logger.info(
                f"Progress for '{table.name}': {produced:,}/{n_total:,} rows "
                f"({progress_pct:.1f}%) | "
                f"Rate: {rate:,.0f} rows/sec | "
                f"ETA: {remaining/60:.1f} min"
            )
            last_progress_time = current_time
        else:
            logger.debug(
                f"Generated chunk {chunk_num} for '{table.name}': "
                f"{len(df_chunk)} rows (total: {produced}/{n_total})"
            )
        
        yield df_chunk

    total_elapsed = time.time() - start_time
    rate = produced / total_elapsed if total_elapsed > 0 else 0
    logger.info(
        f"Completed fact table '{table.name}': {produced:,} rows in {chunk_num} chunks "
        f"({total_elapsed/60:.1f} min, {rate:,.0f} rows/sec)"
    )

