"""Fact table generator with streaming support."""

import numpy as np
import pandas as pd
from typing import Dict, Iterator, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.logical import TableSpec, ColumnSpec
from nl2data.ir.generation import DistZipf, DistDerived, DistWindow
from nl2data.generation.distributions import get_sampler
from nl2data.generation.derived_registry import DerivedRegistry
from nl2data.generation.derived_eval import eval_derived
from nl2data.generation.window_eval import compute_window_columns
from nl2data.generation.event_eval import apply_events_to_chunk
from nl2data.generation.enforce import enforce_batch
from nl2data.generation.allocator import (
    zipf_probs,
    fk_assignments,
)
from nl2data.generation.constants import (
    DEFAULT_FACT_ROWS,
    PROGRESS_LOG_INTERVAL_SECONDS,
    LARGE_TABLE_THRESHOLD,
)
from nl2data.generation.error_logging import log_error, log_error_with_recovery
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


def _generate_single_chunk(
    chunk_num: int,
    chunk_start: int,
    chunk_size: int,
    table: TableSpec,
    ir: DatasetIR,
    dims: Dict[str, pd.DataFrame],
    base_seed: int,
    gen_map: Dict,
    derived_cols: list,
    derived_reg: DerivedRegistry,
    filtered_constraints,
) -> Tuple[int, pd.DataFrame]:
    """
    Generate a single chunk of fact table data.
    
    Returns:
        Tuple of (chunk_num, DataFrame) for ordering
    """
    import time
    chunk_gen_start = time.time()
    
    # Create RNG with unique seed for this chunk
    chunk_seed = base_seed + chunk_num * 1000 + chunk_start
    rng = np.random.default_rng(chunk_seed)
    logger.debug(
        f"  Generating chunk {chunk_num} for '{table.name}': "
        f"rows {chunk_start} to {chunk_start + chunk_size - 1} (seed={chunk_seed})"
    )
    
    # Phase 1: Generate base columns (non-derived, non-window)
    base_start = time.time()
    base_block = {}
    base_col_count = 0
    for col in table.columns:
        dist = gen_map.get((table.name, col.name))
        if isinstance(dist, (DistDerived, DistWindow)):
            continue  # Skip derived and window columns in first phase
        base_block[col.name] = sample_fact_column(
            table, col, dist, chunk_size, rng, dims
        )
        base_col_count += 1

    df_chunk = pd.DataFrame(base_block)
    base_time = time.time() - base_start
    logger.debug(
        f"    Chunk {chunk_num}: Generated {base_col_count} base columns in {base_time:.3f}s"
    )

    # Phase 1.5: Apply event effects (after base columns, before derived)
    if ir.generation.events:
        try:
            df_chunk = apply_events_to_chunk(
                df_chunk, table, ir, chunk_start_idx=chunk_start, rng=rng
            )
        except Exception as e:
            log_error_with_recovery(
                error=e,
                recovery_action="Continuing without event effects",
                context={
                    'chunk_start': chunk_start,
                    'chunk_size': chunk_size,
                    'events_count': len(ir.generation.events) if ir.generation.events else 0
                },
                operation="applying event effects",
                table_name=table.name,
                chunk_num=chunk_num
            )
            # Continue without event effects

    # Phase 1.6: Join dimension tables (for derived column lookups)
    for fk in table.foreign_keys:
        ref_table_name = fk.ref_table
        if ref_table_name in dims:
            dim_df = dims[ref_table_name]
            df_chunk = df_chunk.merge(
                dim_df,
                how="left",
                left_on=fk.column,
                right_on=fk.ref_column,
                suffixes=("", f"_{ref_table_name}")
            )
        else:
            logger.warning(
                f"Dimension table '{ref_table_name}' not found for join. "
                f"Derived expressions referencing this dimension will fail."
            )

    # Phase 1.7: Enforce constraints
    if filtered_constraints.fds or filtered_constraints.implications or filtered_constraints.composite_pks:
        try:
            df_chunk = enforce_batch(
                df_chunk,
                filtered_constraints,
                table_spec=table,
            )
        except Exception as e:
            log_error_with_recovery(
                error=e,
                recovery_action="Continuing without constraint enforcement",
                context={
                    'chunk_start': chunk_start,
                    'chunk_size': chunk_size,
                    'fds_count': len(filtered_constraints.fds),
                    'implications_count': len(filtered_constraints.implications),
                    'composite_pks_count': len(filtered_constraints.composite_pks)
                },
                operation="enforcing constraints",
                table_name=table.name,
                chunk_num=chunk_num
            )
            # Continue without constraint enforcement

    # Phase 2: Compute derived columns in dependency order
    derived_start = time.time()
    if derived_cols:
        logger.debug(f"    Chunk {chunk_num}: Computing {len(derived_cols)} derived columns")
        for col_name in derived_cols:
            key = (table.name, col_name)
            if key in derived_reg.programs:
                prog = derived_reg.programs[key]
                try:
                    col_start = time.time()
                    df_chunk[col_name] = eval_derived(prog, df_chunk, rng=rng)
                    col_time = time.time() - col_start
                    logger.debug(f"      Computed '{col_name}' in {col_time:.3f}s")
                except KeyError as e:
                    missing_col = str(e).strip("'")
                    log_error(
                        error=e,
                        context={
                            'expression': prog.expr if hasattr(prog, 'expr') else 'N/A',
                            'dependencies': getattr(prog, 'dependencies', []),
                            'available_columns': list(df_chunk.columns),
                            'missing_column': missing_col
                        },
                        operation="derived column computation",
                        table_name=table.name,
                        column_name=col_name,
                        chunk_num=chunk_num
                    )
                    raise ValueError(
                        f"Derived column '{col_name}' in table '{table.name}' depends on "
                        f"column '{missing_col}' which is not available."
                    ) from e
                except Exception as e:
                    log_error(
                        error=e,
                        context={
                            'expression': prog.expr if hasattr(prog, 'expr') else 'N/A',
                            'dependencies': getattr(prog, 'dependencies', []),
                            'available_columns': list(df_chunk.columns),
                            'chunk_size': len(df_chunk)
                        },
                        operation="derived column computation",
                        table_name=table.name,
                        column_name=col_name,
                        chunk_num=chunk_num
                    )
                    raise
        derived_time = time.time() - derived_start
        logger.debug(f"    Chunk {chunk_num}: All derived columns computed in {derived_time:.3f}s")
    
    total_chunk_time = time.time() - chunk_gen_start
    logger.debug(
        f"  Chunk {chunk_num} completed: {len(df_chunk)} rows, {len(df_chunk.columns)} columns "
        f"in {total_chunk_time:.3f}s"
    )

    return (chunk_num, df_chunk)


def generate_fact_stream(
    table: TableSpec,
    ir: DatasetIR,
    dims: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    chunk_rows: int,
    derived_reg: DerivedRegistry,
    use_parallel: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Generate fact table in chunks (streaming) with two-phase generation (base → derived).
    Supports parallel chunk generation for improved performance.

    Args:
        table: Table specification
        ir: Dataset IR
        dims: Dictionary of dimension DataFrames
        rng: Random number generator (base seed)
        chunk_rows: Number of rows per chunk
        derived_reg: Derived registry with compiled expressions
        use_parallel: Whether to use parallel chunk generation (default: True)

    Yields:
        DataFrame chunks (with derived columns computed)
    """
    n_total = table.row_count or DEFAULT_FACT_ROWS
    
    # Build distribution map
    from nl2data.generation.ir_helpers import build_distribution_map
    gen_map = build_distribution_map(ir)

    # Get derived column order for this table
    derived_cols = derived_reg.order.get(table.name, [])
    
    # Check if this table has window columns (requires full materialization)
    window_cols = derived_reg.window_order.get(table.name, [])
    has_windows = len(window_cols) > 0
    
    # Filter constraints for this table
    table_constraints = ir.logical.constraints
    from nl2data.ir.constraint_ir import ConstraintSpec
    filtered_constraints = ConstraintSpec(
        fds=[fd for fd in table_constraints.fds if fd.table == table.name],
        implications=[impl for impl in table_constraints.implications if impl.table == table.name],
        composite_pks=[cpk for cpk in table_constraints.composite_pks if cpk.table == table.name]
    )
    
    # Get base seed from RNG
    base_seed = rng.integers(0, 2**31)
    
    # Calculate number of chunks
    num_chunks = (n_total + chunk_rows - 1) // chunk_rows
    
    # Determine if we should use parallel processing
    # Don't use parallel for small tables or tables with windows
    should_parallelize = (
        use_parallel 
        and not has_windows 
        and num_chunks > 1 
        and n_total > chunk_rows * 2  # At least 2 chunks
    )
    
    if has_windows:
        logger.info(
            f"Generating fact table '{table.name}' with {n_total} rows "
            f"(chunk size: {chunk_rows}, {num_chunks} chunks, window columns require sequential processing)"
        )
    elif should_parallelize:
        # Calculate number of worker threads (CPU count - 2, minimum 1)
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count - 2)
        logger.info(
            f"Generating fact table '{table.name}' with {n_total} rows "
            f"(chunk size: {chunk_rows}, {num_chunks} chunks, parallel with {max_workers} workers)"
        )
    else:
        logger.info(
            f"Generating fact table '{table.name}' with {n_total} rows "
            f"(chunk size: {chunk_rows}, {num_chunks} chunks, sequential)"
        )
    
    import time
    start_time = time.time()
    produced = 0  # Track total rows produced
    chunk_num = 0  # Track number of chunks
    
    if should_parallelize:
        # Parallel chunk generation
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count - 2)
        
        # Prepare chunk tasks
        chunk_tasks = []
        temp_produced = 0
        temp_chunk_num = 0
        while temp_produced < n_total:
            m = min(chunk_rows, n_total - temp_produced)
            chunk_tasks.append((temp_chunk_num, temp_produced, m))
            temp_produced += m
            temp_chunk_num += 1
        
        num_chunks = len(chunk_tasks)
        
        # Generate chunks in parallel
        produced = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk generation tasks
            future_to_chunk = {
                executor.submit(
                    _generate_single_chunk,
                    chunk_num, chunk_start, chunk_size,
                    table, ir, dims, base_seed,
                    gen_map, derived_cols, derived_reg, filtered_constraints
                ): chunk_num
                for chunk_num, chunk_start, chunk_size in chunk_tasks
            }
            
            # Collect results as they complete (but maintain order when yielding)
            completed_chunks = {}
            next_chunk = 0
            last_progress_time = start_time
            
            for future in as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    result_chunk_num, df_chunk = future.result()
                    completed_chunks[result_chunk_num] = df_chunk
                    produced += len(df_chunk)
                    
                    # Yield chunks in order
                    while next_chunk in completed_chunks:
                        yield completed_chunks.pop(next_chunk)
                        next_chunk += 1
                    
                    # Progress tracking
                    current_time = time.time()
                    if n_total > LARGE_TABLE_THRESHOLD and (current_time - last_progress_time) >= PROGRESS_LOG_INTERVAL_SECONDS:
                        elapsed = current_time - start_time
                        progress_pct = (produced / n_total * 100) if n_total > 0 else 0
                        rate = produced / elapsed if elapsed > 0 else 0
                        remaining = (n_total - produced) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress for '{table.name}': {produced:,}/{n_total:,} rows "
                            f"({progress_pct:.1f}%) | "
                            f"Rate: {rate:,.0f} rows/sec | "
                            f"ETA: {remaining/60:.1f} min"
                        )
                        last_progress_time = current_time
                        
                except Exception as e:
                    log_error(
                        error=e,
                        context={
                            'chunk_num': chunk_num,
                            'total_chunks': num_chunks,
                            'chunk_start': chunk_start if 'chunk_start' in locals() else None,
                            'chunk_size': chunk_size if 'chunk_size' in locals() else None
                        },
                        operation="parallel chunk generation",
                        table_name=table.name,
                        chunk_num=chunk_num
                    )
                    raise
            
            # Yield any remaining chunks in order
            while next_chunk < num_chunks:
                if next_chunk in completed_chunks:
                    yield completed_chunks.pop(next_chunk)
                next_chunk += 1
        chunk_num = num_chunks
    else:
        # Sequential chunk generation (original logic)
        if has_windows:
            all_chunks = []
        
        last_progress_time = start_time

        while produced < n_total:
            m = min(chunk_rows, n_total - produced)
            
            chunk_num_result, df_chunk = _generate_single_chunk(
                chunk_num, produced, m,
                table, ir, dims, base_seed,
                gen_map, derived_cols, derived_reg, filtered_constraints
            )

            produced += len(df_chunk)
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
            
            if has_windows:
                # Store chunk for later window computation
                all_chunks.append(df_chunk)
            else:
                # No windows - yield immediately (streaming)
                yield df_chunk
        
        # Phase 3: Compute window columns if needed (after all rows are generated)
        if has_windows:
            logger.info(
                f"Materializing full table '{table.name}' for window computation "
                f"({n_total:,} rows)..."
            )
            # Concatenate all chunks
            df_full = pd.concat(all_chunks, ignore_index=True)
            
            # Collect window specifications for this table
            window_specs = {}
            for col_name in window_cols:
                key = (table.name, col_name)
                if key in derived_reg.windows:
                    window_specs[col_name] = derived_reg.windows[key]
            
            # Compute window columns
            logger.info(f"Computing {len(window_specs)} window column(s)...")
            try:
                df_full = compute_window_columns(df_full, window_specs)
            except Exception as e:
                logger.error(
                    f"Failed to compute window columns for table '{table.name}': {e}"
                )
                raise
            
            # Yield in chunks (for consistency with streaming interface)
            # Split back into chunks of chunk_rows
            for i in range(0, len(df_full), chunk_rows):
                chunk = df_full.iloc[i:i + chunk_rows].copy()
                yield chunk

    # Final summary
    total_elapsed = time.time() - start_time
    rate = produced / total_elapsed if total_elapsed > 0 else 0
    logger.info(
        f"Completed fact table '{table.name}': {produced:,} rows in {chunk_num} chunks "
        f"({total_elapsed/60:.1f} min, {rate:,.0f} rows/sec)"
    )

