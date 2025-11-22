"""Dimension table generator."""

import numpy as np
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.logical import TableSpec, ColumnSpec
from nl2data.ir.generation import DistDerived
from nl2data.generation.distributions import get_sampler
from nl2data.generation.derived_registry import DerivedRegistry
from nl2data.generation.derived_eval import eval_derived
from nl2data.generation.type_enforcement import enforce_column_type
from nl2data.generation.ir_helpers import build_distribution_map, build_provider_map
from nl2data.generation.uniqueness import (
    is_category_column,
    is_person_name_column,
    enforce_unique_categorical_column,
    enforce_unique_non_text_column,
)
from nl2data.generation.constants import DEFAULT_DIMENSION_ROWS
from nl2data.generation.error_logging import log_error, log_error_with_recovery
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def sample_column(
    table: TableSpec,
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
    provider_ref=None,
) -> np.ndarray:
    """
    Sample values for a column.

    Args:
        table: Table specification
        col: Column specification
        dist: Distribution specification (or None for fallback)
        n: Number of samples
        rng: Random number generator
        provider_ref: Optional ProviderRef to use instead of distribution

    Returns:
        Array of sampled values
    """
    # Check if provider should be used (provider takes precedence over distribution)
    if provider_ref is not None:
        try:
            from nl2data.generation.providers.registry import get_provider
            provider = get_provider(provider_ref.name, provider_ref.config)
            result = provider.sample(n)
            # Convert to numpy array if needed
            if isinstance(result, pd.Series):
                result = result.values
            elif not isinstance(result, np.ndarray):
                result = np.array(result)
            # Enforce SQL type after sampling
            return enforce_column_type(result, col.sql_type, col.name, rng)
        except Exception as e:
            log_error_with_recovery(
                error=e,
                recovery_action=f"Falling back to distribution for {table.name}.{col.name}",
                context={'provider': provider_ref.name, 'provider_config': provider_ref.config},
                operation="provider-based column sampling",
                table_name=table.name,
                column_name=col.name
            )
            # Fall through to distribution-based generation
    
    if dist is None:
        # Fallback by SQL type
        from nl2data.generation.column_sampling import sample_fallback_column
        return sample_fallback_column(col, n, rng)

    # Use specified distribution
    sampler = get_sampler(dist, rng=rng)
    result = sampler.sample(n, rng=rng)
    
    # Enforce SQL type after sampling
    return enforce_column_type(result, col.sql_type, col.name, rng)


def generate_dimension(
    table: TableSpec,
    ir: DatasetIR,
    rng: np.random.Generator,
    derived_reg: DerivedRegistry,
) -> pd.DataFrame:
    """
    Generate a dimension table with two-phase generation (base → derived).

    Args:
        table: Table specification
        ir: Dataset IR
        rng: Random number generator
        derived_reg: Derived registry with compiled expressions

    Returns:
        DataFrame with generated data (including derived columns)
    """
    import time
    table_start = time.time()
    n = table.row_count or DEFAULT_DIMENSION_ROWS
    logger.info(f"Generating dimension table '{table.name}' with {n} rows, {len(table.columns)} columns")
    
    # Build distribution and provider maps
    map_start = time.time()
    gen_map = build_distribution_map(ir)
    provider_map = build_provider_map(ir)
    map_time = time.time() - map_start
    logger.debug(f"  Built distribution/provider maps in {map_time:.3f}s")
    
    # Log column details
    base_cols = [col for col in table.columns if not isinstance(gen_map.get((table.name, col.name)), DistDerived)]
    derived_cols_count = len(table.columns) - len(base_cols)
    logger.debug(
        f"  Columns: {len(base_cols)} base, {derived_cols_count} derived, "
        f"{len([c for c in table.columns if c.role == 'primary_key'])} primary key(s), "
        f"{len([c for c in table.columns if c.unique])} unique column(s)"
    )

    # Phase 1: Generate base columns (non-derived)
    base_start = time.time()
    base_data = {}
    for col in table.columns:
        dist = gen_map.get((table.name, col.name))
        if isinstance(dist, DistDerived):
            continue  # Skip derived columns in first phase
        provider_ref = provider_map.get((table.name, col.name))
        
        # Special handling for primary key columns - ensure uniqueness
        if col.role == "primary_key":
            from nl2data.generation.column_sampling import sample_primary_key_column
            base_data[col.name] = sample_primary_key_column(col, dist, n, rng)
        else:
            base_data[col.name] = sample_column(table, col, dist, n, rng, provider_ref=provider_ref)
            logger.debug(f"  Generated base column '{col.name}' ({col.sql_type})")

    df = pd.DataFrame(base_data)
    base_time = time.time() - base_start
    logger.debug(f"  Base columns generated in {base_time:.3f}s: {len(base_data)} columns")

    # Heuristic: For dimension tables, enforce uniqueness on type/category columns
    # even if not explicitly marked in IR (common LLM oversight)
    # Note: We're more conservative - only apply to columns that clearly represent
    # categories/types, not person names which can have duplicates
    if table.kind == "dimension":
        for col in table.columns:
            if is_category_column(col.name) and not is_person_name_column(col.name):
                if col.role != "primary_key" and col.sql_type == "TEXT":
                    # Check if we have a categorical distribution with limited domain
                    dist = gen_map.get((table.name, col.name))
                    if dist and hasattr(dist, 'domain') and hasattr(dist.domain, 'values'):
                        domain_size = len(dist.domain.values)
                        if domain_size < n:
                            # Domain is smaller than row count - this column should be unique
                            # Limit table to domain_size rows (one per unique value)
                            all_values = list(dist.domain.values)
                            # Sample all unique values exactly once
                            sampled = rng.choice(all_values, size=min(n, domain_size), replace=False)
                            # Update the dataframe to only have unique values
                            df = df.iloc[:len(sampled)].copy()
                            df[col.name] = sampled
                            # Regenerate other columns for the reduced row count
                            new_n = len(sampled)
                            for c in table.columns:
                                if c.name != col.name and c.name in df.columns:
                                    c_dist = gen_map.get((table.name, c.name))
                                    if not isinstance(c_dist, DistDerived):
                                        # Special handling for PKs to ensure uniqueness
                                        if c.role == "primary_key":
                                            from nl2data.generation.column_sampling import sample_primary_key_column
                                            df[c.name] = sample_primary_key_column(c, c_dist, new_n, rng)
                                        else:
                                            c_provider = provider_map.get((table.name, c.name))
                                            df[c.name] = sample_column(table, c, c_dist, new_n, rng, provider_ref=c_provider)
                            logger.info(
                                f"Enforced uniqueness on {table.name}.{col.name}: "
                                f"reduced table from {n} to {new_n} rows "
                                f"(domain has only {domain_size} unique values)"
                            )
                            n = new_n  # Update n for derived column computation

    # Enforce uniqueness constraints (explicitly marked in IR)
    for col in table.columns:
        if col.unique and col.name in df.columns:
            # Check if we have duplicates
            if df[col.name].duplicated().any():
                # Has duplicates - need to fix
                if col.sql_type == "TEXT":
                    dist = gen_map.get((table.name, col.name))
                    df, n = enforce_unique_categorical_column(df, col, dist, n, rng)
                else:
                    df, n = enforce_unique_non_text_column(df, col, n)

    # Phase 2: Compute derived columns in dependency order
    derived_cols = derived_reg.order.get(table.name, [])
    derived_time = 0
    if derived_cols:
        derived_start = time.time()
        logger.debug(
            f"Computing {len(derived_cols)} derived columns for '{table.name}'"
        )
        for col_name in derived_cols:
            key = (table.name, col_name)
            if key in derived_reg.programs:
                prog = derived_reg.programs[key]
                try:
                    col_start = time.time()
                    df[col_name] = eval_derived(prog, df, rng=rng)
                    col_time = time.time() - col_start
                    logger.debug(f"  Computed derived column '{col_name}' in {col_time:.3f}s")
                except Exception as e:
                    log_error(
                        error=e,
                        context={
                            'expression': prog.expr if hasattr(prog, 'expr') else 'N/A',
                            'dependencies': getattr(prog, 'dependencies', []),
                            'available_columns': list(df.columns)
                        },
                        operation="derived column computation",
                        table_name=table.name,
                        column_name=col_name
                    )
                    raise
        derived_time = time.time() - derived_start
        logger.debug(f"  All derived columns computed in {derived_time:.3f}s")
    else:
        logger.debug(f"  No derived columns to compute for '{table.name}'")

    total_time = time.time() - table_start
    logger.info(
        f"Generated dimension table '{table.name}': {len(df)} rows, {len(df.columns)} columns "
        f"(base: {base_time:.3f}s, derived: {derived_time:.3f}s, total: {total_time:.3f}s)"
    )
    return df

