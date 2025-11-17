"""Main pipeline for IR→Data generation."""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from nl2data.ir.dataset import DatasetIR
from .dim_generator import generate_dimension
from .fact_generator import generate_fact_stream
from .writer import write_csv_stream, write_csv
from nl2data.generation.derived_registry import build_derived_registry
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def generate_from_ir(
    ir: DatasetIR, out_dir: Path, seed: int, chunk_rows: int
) -> None:
    """
    Generate data from DatasetIR.

    Args:
        ir: Dataset intermediate representation
        out_dir: Output directory for generated files
        seed: Random seed
        chunk_rows: Number of rows per chunk for fact tables
    """
    logger.info(f"Starting data generation (seed={seed})")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build derived registry (compiles expressions, computes dependency order)
    logger.info("Building derived column registry")
    derived_reg = build_derived_registry(ir)
    if derived_reg.programs:
        logger.info(
            f"Found {len(derived_reg.programs)} derived columns across "
            f"{len(derived_reg.order)} tables"
        )

    # Separate dimension and fact tables
    dims = {
        name: t
        for name, t in ir.logical.tables.items()
        if t.kind == "dimension"
    }
    facts = {
        name: t
        for name, t in ir.logical.tables.items()
        if t.kind == "fact"
    }

    logger.info(
        f"Found {len(dims)} dimension tables and {len(facts)} fact tables"
    )

    # Generate dimension tables first
    dim_dfs: Dict[str, pd.DataFrame] = {}
    failed_dimensions = []
    for name, table_spec in dims.items():
        logger.info(f"Generating dimension table: {name}")
        try:
            df = generate_dimension(table_spec, ir, rng, derived_reg)
            output_path = out_dir / f"{name}.csv"
            write_csv(df, output_path)
            dim_dfs[name] = df
            logger.info(f"Successfully generated dimension table: {name}")
        except Exception as e:
            logger.error(
                f"Failed to generate dimension table '{name}': {e}. "
                f"Continuing with remaining tables."
            )
            failed_dimensions.append(name)
            # Continue with next table instead of stopping

    # Generate fact tables (streaming)
    failed_tables = []
    for name, table_spec in facts.items():
        logger.info(f"Generating fact table: {name}")
        try:
            stream = generate_fact_stream(table_spec, ir, dim_dfs, rng, chunk_rows, derived_reg)
            output_path = out_dir / f"{name}.csv"
            write_csv_stream(stream, output_path)
            logger.info(f"Successfully generated fact table: {name}")
        except Exception as e:
            logger.error(
                f"Failed to generate fact table '{name}': {e}. "
                f"Continuing with remaining tables."
            )
            failed_tables.append(name)
            # Continue with next table instead of stopping

    # Report results
    total_failed = len(failed_dimensions) + len(failed_tables)
    if total_failed > 0:
        logger.warning(
            f"Data generation completed with {total_failed} failed table(s). "
            f"Failed dimensions: {failed_dimensions if failed_dimensions else 'none'}. "
            f"Failed facts: {failed_tables if failed_tables else 'none'}. "
            f"Output directory: {out_dir}"
        )
    else:
        logger.info(f"Data generation completed successfully. Output directory: {out_dir}")

