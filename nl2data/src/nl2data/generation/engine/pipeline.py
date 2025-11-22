"""Main pipeline for IR→Data generation."""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import time
from nl2data.ir.dataset import DatasetIR
from .dim_generator import generate_dimension
from .fact_generator import generate_fact_stream
from .writer import write_csv_stream, write_csv
from nl2data.generation.derived_registry import build_derived_registry
from nl2data.generation.error_logging import log_error
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
    pipeline_start = time.time()
    logger.info(f"Starting data generation (seed={seed}, chunk_rows={chunk_rows}, output_dir={out_dir})")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory created/verified: {out_dir}")

    # Build derived registry (compiles expressions, computes dependency order)
    logger.info("Building derived column registry")
    reg_start = time.time()
    derived_reg = build_derived_registry(ir)
    reg_elapsed = time.time() - reg_start
    logger.debug(f"Derived registry built in {reg_elapsed:.3f} seconds")
    
    if derived_reg.programs:
        logger.info(
            f"Found {len(derived_reg.programs)} derived columns across "
            f"{len(derived_reg.order)} tables"
        )
        for table_name, cols in derived_reg.order.items():
            logger.debug(f"  Table '{table_name}': {len(cols)} derived columns: {', '.join(cols)}")
    else:
        logger.debug("No derived columns found in IR")

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
        f"Found {len(dims)} dimension table(s) and {len(facts)} fact table(s)"
    )
    if dims:
        logger.debug(f"Dimension tables: {', '.join(dims.keys())}")
    if facts:
        logger.debug(f"Fact tables: {', '.join(facts.keys())}")

    # Generate dimension tables first
    dim_dfs: Dict[str, pd.DataFrame] = {}
    failed_dimensions = []
    dim_start = time.time()
    
    logger.info(f"Starting dimension table generation ({len(dims)} table(s))")
    for idx, (name, table_spec) in enumerate(dims.items(), 1):
        table_start = time.time()
        logger.info(f"[{idx}/{len(dims)}] Generating dimension table: {name}")
        logger.debug(
            f"  Table spec: {len(table_spec.columns)} columns, "
            f"row_count={table_spec.row_count}, "
            f"primary_key={table_spec.primary_key}, "
            f"foreign_keys={len(table_spec.foreign_keys)}"
        )
        
        try:
            df = generate_dimension(table_spec, ir, rng, derived_reg)
            table_gen_time = time.time() - table_start
            logger.debug(f"  Generation completed in {table_gen_time:.3f} seconds")
            
            output_path = out_dir / f"{name}.csv"
            write_start = time.time()
            write_csv(df, output_path)
            write_time = time.time() - write_start
            logger.debug(f"  File write completed in {write_time:.3f} seconds")
            
            dim_dfs[name] = df
            total_time = time.time() - table_start
            logger.info(
                f"Successfully generated dimension table '{name}': "
                f"{len(df)} rows, {len(df.columns)} columns "
                f"(total time: {total_time:.3f}s)"
            )
        except Exception as e:
            table_time = time.time() - table_start
            log_error(
                error=e,
                context={
                    'table_index': idx,
                    'total_tables': len(dims),
                    'elapsed_time': f"{table_time:.3f}s",
                    'columns': len(table_spec.columns),
                    'row_count': table_spec.row_count,
                    'primary_keys': table_spec.primary_key,
                    'foreign_keys': len(table_spec.foreign_keys)
                },
                operation="dimension table generation",
                table_name=name
            )
            logger.warning(f"Continuing with remaining dimension tables after failure in '{name}'")
            failed_dimensions.append(name)
            # Continue with next table instead of stopping
    
    dim_total_time = time.time() - dim_start
    logger.info(
        f"Dimension table generation completed: "
        f"{len(dim_dfs)} succeeded, {len(failed_dimensions)} failed "
        f"(total time: {dim_total_time:.3f}s)"
    )

    # Generate fact tables (streaming)
    failed_tables = []
    fact_start = time.time()
    
    logger.info(f"Starting fact table generation ({len(facts)} table(s))")
    for idx, (name, table_spec) in enumerate(facts.items(), 1):
        table_start = time.time()
        logger.info(f"[{idx}/{len(facts)}] Generating fact table: {name}")
        logger.debug(
            f"  Table spec: {len(table_spec.columns)} columns, "
            f"row_count={table_spec.row_count}, "
            f"primary_key={table_spec.primary_key}, "
            f"foreign_keys={len(table_spec.foreign_keys)}"
        )
        
        try:
            stream = generate_fact_stream(table_spec, ir, dim_dfs, rng, chunk_rows, derived_reg)
            output_path = out_dir / f"{name}.csv"
            write_start = time.time()
            write_csv_stream(stream, output_path)
            write_time = time.time() - write_start
            logger.debug(f"  File write completed in {write_time:.3f} seconds")
            
            # Get file size for logging
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.debug(f"  Output file size: {file_size_mb:.2f} MB")
            
            total_time = time.time() - table_start
            logger.info(
                f"Successfully generated fact table '{name}' "
                f"(total time: {total_time:.3f}s, write time: {write_time:.3f}s)"
            )
        except Exception as e:
            table_time = time.time() - table_start
            log_error(
                error=e,
                context={
                    'table_index': idx,
                    'total_tables': len(facts),
                    'elapsed_time': f"{table_time:.3f}s",
                    'columns': len(table_spec.columns),
                    'row_count': table_spec.row_count,
                    'chunk_rows': chunk_rows,
                    'primary_keys': table_spec.primary_key,
                    'foreign_keys': len(table_spec.foreign_keys)
                },
                operation="fact table generation",
                table_name=name
            )
            logger.warning(f"Continuing with remaining fact tables after failure in '{name}'")
            failed_tables.append(name)
            # Continue with next table instead of stopping
    
    fact_total_time = time.time() - fact_start
    logger.info(
        f"Fact table generation completed: "
        f"{len(facts) - len(failed_tables)} succeeded, {len(failed_tables)} failed "
        f"(total time: {fact_total_time:.3f}s)"
    )

    # Report results
    pipeline_total_time = time.time() - pipeline_start
    total_failed = len(failed_dimensions) + len(failed_tables)
    total_succeeded = len(dim_dfs) + (len(facts) - len(failed_tables))
    
    # Count total rows generated
    total_dim_rows = sum(len(df) for df in dim_dfs.values())
    total_fact_rows = 0
    for name in facts:
        if name not in failed_tables:
            fact_file = out_dir / f"{name}.csv"
            if fact_file.exists():
                try:
                    fact_df = pd.read_csv(fact_file, nrows=0)  # Just to count
                    # Count lines in file (approximate)
                    with open(fact_file, 'r', encoding='utf-8') as f:
                        total_fact_rows += sum(1 for _ in f) - 1  # Subtract header
                except Exception as e:
                    log_error(
                        error=e,
                        context={'file': str(fact_file)},
                        operation="counting fact table rows",
                        table_name=name,
                        log_level="warning"
                    )
    
    if total_failed > 0:
        logger.warning(
            f"Data generation completed with {total_failed} failed table(s). "
            f"Failed dimensions: {failed_dimensions if failed_dimensions else 'none'}. "
            f"Failed facts: {failed_tables if failed_tables else 'none'}. "
            f"Output directory: {out_dir}"
        )
    else:
        logger.info(f"Data generation completed successfully. Output directory: {out_dir}")
    
    logger.info(
        f"Pipeline summary: {total_succeeded} table(s) generated, "
        f"{total_dim_rows:,} dimension rows, "
        f"{total_fact_rows:,} fact rows, "
        f"total time: {pipeline_total_time:.3f}s "
        f"({pipeline_total_time/60:.2f} min)"
    )

