"""File writers for generated data."""

from pathlib import Path
from typing import Iterator
import pandas as pd
from nl2data.generation.error_logging import log_error
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def write_csv_stream(df_iter: Iterator[pd.DataFrame], path: Path) -> None:
    """
    Write a stream of DataFrames to a CSV file.

    Args:
        df_iter: Iterator of DataFrames
        path: Output file path
    """
    import time
    write_start = time.time()
    header_written = False
    chunk_count = 0
    total_rows = 0

    logger.debug(f"Starting to write CSV stream to {path}")
    
    for df in df_iter:
        chunk_start = time.time()
        rows_in_chunk = len(df)
        try:
            df.to_csv(
                path,
                index=False,
                header=not header_written,
                mode="w" if not header_written else "a",
            )
            header_written = True
            chunk_count += 1
            total_rows += rows_in_chunk
            chunk_time = time.time() - chunk_start
            logger.debug(
                f"Wrote chunk {chunk_count} to {path}: {rows_in_chunk} rows in {chunk_time:.3f}s"
            )
        except Exception as e:
            from nl2data.generation.error_logging import log_error
            log_error(
                error=e,
                context={
                    'chunk_num': chunk_count + 1,
                    'rows_in_chunk': rows_in_chunk,
                    'columns': len(df.columns),
                    'file_path': str(path),
                    'file_exists': path.exists()
                },
                operation="writing CSV chunk",
                chunk_num=chunk_count + 1
            )
            raise

    total_time = time.time() - write_start
    file_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
    logger.info(
        f"Completed writing {chunk_count} chunk(s) to {path}: "
        f"{total_rows:,} total rows, {file_size_mb:.2f} MB, "
        f"write time: {total_time:.3f}s "
        f"({total_rows/total_time:,.0f} rows/sec)"
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write a single DataFrame to CSV.

    Args:
        df: DataFrame to write
        path: Output file path
    """
    import time
    write_start = time.time()
    rows = len(df)
    cols = len(df.columns)
    
    logger.debug(f"Writing DataFrame to {path}: {rows} rows, {cols} columns")
    
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        log_error(
            error=e,
            context={
                'rows': rows,
                'columns': cols,
                'file_path': str(path),
                'file_exists': path.exists(),
                'parent_dir_exists': path.parent.exists() if path.parent else False
            },
            operation="writing CSV file"
        )
        raise
    
    write_time = time.time() - write_start
    file_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
    logger.info(
        f"Wrote {rows:,} rows, {cols} columns to {path}: "
        f"{file_size_mb:.2f} MB in {write_time:.3f}s "
        f"({rows/write_time:,.0f} rows/sec)"
    )

