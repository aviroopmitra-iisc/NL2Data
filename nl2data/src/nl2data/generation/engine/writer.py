"""File writers for generated data."""

from pathlib import Path
from typing import Iterator
import pandas as pd
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def write_csv_stream(df_iter: Iterator[pd.DataFrame], path: Path) -> None:
    """
    Write a stream of DataFrames to a CSV file.

    Args:
        df_iter: Iterator of DataFrames
        path: Output file path
    """
    header_written = False
    chunk_count = 0

    for df in df_iter:
        df.to_csv(
            path,
            index=False,
            header=not header_written,
            mode="w" if not header_written else "a",
        )
        header_written = True
        chunk_count += 1
        logger.debug(f"Wrote chunk {chunk_count} to {path}")

    logger.info(f"Completed writing {chunk_count} chunks to {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write a single DataFrame to CSV.

    Args:
        df: DataFrame to write
        path: Output file path
    """
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(df)} rows to {path}")

