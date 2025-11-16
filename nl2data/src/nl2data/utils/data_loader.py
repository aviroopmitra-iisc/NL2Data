"""Utilities for loading data files."""

from pathlib import Path
from typing import Dict
import pandas as pd


def load_csv_files(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a directory into DataFrames.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping table names (file stems) to DataFrames
    """
    return {p.stem: pd.read_csv(p) for p in data_dir.glob("*.csv")}

