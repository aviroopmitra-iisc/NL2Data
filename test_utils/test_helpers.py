"""Helper functions for test execution."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import DistDerived


def count_derived_columns(ir: DatasetIR) -> List[Tuple[str, str, str]]:
    """
    Count and list derived columns in the IR.
    
    Args:
        ir: Dataset IR
        
    Returns:
        List of tuples (table_name, column_name, expression)
    """
    return [
        (cg.table, cg.column, cg.distribution.expression)
        for cg in ir.generation.columns
        if isinstance(cg.distribution, DistDerived)
    ]


def load_dataframes(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a directory into DataFrames.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping table names (file stems) to DataFrames
    """
    dfs = {}
    for csv_file in data_dir.glob("*.csv"):
        table_name = csv_file.stem
        dfs[table_name] = pd.read_csv(csv_file)
    return dfs


def get_data_summary(data_dir: Path) -> Dict[str, Any]:
    """
    Get summary statistics about generated data.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary with summary information (file_count, total_size_mb, etc.)
    """
    csv_files = list(data_dir.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files)
    
    return {
        "file_count": len(csv_files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / 1024 / 1024,
        "files": [f.name for f in csv_files],
    }

