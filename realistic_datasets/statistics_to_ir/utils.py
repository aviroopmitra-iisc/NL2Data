"""Utility functions for statistics to GenerationIR conversion."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR
from nl2data.ir.constraint_ir import FDConstraint
from nl2data.utils.ir_io import load_ir_from_json


def load_dataframes(
    data_dir: Path,
    logical_ir: LogicalIR
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV files as pandas DataFrames based on table names in LogicalIR.
    
    Args:
        data_dir: Directory containing CSV files
        logical_ir: LogicalIR with table names
        
    Returns:
        Dictionary mapping table_name -> DataFrame
    """
    dfs = {}
    
    for table_name in logical_ir.tables.keys():
        # Try multiple possible CSV file names
        csv_paths = [
            data_dir / f"{table_name}.csv",
            data_dir / f"{table_name.lower()}.csv",
            data_dir / f"{table_name.upper()}.csv",
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    dfs[table_name] = df
                    break
                except Exception as e:
                    print(f"Warning: Failed to load {csv_path}: {e}")
                    continue
    
    return dfs


def save_discovered_fds(
    discovered_fds: List[FDConstraint],
    output_path: Path,
    support_confidence_map: Optional[Dict[tuple, tuple]] = None
) -> None:
    """
    Save discovered functional dependencies to JSON file.
    
    Args:
        discovered_fds: List of discovered FDConstraint objects
        output_path: Path to save discovered_fds.json
        support_confidence_map: Optional dict mapping (table, tuple(lhs), rhs) -> (support, confidence)
    """
    fds_data = {
        "fds": []
    }
    
    for fd in discovered_fds:
        fd_dict = {
            "table": fd.table,
            "lhs": fd.lhs,
            "rhs": fd.rhs,
            "mode": fd.mode,
        }
        
        # Add support and confidence if available
        if support_confidence_map:
            key = (fd.table, tuple(fd.lhs), fd.rhs[0] if fd.rhs else None)
            if key in support_confidence_map:
                support, confidence = support_confidence_map[key]
                fd_dict["support"] = support
                fd_dict["confidence"] = confidence
        
        fds_data["fds"].append(fd_dict)
    
    output_path.write_text(json.dumps(fds_data, indent=2), encoding="utf-8")


def load_discovered_fds(fds_path: Path) -> List[FDConstraint]:
    """
    Load discovered functional dependencies from JSON file.
    
    Args:
        fds_path: Path to discovered_fds.json
        
    Returns:
        List of FDConstraint objects
    """
    if not fds_path.exists():
        return []
    
    fds_data = json.loads(fds_path.read_text())
    discovered_fds = []
    
    for fd_dict in fds_data.get("fds", []):
        discovered_fds.append(FDConstraint(
            table=fd_dict["table"],
            lhs=fd_dict["lhs"],
            rhs=fd_dict["rhs"],
            mode=fd_dict.get("mode", "intra_row")
        ))
    
    return discovered_fds


def get_column_statistics(
    stats: Dict[str, Any],
    table_name: str,
    column_name: str
) -> tuple[Optional[Dict], Optional[Dict]]:
    """
    Get numeric and categorical statistics for a column.
    
    Args:
        stats: Statistics dictionary from statistics.json
        table_name: Name of the table
        column_name: Name of the column
        
    Returns:
        Tuple of (numeric_stats, categorical_stats), either can be None
    """
    numeric_key = f"table_{table_name}_numeric_stats"
    categorical_key = f"table_{table_name}_categorical_stats"
    
    numeric_stats = None
    if numeric_key in stats and column_name in stats[numeric_key]:
        numeric_stats = stats[numeric_key][column_name]
    
    categorical_stats = None
    if categorical_key in stats and column_name in stats[categorical_key]:
        categorical_stats = stats[categorical_key][column_name]
    
    return numeric_stats, categorical_stats

