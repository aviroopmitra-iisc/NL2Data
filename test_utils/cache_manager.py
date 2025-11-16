"""Utilities for managing cached/generated data."""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from nl2data.utils.ir_io import load_ir_from_json
from nl2data.ir.dataset import DatasetIR


def hash_query_content(query_text: str) -> str:
    """
    Generate a hash of the query content for folder naming.
    
    Args:
        query_text: The natural language query text
        
    Returns:
        First 12 characters of MD5 hash
    """
    return hashlib.md5(query_text.encode('utf-8')).hexdigest()[:12]


def check_existing_data(query_output: Path) -> Tuple[bool, Optional[DatasetIR], Optional[List[str]]]:
    """
    Check if data already exists in the output folder and is complete.
    
    This checks:
    1. If the folder exists
    2. If the IR JSON file exists
    3. If all expected CSV files exist (based on table names in IR)
    
    Args:
        query_output: Path to the query output directory
        
    Returns:
        Tuple of (data_exists, ir, table_names)
        - data_exists: True if all required files exist
        - ir: DatasetIR if loaded successfully, None otherwise
        - table_names: List of expected table names if data exists, None otherwise
    """
    ir_file = query_output / "dataset_ir.json"
    
    # Check if folder exists
    if not query_output.exists():
        return False, None, None
    
    # Check if IR JSON exists
    if not ir_file.exists():
        return False, None, None
    
    try:
        # Load IR from JSON
        ir = load_ir_from_json(ir_file)
        table_names = list(ir.logical.tables.keys())
        
        # Check if all expected CSV files exist
        for table_name in table_names:
            csv_file = query_output / f"{table_name}.csv"
            if not csv_file.exists():
                return False, None, None
        
        # All checks passed
        return True, ir, table_names
        
    except Exception:
        # If loading IR fails, data is incomplete
        return False, None, None

