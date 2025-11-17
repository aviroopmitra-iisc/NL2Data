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


def check_ir_exists(query_output: Path) -> Tuple[bool, Optional[DatasetIR]]:
    """
    Check if IR JSON file exists and can be loaded.
    
    Args:
        query_output: Path to the query output directory
        
    Returns:
        Tuple of (ir_exists, ir)
        - ir_exists: True if IR file exists and is valid
        - ir: DatasetIR if loaded successfully, None otherwise
    """
    ir_file = query_output / "dataset_ir.json"
    
    # Check if folder exists
    if not query_output.exists():
        return False, None
    
    # Check if IR JSON exists
    if not ir_file.exists():
        return False, None
    
    try:
        # Check if file is empty before trying to load
        file_content = ir_file.read_text(encoding="utf-8").strip()
        if not file_content:
            # File exists but is empty - treat as incomplete
            print(f"[WARNING] IR file exists but is empty: {ir_file}")
            print(f"  Deleting empty file to force regeneration...")
            ir_file.unlink()
            return False, None
        
        # Load IR from JSON
        ir = load_ir_from_json(ir_file)
        return True, ir
        
    except (ValueError, FileNotFoundError) as e:
        # If loading IR fails due to empty/corrupted file, delete it and regenerate
        if ir_file.exists():
            print(f"[WARNING] IR file is corrupted: {ir_file}")
            print(f"  Error: {e}")
            print(f"  Deleting corrupted file to force regeneration...")
            ir_file.unlink()
        return False, None
    except Exception as e:
        # Other exceptions - log but don't delete file
        print(f"[WARNING] Failed to load IR from {ir_file}: {e}")
        return False, None


def check_csv_files_exist(query_output: Path, table_names: List[str]) -> bool:
    """
    Check if all expected CSV files exist in the data subfolder.
    
    Args:
        query_output: Path to the query output directory
        table_names: List of expected table names
        
    Returns:
        True if all CSV files exist, False otherwise
    """
    data_dir = query_output / "data"
    
    # Check if data folder exists
    if not data_dir.exists():
        return False
    
    # Check if all expected CSV files exist
    for table_name in table_names:
        csv_file = data_dir / f"{table_name}.csv"
        if not csv_file.exists():
            return False
    
    return True


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
    ir_exists, ir = check_ir_exists(query_output)
    
    if not ir_exists or ir is None:
        return False, None, None
    
    table_names = list(ir.logical.tables.keys())
    
    # Check if all CSV files exist
    if check_csv_files_exist(query_output, table_names):
        return True, ir, table_names
    
    return False, ir, table_names

