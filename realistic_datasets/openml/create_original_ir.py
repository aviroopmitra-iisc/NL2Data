"""Middleware to create LogicalIR from OpenML raw data for evaluation.

This module creates a LogicalIR schema from OpenML raw_data.csv and metadata.json
files, which can then be used as the "original schema" in the evaluation framework.
"""

import json
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec
from nl2data.ir.constraint_ir import ConstraintSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import save_ir_to_json


def decode_column_name(encoded_name: str) -> str:
    """
    Decode URL-encoded column names from OpenML.
    
    Examples:
        "bw%2Fme" -> "bw/me"
        "blue%2Fbright%2Fvarn%2Fclean" -> "blue/bright/varn/clean"
        "product-type" -> "product-type" (no encoding)
    """
    try:
        # URL decode
        decoded = urllib.parse.unquote(encoded_name)
        return decoded
    except Exception:
        # If decoding fails, return original
        return encoded_name


def infer_sql_type(
    openml_type: str,
    column_name: str,
    series: pd.Series
) -> str:
    """
    Infer SQL type from OpenML data type and actual data.
    
    Args:
        openml_type: "numeric" or "nominal" from metadata
        column_name: Column name (for context)
        series: Actual data series for inspection
        
    Returns:
        SQL type: "INT", "FLOAT", "TEXT", "DATE", "DATETIME", or "BOOL"
    """
    if openml_type == "numeric":
        # Check if integer or float
        # First, try to convert to numeric (handles strings like "8", "0.0")
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if numeric_series.notna().any():
            # Check if all non-null values are integers
            non_null = numeric_series.dropna()
            if len(non_null) > 0:
                if (non_null % 1 == 0).all():
                    return "INT"
                else:
                    return "FLOAT"
        
        # Fallback: default to FLOAT for numeric
        return "FLOAT"
    
    elif openml_type == "nominal":
        # Check for boolean-like patterns
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2:
            # Could be boolean, but OpenML doesn't have bool type
            # Check for common boolean patterns
            str_vals = {str(v).upper() for v in unique_vals if pd.notna(v)}
            if str_vals.issubset({"TRUE", "FALSE", "T", "F", "1", "0", "YES", "NO", "Y", "N"}):
                return "BOOL"
        
        # Default to TEXT for nominal
        return "TEXT"
    
    else:
        # Unknown type, default to TEXT
        return "TEXT"


def create_logical_ir_from_openml(
    raw_data_path: Path,
    metadata_path: Path,
    table_name: str = "main"
) -> LogicalIR:
    """
    Create a LogicalIR schema from OpenML raw data and metadata.
    
    This creates a minimal schema suitable for evaluation:
    - Single table with all columns
    - Inferred SQL types from metadata + data inspection
    - No primary keys, foreign keys, or functional dependencies
    - Nullable columns (OpenML has missing values)
    
    Args:
        raw_data_path: Path to raw_data.csv
        metadata_path: Path to metadata.json
        table_name: Name for the table (default: "main")
        
    Returns:
        LogicalIR instance
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load raw data
    raw_df = pd.read_csv(raw_data_path, low_memory=False)
    
    # Replace OpenML missing value indicators
    # OpenML uses '?' for missing, but CSV might have empty strings or NaN
    raw_df = raw_df.replace('?', pd.NA)
    raw_df = raw_df.replace('', pd.NA)
    
    # Create column specifications
    columns = []
    feature_map = {feat["name"]: feat for feat in metadata.get("features", [])}
    
    for col_name in raw_df.columns:
        # Decode URL-encoded column names
        decoded_name = decode_column_name(col_name)
        
        # Get feature metadata
        feature = feature_map.get(col_name, {})
        openml_type = feature.get("data_type", "nominal")
        
        # Get actual data series for type inference
        series = raw_df[col_name]
        
        # Infer SQL type
        sql_type = infer_sql_type(openml_type, decoded_name, series)
        
        # Check if column has nulls
        has_nulls = series.isna().any()
        
        # Create column spec
        col_spec = ColumnSpec(
            name=decoded_name,  # Use decoded name
            sql_type=sql_type,
            nullable=has_nulls,  # Nullable if has nulls
            unique=False,  # OpenML doesn't specify uniqueness
            role=None,  # No role information
            references=None  # No FK information
        )
        
        columns.append(col_spec)
    
    # Create table specification
    table_spec = TableSpec(
        name=table_name,
        kind=None,  # OpenML datasets are typically single-table, no fact/dimension distinction
        row_count=len(raw_df),  # Actual row count from data
        columns=columns,
        primary_key=[],  # No PK information available
        foreign_keys=[]  # No FK information available
    )
    
    # Create LogicalIR
    logical_ir = LogicalIR(
        tables={table_name: table_spec},
        constraints=ConstraintSpec(),  # No FDs, implications, or composite PKs
        schema_mode="oltp"  # OpenML datasets are typically normalized single tables
    )
    
    return logical_ir


def create_dataset_ir_from_openml(
    raw_data_path: Path,
    metadata_path: Path,
    table_name: str = "main"
) -> DatasetIR:
    """
    Create a complete DatasetIR from OpenML raw data.
    
    This includes:
    - LogicalIR (schema)
    - Empty GenerationIR (no generation specs needed for original)
    - No WorkloadIR (optional)
    
    Args:
        raw_data_path: Path to raw_data.csv
        metadata_path: Path to metadata.json
        table_name: Name for the table
        
    Returns:
        DatasetIR instance
    """
    logical_ir = create_logical_ir_from_openml(
        raw_data_path, metadata_path, table_name
    )
    
    # Create minimal DatasetIR
    dataset_ir = DatasetIR(
        logical=logical_ir,
        generation=GenerationIR(columns=[]),  # No generation specs for original
        workload=None  # No workload specs
    )
    
    return dataset_ir


def create_original_ir_for_dataset(
    dataset_dir: Path,
    output_path: Optional[Path] = None,
    table_name: str = "main"
) -> Path:
    """
    Create original_ir.json for an OpenML dataset directory.
    
    Args:
        dataset_dir: Directory containing raw_data.csv and metadata.json
        output_path: Where to save original_ir.json (default: dataset_dir/original_ir.json)
        table_name: Name for the table
        
    Returns:
        Path to created original_ir.json file
    """
    raw_data_path = dataset_dir / "raw_data.csv"
    metadata_path = dataset_dir / "metadata.json"
    
    if not raw_data_path.exists():
        raise FileNotFoundError(f"raw_data.csv not found in {dataset_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dataset_dir}")
    
    # Create DatasetIR
    dataset_ir = create_dataset_ir_from_openml(
        raw_data_path, metadata_path, table_name
    )
    
    # Determine output path
    if output_path is None:
        output_path = dataset_dir / "original_ir.json"
    
    # Save to JSON
    save_ir_to_json(dataset_ir, output_path)
    
    return output_path


def create_original_irs_for_all_datasets(
    base_dir: Path,
    pattern: str = "**/raw_data.csv"
) -> Dict[str, Path]:
    """
    Create original_ir.json for all OpenML datasets in a directory.
    
    Args:
        base_dir: Base directory containing OpenML dataset folders
        pattern: Glob pattern to find raw_data.csv files
        
    Returns:
        Dictionary mapping dataset_id -> output_path
    """
    results = {}
    
    for raw_data_path in base_dir.glob(pattern):
        dataset_dir = raw_data_path.parent
        
        try:
            output_path = create_original_ir_for_dataset(dataset_dir)
            dataset_id = dataset_dir.name
            results[dataset_id] = output_path
            print(f"[OK] Created original_ir.json for dataset {dataset_id}")
        except Exception as e:
            dataset_id = dataset_dir.name
            print(f"[ERROR] Failed to create original_ir.json for dataset {dataset_id}: {e}")
            results[dataset_id] = None
    
    return results


if __name__ == "__main__":
    """CLI interface for creating original IRs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create LogicalIR from OpenML raw data"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing raw_data.csv and metadata.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for original_ir.json (default: dataset_dir/original_ir.json)"
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="main",
        help="Name for the table (default: main)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets in base directory"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Process all datasets
        results = create_original_irs_for_all_datasets(args.dataset_dir)
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\n[OK] Successfully created {successful}/{len(results)} original IRs")
    else:
        # Process single dataset
        output_path = create_original_ir_for_dataset(
            args.dataset_dir,
            args.output,
            args.table_name
        )
        print(f"[OK] Created original_ir.json: {output_path}")

