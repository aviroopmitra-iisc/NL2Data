"""Create LogicalIR from World Bank data, similar to OpenML middleware."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec
from nl2data.ir.constraint_ir import ConstraintSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import save_ir_to_json


def infer_sql_type_from_series(series: pd.Series) -> str:
    """Infer SQL type from pandas Series."""
    dtype = series.dtype
    
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOL"
    else:
        return "TEXT"


def create_logical_ir_from_worldbank_data(
    dataset_dir: Path,
    table_name: str = "main"
) -> LogicalIR:
    """
    Create LogicalIR from World Bank indicator data.
    
    Args:
        dataset_dir: Directory containing CSV files and metadata.json
        table_name: Name for the main table
        
    Returns:
        LogicalIR object
    """
    # Load metadata if available
    metadata_path = dataset_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Find all CSV files
    csv_files = list(dataset_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_dir}")
    
    # For World Bank data, we'll combine all indicators into one table
    # or create separate tables per indicator
    # Let's combine them into one table for simplicity
    
    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Add indicator code as a column if not present
        indicator_code = csv_file.stem
        if "indicator_code" not in df.columns:
            df["indicator_code"] = indicator_code
        all_dfs.append(df)
    
    # Combine all dataframes
    if len(all_dfs) > 1:
        # Find common columns
        common_cols = set(all_dfs[0].columns)
        for df in all_dfs[1:]:
            common_cols &= set(df.columns)
        
        # Combine on common columns
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = all_dfs[0]
    
    # Create columns
    columns = []
    for col_name in combined_df.columns:
        col_series = combined_df[col_name]
        sql_type = infer_sql_type_from_series(col_series)
        
        # Check if nullable
        is_nullable = col_series.isnull().any()
        
        column = ColumnSpec(
            name=col_name,
            sql_type=sql_type,
            nullable=is_nullable
        )
        columns.append(column)
    
    # Create table
    table = TableSpec(
        name=table_name,
        columns=columns,
        primary_key=[],  # World Bank data typically doesn't have explicit PKs
        foreign_keys=[],
        constraints=[]  # No FDs for World Bank data
    )
    
    # Create LogicalIR
    logical_ir = LogicalIR(
        tables={table_name: table},
        constraints=ConstraintSpec()  # No constraints for World Bank data
    )
    
    return logical_ir


def create_dataset_ir_from_worldbank_data(
    dataset_dir: Path,
    table_name: str = "main"
) -> DatasetIR:
    """
    Create a complete DatasetIR from World Bank indicator data.
    
    This includes:
    - LogicalIR (schema)
    - Empty GenerationIR (no generation specs needed for original)
    - No WorkloadIR (optional)
    
    Args:
        dataset_dir: Directory containing CSV files and metadata.json
        table_name: Name for the main table
        
    Returns:
        DatasetIR instance
    """
    logical_ir = create_logical_ir_from_worldbank_data(
        dataset_dir, table_name
    )
    
    # Create minimal DatasetIR
    dataset_ir = DatasetIR(
        logical=logical_ir,
        generation=GenerationIR(columns=[]),  # No generation specs for original
        workload=None  # No workload specs
    )
    
    return dataset_ir


if __name__ == "__main__":
    # Example: Create DatasetIR for India indicators
    # Data is in data/worldbank subdirectory
    data_dir = Path(__file__).parent.parent / "data" / "worldbank"
    dataset_dir = data_dir / "india_indicators"
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        print("Please run fetch_worldbank_data.py first")
        sys.exit(1)
    
    print(f"Creating DatasetIR from {dataset_dir}...")
    dataset_ir = create_dataset_ir_from_worldbank_data(dataset_dir)
    
    # Save DatasetIR
    output_path = dataset_dir / "original_ir.json"
    save_ir_to_json(dataset_ir, output_path)
    
    print(f"Saved DatasetIR to {output_path}")
    print(f"Tables: {list(dataset_ir.logical.tables.keys())}")
    for table_name, table in dataset_ir.logical.tables.items():
        print(f"  {table_name}: {len(table.columns)} columns")

