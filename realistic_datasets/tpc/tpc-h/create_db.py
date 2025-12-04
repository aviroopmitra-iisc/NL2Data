"""Create LogicalIR from TPC-H TBL files and extract CSV files."""

import csv
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec, ForeignKeySpec, SQLType
from nl2data.ir.constraint_ir import ConstraintSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import save_ir_to_json


# TPC-H schema definitions (standard TPC-H benchmark schema)
TPCH_SCHEMAS = {
    "region": {
        "columns": [
            ("R_REGIONKEY", "INT", False),  # Primary key
            ("R_NAME", "TEXT", False),
            ("R_COMMENT", "TEXT", True),
        ],
        "primary_key": ["R_REGIONKEY"],
        "foreign_keys": []
    },
    "nation": {
        "columns": [
            ("N_NATIONKEY", "INT", False),  # Primary key
            ("N_NAME", "TEXT", False),
            ("N_REGIONKEY", "INT", False),  # Foreign key
            ("N_COMMENT", "TEXT", True),
        ],
        "primary_key": ["N_NATIONKEY"],
        "foreign_keys": [
            {"column": "N_REGIONKEY", "ref_table": "region", "ref_column": "R_REGIONKEY"}
        ]
    },
    "supplier": {
        "columns": [
            ("S_SUPPKEY", "INT", False),  # Primary key
            ("S_NAME", "TEXT", False),
            ("S_ADDRESS", "TEXT", False),
            ("S_NATIONKEY", "INT", False),  # Foreign key
            ("S_PHONE", "TEXT", False),
            ("S_ACCTBAL", "FLOAT", False),
            ("S_COMMENT", "TEXT", False),
        ],
        "primary_key": ["S_SUPPKEY"],
        "foreign_keys": [
            {"column": "S_NATIONKEY", "ref_table": "nation", "ref_column": "N_NATIONKEY"}
        ]
    },
    "customer": {
        "columns": [
            ("C_CUSTKEY", "INT", False),  # Primary key
            ("C_NAME", "TEXT", False),
            ("C_ADDRESS", "TEXT", False),
            ("C_NATIONKEY", "INT", False),  # Foreign key
            ("C_PHONE", "TEXT", False),
            ("C_ACCTBAL", "FLOAT", False),
            ("C_MKTSEGMENT", "TEXT", False),
            ("C_COMMENT", "TEXT", False),
        ],
        "primary_key": ["C_CUSTKEY"],
        "foreign_keys": [
            {"column": "C_NATIONKEY", "ref_table": "nation", "ref_column": "N_NATIONKEY"}
        ]
    },
    "part": {
        "columns": [
            ("P_PARTKEY", "INT", False),  # Primary key
            ("P_NAME", "TEXT", False),
            ("P_MFGR", "TEXT", False),
            ("P_BRAND", "TEXT", False),
            ("P_TYPE", "TEXT", False),
            ("P_SIZE", "INT", False),
            ("P_CONTAINER", "TEXT", False),
            ("P_RETAILPRICE", "FLOAT", False),
            ("P_COMMENT", "TEXT", False),
        ],
        "primary_key": ["P_PARTKEY"],
        "foreign_keys": []
    },
    "partsupp": {
        "columns": [
            ("PS_PARTKEY", "INT", False),  # Part of composite PK
            ("PS_SUPPKEY", "INT", False),  # Part of composite PK
            ("PS_AVAILQTY", "INT", False),
            ("PS_SUPPLYCOST", "FLOAT", False),
            ("PS_COMMENT", "TEXT", False),
        ],
        "primary_key": ["PS_PARTKEY", "PS_SUPPKEY"],
        "foreign_keys": [
            {"column": "PS_PARTKEY", "ref_table": "part", "ref_column": "P_PARTKEY"},
            {"column": "PS_SUPPKEY", "ref_table": "supplier", "ref_column": "S_SUPPKEY"}
        ]
    },
    "orders": {
        "columns": [
            ("O_ORDERKEY", "INT", False),  # Primary key
            ("O_CUSTKEY", "INT", False),  # Foreign key
            ("O_ORDERSTATUS", "TEXT", False),
            ("O_TOTALPRICE", "FLOAT", False),
            ("O_ORDERDATE", "DATE", False),
            ("O_ORDERPRIORITY", "TEXT", False),
            ("O_CLERK", "TEXT", False),
            ("O_SHIPPRIORITY", "INT", False),
            ("O_COMMENT", "TEXT", False),
        ],
        "primary_key": ["O_ORDERKEY"],
        "foreign_keys": [
            {"column": "O_CUSTKEY", "ref_table": "customer", "ref_column": "C_CUSTKEY"}
        ]
    },
    "lineitem": {
        "columns": [
            ("L_ORDERKEY", "INT", False),  # Part of composite PK
            ("L_PARTKEY", "INT", False),  # Foreign key (to part)
            ("L_SUPPKEY", "INT", False),  # Foreign key (to supplier)
            ("L_LINENUMBER", "INT", False),  # Part of composite PK
            ("L_QUANTITY", "FLOAT", False),
            ("L_EXTENDEDPRICE", "FLOAT", False),
            ("L_DISCOUNT", "FLOAT", False),
            ("L_TAX", "FLOAT", False),
            ("L_RETURNFLAG", "TEXT", False),
            ("L_LINESTATUS", "TEXT", False),
            ("L_SHIPDATE", "DATE", False),
            ("L_COMMITDATE", "DATE", False),
            ("L_RECEIPTDATE", "DATE", False),
            ("L_SHIPINSTRUCT", "TEXT", False),
            ("L_SHIPMODE", "TEXT", False),
            ("L_COMMENT", "TEXT", False),
        ],
        "primary_key": ["L_ORDERKEY", "L_LINENUMBER"],
        "foreign_keys": [
            {"column": "L_ORDERKEY", "ref_table": "orders", "ref_column": "O_ORDERKEY"},
            {"column": "L_PARTKEY", "ref_table": "part", "ref_column": "P_PARTKEY"},
            {"column": "L_SUPPKEY", "ref_table": "supplier", "ref_column": "S_SUPPKEY"}
        ]
    },
}


def create_logical_ir_from_tpch_schema() -> LogicalIR:
    """
    Create LogicalIR from TPC-H schema definition.
    
    Returns:
        LogicalIR object
    """
    print(f"Creating LogicalIR from TPC-H schema...")
    
    tables = {}
    
    for table_name, schema_info in TPCH_SCHEMAS.items():
        print(f"  Processing {table_name}...")
        
        # Build columns
        columns = []
        for col_name, sql_type_str, nullable in schema_info["columns"]:
            # Map string type to SQLType
            if sql_type_str == "INT":
                sql_type: SQLType = "INT"
            elif sql_type_str == "FLOAT":
                sql_type = "FLOAT"
            elif sql_type_str == "DATE":
                sql_type = "DATE"
            elif sql_type_str == "DATETIME":
                sql_type = "DATETIME"
            else:
                sql_type = "TEXT"
            
            col = ColumnSpec(
                name=col_name,
                sql_type=sql_type,
                nullable=nullable
            )
            columns.append(col)
        
        # Build foreign keys
        foreign_keys = []
        for fk_info in schema_info["foreign_keys"]:
            fk = ForeignKeySpec(
                column=fk_info["column"],
                ref_table=fk_info["ref_table"],
                ref_column=fk_info["ref_column"]
            )
            foreign_keys.append(fk)
        
        table = TableSpec(
            name=table_name,
            columns=columns,
            primary_key=schema_info["primary_key"],
            foreign_keys=foreign_keys
        )
        tables[table_name] = table
        
        print(f"    - {len(columns)} columns, PK: {schema_info['primary_key']}, {len(foreign_keys)} FKs")
    
    logical_ir = LogicalIR(
        tables=tables,
        constraints=ConstraintSpec(),
        schema_mode="star"  # TPC-H is a star schema
    )
    
    print(f"Created LogicalIR with {len(tables)} tables")
    return logical_ir


def convert_tbl_to_csv(tbl_dir: Path, data_dir: Path, logical_ir: LogicalIR):
    """
    Convert TBL files (pipe-delimited) to CSV files.
    
    Args:
        tbl_dir: Directory containing TBL files
        data_dir: Directory to save CSV files (data/tpc/tpc-h/)
        logical_ir: LogicalIR to get table names
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Converting TBL files to CSV...")
    
    for table_name in logical_ir.tables.keys():
        tbl_file = tbl_dir / f"{table_name}.tbl"
        
        if not tbl_file.exists():
            print(f"    Warning: {tbl_file.name} not found, skipping")
            continue
        
        csv_file = data_dir / f"{table_name}.csv"
        table = logical_ir.tables[table_name]
        column_names = [col.name for col in table.columns]
        
        print(f"    Converting {tbl_file.name}...")
        
        try:
            row_count = 0
            with open(tbl_file, 'r', encoding='utf-8') as tbl_in:
                with open(csv_file, 'w', encoding='utf-8', newline='') as csv_out:
                    writer = csv.writer(csv_out)
                    writer.writerow(column_names)
                    
                    for line in tbl_in:
                        # TPC-H TBL files are pipe-delimited and end with |
                        # Remove trailing | and split
                        line = line.rstrip('|\n')
                        if not line:
                            continue
                        
                        values = line.split('|')
                        # Ensure we have the right number of columns
                        if len(values) == len(column_names):
                            writer.writerow(values)
                            row_count += 1
                        elif len(values) > len(column_names):
                            # Truncate if too many
                            writer.writerow(values[:len(column_names)])
                            row_count += 1
                        else:
                            # Pad if too few
                            padded = values + [''] * (len(column_names) - len(values))
                            writer.writerow(padded)
                            row_count += 1
            
            print(f"    Created {csv_file.name} with {row_count:,} rows")
            
        except Exception as e:
            print(f"    Error converting {tbl_file.name}: {e}")


if __name__ == "__main__":
    print("="*60)
    print("TPC-H Dataset - Create LogicalIR and Extract CSV Files")
    print("="*60)
    
    # Get paths
    script_dir = Path(__file__).parent  # tpc/tpc-h/
    tbl_dir = script_dir
    base_dir = script_dir.parent.parent  # Go up to realistic_datasets/
    data_dir = base_dir / "data" / "tpc" / "tpc-h"  # data/tpc/tpc-h/
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not tbl_dir.exists():
        print(f"ERROR: {tbl_dir} not found")
        sys.exit(1)
    
    print(f"\nTBL directory: {tbl_dir}")
    print(f"Output directory: {data_dir}")
    
    # Step 1: Create LogicalIR from schema
    print("\n" + "="*60)
    print("Step 1: Creating LogicalIR from TPC-H schema...")
    print("="*60)
    logical_ir = create_logical_ir_from_tpch_schema()
    
    # Step 2: Save LogicalIR to DatasetIR
    print("\n" + "="*60)
    print("Step 2: Saving LogicalIR to original_ir.json...")
    print("="*60)
    dataset_ir = DatasetIR(
        logical=logical_ir,
        generation=GenerationIR(columns=[]),
        workload=None
    )
    output_file = data_dir / "original_ir.json"
    save_ir_to_json(dataset_ir, output_file)
    
    print(f"\n[OK] Saved original_ir.json to {output_file}")
    print(f"\nSummary:")
    print(f"  Total tables: {len(logical_ir.tables)}")
    for table_name in sorted(logical_ir.tables.keys()):
        table = logical_ir.tables[table_name]
        print(f"  - {table_name}: {len(table.columns)} columns, "
              f"PK: {table.primary_key}, FKs: {len(table.foreign_keys)}")
    
    # Step 3: Convert TBL to CSV
    print("\n" + "="*60)
    print("Step 3: Converting TBL files to CSV...")
    print("="*60)
    convert_tbl_to_csv(tbl_dir, data_dir, logical_ir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

