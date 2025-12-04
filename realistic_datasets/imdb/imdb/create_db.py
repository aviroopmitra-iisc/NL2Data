"""Create LogicalIR from IMDB TSV files and extract CSV files."""

import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec, ForeignKeySpec, SQLType
from nl2data.ir.constraint_ir import ConstraintSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import save_ir_to_json


# IMDB schema definitions (based on standard IMDB dataset structure)
IMDB_SCHEMAS = {
    "name.basics": {
        "columns": [
            ("nconst", "TEXT", False),  # Primary key
            ("primaryName", "TEXT", True),
            ("birthYear", "TEXT", True),  # Can be \N for null
            ("deathYear", "TEXT", True),
            ("primaryProfession", "TEXT", True),
            ("knownForTitles", "TEXT", True),
        ],
        "primary_key": ["nconst"],
        "foreign_keys": []
    },
    "title.basics": {
        "columns": [
            ("tconst", "TEXT", False),  # Primary key
            ("titleType", "TEXT", True),
            ("primaryTitle", "TEXT", True),
            ("originalTitle", "TEXT", True),
            ("isAdult", "TEXT", True),  # 0 or 1
            ("startYear", "TEXT", True),
            ("endYear", "TEXT", True),
            ("runtimeMinutes", "TEXT", True),
            ("genres", "TEXT", True),
        ],
        "primary_key": ["tconst"],
        "foreign_keys": []
    },
    "title.akas": {
        "columns": [
            ("titleId", "TEXT", False),  # Part of composite PK
            ("ordering", "TEXT", False),  # Part of composite PK
            ("title", "TEXT", True),
            ("region", "TEXT", True),
            ("language", "TEXT", True),
            ("types", "TEXT", True),
            ("attributes", "TEXT", True),
            ("isOriginalTitle", "TEXT", True),
        ],
        "primary_key": ["titleId", "ordering"],
        "foreign_keys": [
            {"column": "titleId", "ref_table": "title.basics", "ref_column": "tconst"}
        ]
    },
    "title.crew": {
        "columns": [
            ("tconst", "TEXT", False),  # Primary key
            ("directors", "TEXT", True),
            ("writers", "TEXT", True),
        ],
        "primary_key": ["tconst"],
        "foreign_keys": [
            {"column": "tconst", "ref_table": "title.basics", "ref_column": "tconst"}
        ]
    },
    "title.episode": {
        "columns": [
            ("tconst", "TEXT", False),  # Primary key
            ("parentTconst", "TEXT", True),
            ("seasonNumber", "TEXT", True),
            ("episodeNumber", "TEXT", True),
        ],
        "primary_key": ["tconst"],
        "foreign_keys": [
            {"column": "tconst", "ref_table": "title.basics", "ref_column": "tconst"},
            {"column": "parentTconst", "ref_table": "title.basics", "ref_column": "tconst"}
        ]
    },
    "title.principals": {
        "columns": [
            ("tconst", "TEXT", False),  # Part of composite PK
            ("ordering", "TEXT", False),  # Part of composite PK
            ("nconst", "TEXT", True),
            ("category", "TEXT", True),
            ("job", "TEXT", True),
            ("characters", "TEXT", True),
        ],
        "primary_key": ["tconst", "ordering"],
        "foreign_keys": [
            {"column": "tconst", "ref_table": "title.basics", "ref_column": "tconst"},
            {"column": "nconst", "ref_table": "name.basics", "ref_column": "nconst"}
        ]
    },
    "title.ratings": {
        "columns": [
            ("tconst", "TEXT", False),  # Primary key
            ("averageRating", "TEXT", True),
            ("numVotes", "TEXT", True),
        ],
        "primary_key": ["tconst"],
        "foreign_keys": [
            {"column": "tconst", "ref_table": "title.basics", "ref_column": "tconst"}
        ]
    },
}


def infer_sql_type_from_data(values: List[str]) -> SQLType:
    """Infer SQL type from a sample of values."""
    if not values:
        return "TEXT"
    
    # Check for numeric types
    numeric_count = 0
    float_count = 0
    for val in values[:100]:  # Sample first 100 values
        if val and val != "\\N" and val.strip():
            try:
                float_val = float(val)
                float_count += 1
                if float_val == int(float_val):
                    numeric_count += 1
            except (ValueError, TypeError):
                pass
    
    if numeric_count > len(values) * 0.8:
        return "INT"
    elif float_count > len(values) * 0.8:
        return "FLOAT"
    
    # Check for date/datetime patterns
    date_patterns = ["-", "/"]
    if any(any(p in str(v) for p in date_patterns) for v in values[:10] if v and v != "\\N"):
        # Could be date, but IMDB uses text for years, so keep as TEXT
        pass
    
    return "TEXT"


def create_logical_ir_from_imdb_tsv(tsv_dir: Path) -> LogicalIR:
    """
    Create LogicalIR from IMDB TSV files.
    
    Args:
        tsv_dir: Directory containing IMDB TSV files
        
    Returns:
        LogicalIR object
    """
    print(f"Reading TSV files from: {tsv_dir}")
    
    tables = {}
    
    for table_name, schema_info in IMDB_SCHEMAS.items():
        tsv_file = tsv_dir / f"{table_name}.tsv"
        
        if not tsv_file.exists():
            print(f"  Warning: {tsv_file.name} not found, skipping")
            continue
        
        print(f"  Processing {table_name}...")
        
        # Read header to verify columns
        with open(tsv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
        
        # Build columns from schema
        columns = []
        for col_name, sql_type_str, nullable in schema_info["columns"]:
            # Map string type to SQLType
            if sql_type_str == "INT":
                sql_type: SQLType = "INT"
            elif sql_type_str == "FLOAT":
                sql_type = "FLOAT"
            elif sql_type_str in ["DATE", "DATETIME"]:
                sql_type = sql_type_str
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
        schema_mode="oltp"
    )
    
    print(f"Created LogicalIR with {len(tables)} tables")
    return logical_ir


def convert_tsv_to_csv(tsv_dir: Path, data_dir: Path, logical_ir: LogicalIR):
    """
    Convert TSV files to CSV files.
    
    Args:
        tsv_dir: Directory containing TSV files
        data_dir: Directory to save CSV files (data/imdb/imdb/)
        logical_ir: LogicalIR to get table names
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Converting TSV files to CSV...")
    
    for table_name in logical_ir.tables.keys():
        tsv_file = tsv_dir / f"{table_name}.tsv"
        
        if not tsv_file.exists():
            print(f"    Warning: {tsv_file.name} not found, skipping")
            continue
        
        csv_file = data_dir / f"{table_name}.csv"
        
        # Read TSV and write CSV
        # Use pandas for better handling of large files and encoding
        try:
            # Read in chunks to handle large files
            chunk_size = 10000
            first_chunk = True
            
            for chunk in pd.read_csv(tsv_file, sep='\t', dtype=str, chunksize=chunk_size, 
                                     encoding='utf-8', na_values=['\\N'], keep_default_na=False):
                # Replace \N with empty string for CSV
                chunk = chunk.fillna('')
                
                # Write header only for first chunk
                chunk.to_csv(csv_file, mode='w' if first_chunk else 'a', 
                           index=False, header=first_chunk, lineterminator='\n')
                first_chunk = False
            
            # Count rows
            row_count = sum(1 for _ in open(csv_file, 'r', encoding='utf-8')) - 1  # Subtract header
            print(f"    Created {csv_file.name} with {row_count:,} rows")
            
        except Exception as e:
            print(f"    Error converting {tsv_file.name}: {e}")
            # Fallback to basic CSV conversion
            try:
                with open(tsv_file, 'r', encoding='utf-8', newline='') as tsv_in:
                    with open(csv_file, 'w', encoding='utf-8', newline='') as csv_out:
                        reader = csv.reader(tsv_in, delimiter='\t')
                        writer = csv.writer(csv_out)
                        
                        row_count = 0
                        for row in reader:
                            # Replace \N with empty string
                            row = ['' if val == '\\N' else val for val in row]
                            writer.writerow(row)
                            row_count += 1
                        
                        print(f"    Created {csv_file.name} with {row_count-1:,} rows (header excluded)")
            except Exception as e2:
                print(f"    Failed to convert {tsv_file.name}: {e2}")


if __name__ == "__main__":
    print("="*60)
    print("IMDB Dataset - Create LogicalIR and Extract CSV Files")
    print("="*60)
    
    # Get paths
    script_dir = Path(__file__).parent  # imdb/imdb/
    tsv_dir = script_dir  # TSV files are in the same directory as the script
    base_dir = script_dir.parent.parent  # Go up to realistic_datasets/
    data_dir = base_dir / "data" / "imdb" / "imdb"  # data/imdb/imdb/
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not tsv_dir.exists():
        print(f"ERROR: {tsv_dir} not found")
        sys.exit(1)
    
    print(f"\nTSV directory: {tsv_dir}")
    print(f"Output directory: {data_dir}")
    
    # Step 1: Create LogicalIR from TSV files
    print("\n" + "="*60)
    print("Step 1: Creating LogicalIR from TSV files...")
    print("="*60)
    logical_ir = create_logical_ir_from_imdb_tsv(tsv_dir)
    
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
    
    # Step 3: Convert TSV to CSV
    print("\n" + "="*60)
    print("Step 3: Converting TSV files to CSV...")
    print("="*60)
    convert_tsv_to_csv(tsv_dir, data_dir, logical_ir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

