"""Create LogicalIR from World database SQL schema and extract CSV files."""

import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))

from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec, ForeignKeySpec, SQLType
from nl2data.ir.constraint_ir import ConstraintSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.ir.generation import GenerationIR
from nl2data.utils.ir_io import save_ir_to_json


def mysql_type_to_sql_type(mysql_type: str) -> SQLType:
    """Convert MySQL type to our SQLType."""
    mysql_type = mysql_type.upper().strip()
    
    # Remove UNSIGNED, AUTO_INCREMENT modifiers
    mysql_type = re.sub(r'\s+UNSIGNED', '', mysql_type)
    mysql_type = re.sub(r'\s+AUTO_INCREMENT', '', mysql_type)
    
    # Extract base type (before parentheses)
    base_type = re.split(r'[\(\)]', mysql_type)[0].strip()
    
    # Map MySQL types to our types
    if base_type in ['TINYINT', 'SMALLINT', 'MEDIUMINT', 'INT', 'INTEGER', 'BIGINT']:
        return "INT"
    elif base_type in ['DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE', 'REAL']:
        return "FLOAT"
    elif base_type in ['DATE']:
        return "DATE"
    elif base_type in ['DATETIME', 'TIMESTAMP', 'TIME', 'YEAR']:
        return "DATETIME"
    elif base_type in ['BOOLEAN', 'BOOL', 'BIT']:
        return "BOOL"
    elif base_type in ['TEXT', 'TINYTEXT', 'MEDIUMTEXT', 'LONGTEXT', 'CHAR', 'VARCHAR', 
                       'BINARY', 'VARBINARY', 'BLOB', 'TINYBLOB', 'MEDIUMBLOB', 'LONGBLOB',
                       'ENUM', 'SET']:
        return "TEXT"
    else:
        return "TEXT"


def parse_create_table(sql_content: str) -> Dict[str, Dict]:
    """
    Parse CREATE TABLE statements from SQL content.
    
    Returns:
        Dict mapping table_name -> {
            'columns': [{'name': str, 'type': str, 'nullable': bool, 'default': Optional[str]}],
            'primary_key': [str],
            'foreign_keys': [{'column': str, 'ref_table': str, 'ref_column': str}]
        }
    """
    tables = {}
    
    print(f"  Parsing SQL content ({len(sql_content)} characters)...")
    
    # Find all CREATE TABLE statements
    create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?\s*\((.*?)\)\s*ENGINE\s*='
    matches = list(re.finditer(create_table_pattern, sql_content, re.IGNORECASE | re.DOTALL))
    
    if not matches:
        # Try alternative pattern without ENGINE requirement
        create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?\s*\((.*?)\)\s*(?:ENGINE|;|\n\n)'
        matches = list(re.finditer(create_table_pattern, sql_content, re.IGNORECASE | re.DOTALL))
    
    for match in matches:
        table_name = match.group(1)
        table_body = match.group(2)
        
        print(f"  Processing table: {table_name}")
        
        columns = []
        primary_key = []
        foreign_keys = []
        
        # Split by commas, but be careful with nested parentheses
        parts = []
        current = ""
        depth = 0
        for char in table_body:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        
        # Parse each part
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for PRIMARY KEY
            if re.match(r'PRIMARY\s+KEY', part, re.IGNORECASE):
                pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', part, re.IGNORECASE)
                if pk_match:
                    pk_cols = [c.strip().strip('`') for c in pk_match.group(1).split(',')]
                    primary_key.extend(pk_cols)
                continue
            
            # Check for FOREIGN KEY
            fk_match = re.search(
                r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+`?(\w+)`?\s*\(([^)]+)\)',
                part, re.IGNORECASE
            )
            if fk_match:
                fk_col = fk_match.group(1).strip().strip('`')
                ref_table = fk_match.group(2)
                ref_col = fk_match.group(3).strip().strip('`')
                foreign_keys.append({
                    'column': fk_col,
                    'ref_table': ref_table,
                    'ref_column': ref_col
                })
                continue
            
            # Parse column definition
            col_match = re.match(
                r'`?(\w+)`?\s+(\w+(?:\([^)]+\))?)\s*(.*)',
                part, re.IGNORECASE
            )
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                col_attrs = col_match.group(3)
                
                # Determine nullable
                nullable = 'NOT NULL' not in col_attrs.upper()
                
                # Determine SQL type
                sql_type = mysql_type_to_sql_type(col_type)
                
                columns.append({
                    'name': col_name,
                    'type': sql_type,
                    'nullable': nullable
                })
        
        tables[table_name] = {
            'columns': columns,
            'primary_key': primary_key,
            'foreign_keys': foreign_keys
        }
        
        print(f"    - {len(columns)} columns, {len(primary_key)} PK columns, {len(foreign_keys)} FKs")
    
    print(f"  Total tables parsed: {len(tables)}")
    return tables


def create_logical_ir_from_world_sql(schema_file: Path) -> LogicalIR:
    """
    Create LogicalIR from World SQL schema file.
    
    Args:
        schema_file: Path to world.sql
        
    Returns:
        LogicalIR object
    """
    print(f"Reading schema file: {schema_file}")
    with open(schema_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Parse tables
    parsed_tables = parse_create_table(sql_content)
    
    print(f"Converting {len(parsed_tables)} tables to LogicalIR format...")
    
    # Convert to LogicalIR format
    tables = {}
    for table_name, table_info in parsed_tables.items():
        columns = []
        for col_info in table_info['columns']:
            col = ColumnSpec(
                name=col_info['name'],
                sql_type=col_info['type'],
                nullable=col_info['nullable']
            )
            columns.append(col)
        
        foreign_keys = []
        for fk_info in table_info['foreign_keys']:
            fk = ForeignKeySpec(
                column=fk_info['column'],
                ref_table=fk_info['ref_table'],
                ref_column=fk_info['ref_column']
            )
            foreign_keys.append(fk)
        
        table = TableSpec(
            name=table_name,
            columns=columns,
            primary_key=table_info['primary_key'],
            foreign_keys=foreign_keys
        )
        tables[table_name] = table
    
    logical_ir = LogicalIR(
        tables=tables,
        constraints=ConstraintSpec(),
        schema_mode="oltp"
    )
    
    print(f"Created LogicalIR with {len(tables)} tables")
    return logical_ir


def parse_insert_statements(sql_file: Path) -> Dict[str, List[List[str]]]:
    """
    Parse INSERT statements from SQL file.
    
    Returns:
        Dict mapping table_name -> list of rows (each row is a list of values)
    """
    print(f"  Reading SQL file: {sql_file}")
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match INSERT INTO statements
    insert_pattern = r'INSERT\s+INTO\s+`?(\w+)`?\s+VALUES\s+(.+?)(?=INSERT\s+INTO|set\s+autocommit|$)'
    
    tables_data = {}
    matches = list(re.finditer(insert_pattern, content, re.IGNORECASE | re.DOTALL))
    
    for match in matches:
        table_name = match.group(1)
        values_str = match.group(2).strip()
        
        if table_name not in tables_data:
            tables_data[table_name] = []
        
        # Parse VALUES - handle multi-row inserts
        values_str = values_str.rstrip(';')
        rows = re.split(r'\)\s*,\s*\(', values_str)
        
        for row_str in rows:
            row_str = row_str.strip().strip('();')
            
            # Parse values - handle quoted strings, NULL, numbers, etc.
            values = []
            current_value = ""
            in_quotes = False
            quote_char = None
            i = 0
            
            while i < len(row_str):
                char = row_str[i]
                
                if not in_quotes:
                    if char in ("'", '"'):
                        in_quotes = True
                        quote_char = char
                    elif char == ',':
                        cleaned_value = current_value.strip()
                        if cleaned_value.startswith("'") and cleaned_value.endswith("'"):
                            cleaned_value = cleaned_value[1:-1]
                        elif cleaned_value.startswith('"') and cleaned_value.endswith('"'):
                            cleaned_value = cleaned_value[1:-1]
                        cleaned_value = cleaned_value.replace("\\'", "'").replace('\\"', '"')
                        values.append(cleaned_value)
                        current_value = ""
                    else:
                        current_value += char
                else:
                    if char == quote_char and (i == 0 or row_str[i-1] != '\\'):
                        in_quotes = False
                        quote_char = None
                    else:
                        current_value += char
                
                i += 1
            
            # Handle last value
            if current_value.strip():
                cleaned_value = current_value.strip()
                if cleaned_value.startswith("'") and cleaned_value.endswith("'"):
                    cleaned_value = cleaned_value[1:-1]
                elif cleaned_value.startswith('"') and cleaned_value.endswith('"'):
                    cleaned_value = cleaned_value[1:-1]
                cleaned_value = cleaned_value.replace("\\'", "'").replace('\\"', '"')
                values.append(cleaned_value)
            
            if values:
                tables_data[table_name].append(values)
    
    return tables_data


def create_csv_files(data_dir: Path, tables_data: Dict[str, List[List[str]]], logical_ir: LogicalIR):
    """
    Create CSV files from parsed INSERT data.
    
    Args:
        data_dir: Directory to save CSV files
        tables_data: Parsed data from INSERT statements
        logical_ir: LogicalIR to get column names
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Creating CSV files...")
    for table_name, rows in tables_data.items():
        if table_name not in logical_ir.tables:
            print(f"    Warning: Table {table_name} not found in IR, skipping")
            continue
        
        table = logical_ir.tables[table_name]
        column_names = [col.name for col in table.columns]
        csv_file = data_dir / f"{table_name}.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            
            for row in rows:
                # Ensure row has same number of columns
                if len(row) == len(column_names):
                    writer.writerow(row)
                elif len(row) > len(column_names):
                    writer.writerow(row[:len(column_names)])
                else:
                    padded_row = row + [''] * (len(column_names) - len(row))
                    writer.writerow(padded_row)
        
        print(f"    Created {csv_file.name} with {len(rows)} rows")


if __name__ == "__main__":
    print("="*60)
    print("World Database - Create LogicalIR and Extract CSV Files")
    print("="*60)
    
    # Get paths
    script_dir = Path(__file__).parent
    schema_file = script_dir / "world.sql"
    base_dir = script_dir.parent.parent  # Go up to realistic_datasets/
    data_dir = base_dir / "data" / "mysql" / "world"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not schema_file.exists():
        print(f"ERROR: {schema_file} not found")
        sys.exit(1)
    
    print(f"\nSchema file: {schema_file}")
    print(f"Output directory: {data_dir}")
    
    # Step 1: Create LogicalIR from SQL schema
    print("\n" + "="*60)
    print("Step 1: Creating LogicalIR from SQL schema...")
    print("="*60)
    logical_ir = create_logical_ir_from_world_sql(schema_file)
    
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
    
    print(f"\n✓ Saved original_ir.json to {output_file}")
    print(f"\nSummary:")
    print(f"  Total tables: {len(logical_ir.tables)}")
    for table_name in sorted(logical_ir.tables.keys()):
        table = logical_ir.tables[table_name]
        print(f"  - {table_name}: {len(table.columns)} columns, "
              f"PK: {table.primary_key}, FKs: {len(table.foreign_keys)}")
    
    # Step 3: Extract data from INSERT statements
    print("\n" + "="*60)
    print("Step 3: Extracting data from INSERT statements...")
    print("="*60)
    tables_data = parse_insert_statements(schema_file)
    
    print(f"\nFound data for {len(tables_data)} tables:")
    for table_name, rows in tables_data.items():
        print(f"  - {table_name}: {len(rows)} rows")
    
    # Step 4: Create CSV files
    print("\n" + "="*60)
    print("Step 4: Creating CSV files...")
    print("="*60)
    create_csv_files(data_dir, tables_data, logical_ir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
