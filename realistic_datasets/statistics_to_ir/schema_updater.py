"""Schema update logic for adding candidate keys and primary keys."""

from typing import Dict, List, Tuple
import pandas as pd
import sys
from pathlib import Path
import json

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR, TableSpec
from nl2data.ir.constraint_ir import FDConstraint, ConstraintSpec, TableFDConstraint
from nl2data.utils.ir_io import load_ir_from_json, save_ir_to_json
from .candidate_key_discovery import (
    find_candidate_keys,
    find_minimal_candidate_keys
)
from .llm_assistant import LLMClient


def process_candidate_keys_and_update_schema(
    dfs: Dict[str, pd.DataFrame],
    logical_ir: LogicalIR,
    discovered_fds: List[FDConstraint],
    support_confidence_map: Dict[tuple, tuple],
    llm_client: LLMClient = None,
    min_support: float = 1.0,
    min_confidence: float = 1.0
) -> Tuple[LogicalIR, List[FDConstraint]]:
    """
    Process candidate keys, select primary keys, and update schema.
    
    Args:
        dfs: Dictionary of table_name -> DataFrame
        logical_ir: LogicalIR schema to update
        discovered_fds: List of all discovered FDs
        support_confidence_map: Dict mapping (table, tuple(lhs), rhs) -> (support, confidence)
        llm_client: Optional LLM client for primary key selection
        min_support: Minimum support for candidate keys (default 1.0)
        min_confidence: Minimum confidence for candidate keys (default 1.0)
        
    Returns:
        Tuple of (updated_logical_ir, regular_fds)
        regular_fds: FDs that are not candidate-key FDs
    """
    # Create a copy of the logical IR to modify
    updated_ir = logical_ir.model_copy(deep=True)
    all_regular_fds = []
    
    # Process each table
    for table_name, df in dfs.items():
        if table_name not in updated_ir.tables:
            continue
        
        table = updated_ir.tables[table_name]
        
        # Check if primary key is already defined from DDL
        existing_pk = None
        if table.primary_key:
            existing_pk = table.primary_key
        else:
            # Check column roles for primary_key
            pk_cols = [col.name for col in table.columns if col.role == "primary_key"]
            if pk_cols:
                existing_pk = pk_cols if len(pk_cols) > 1 else pk_cols[0]
        
        # Update role fields for existing primary keys from DDL
        if existing_pk:
            pk_list = existing_pk if isinstance(existing_pk, list) else [existing_pk]
            for col_name in pk_list:
                for col in table.columns:
                    if col.name == col_name:
                        col.role = "primary_key"
                        col.nullable = False
                        # Only set unique=True for single-column primary keys
                        if len(pk_list) == 1:
                            col.unique = True
                        break
            
            # Only store multi-column primary keys in primary_key list
            if len(pk_list) > 1:
                table.primary_key = pk_list
            else:
                table.primary_key = []
            
            print(f"  [{table_name}] Primary key from DDL: {pk_list}")
        
        # Find candidate keys (excluding the primary key if it exists)
        # Candidate keys are other sets of columns that could uniquely identify rows
        candidate_keys = find_candidate_keys(
            df, table_name, logical_ir, discovered_fds,
            support_confidence_map, min_support, min_confidence
        )
        
        # Remove primary key from candidate keys (if it exists)
        if existing_pk:
            pk_set = set(existing_pk if isinstance(existing_pk, list) else [existing_pk])
            candidate_keys = [
                ck for ck in candidate_keys 
                if set(ck) != pk_set
            ]
        
        # Build list of all keys (primary key + candidate keys) for FD filtering
        all_keys = []
        if existing_pk:
            pk_list = existing_pk if isinstance(existing_pk, list) else [existing_pk]
            all_keys.append(pk_list)
        
        if candidate_keys:
            # Find minimal candidate keys
            minimal_candidate_keys = find_minimal_candidate_keys(candidate_keys)
            all_keys.extend(minimal_candidate_keys)
            
            # Store candidate keys (excluding single-column ones, which are just unique columns)
            multi_col_candidate_keys = [
                ck for ck in minimal_candidate_keys if len(ck) > 1
            ]
            if multi_col_candidate_keys:
                table.candidate_keys = multi_col_candidate_keys
                print(f"  [{table_name}] Found {len(multi_col_candidate_keys)} candidate key set(s)")
            
            # Mark single-column candidate keys as unique (but not as primary keys)
            for ck in minimal_candidate_keys:
                if len(ck) == 1:
                    col_name = ck[0]
                    # Don't mark if it's already the primary key
                    if existing_pk and col_name in (existing_pk if isinstance(existing_pk, list) else [existing_pk]):
                        continue
                    for col in table.columns:
                        if col.name == col_name:
                            col.unique = True
                            break
        else:
            # No additional candidate keys found beyond the primary key
            table.candidate_keys = []
        
        # Get all FDs for this table and filter out those implied by keys
        table_fds_list = [fd for fd in discovered_fds if fd.table == table_name]
        
        # Filter FDs: remove those where LHS is subset/superset of any key (primary key or candidate key)
        filtered_regular_fds = []
        filtered_count = 0
        for fd in table_fds_list:
            fd_lhs_set = set(fd.lhs)
            # Check if LHS is a subset or superset of any key
            is_implied = False
            for key in all_keys:
                key_set = set(key)
                if fd_lhs_set.issubset(key_set) or key_set.issubset(fd_lhs_set):
                    # LHS is a subset or superset of a key, so this FD is implied
                    is_implied = True
                    filtered_count += 1
                    break
            
            if not is_implied:
                filtered_regular_fds.append(fd)
        
        if filtered_count > 0:
            print(f"  [{table_name}] Filtered out {filtered_count} FDs that are subsets/supersets of candidate/primary keys")
        
        all_regular_fds.extend(filtered_regular_fds)
        
        # Store FDs per table (convert FDConstraint to TableFDConstraint)
        table_fds = [
            TableFDConstraint(lhs=fd.lhs, rhs=fd.rhs, mode=fd.mode)
            for fd in filtered_regular_fds
            if fd.table == table_name
        ]
        table.fds = table_fds
    
    return updated_ir, all_regular_fds


def update_original_ir_file(
    original_ir_path: Path,
    updated_logical_ir: LogicalIR
) -> None:
    """
    Update original_ir.json file with new schema (candidate keys, primary keys, FDs).
    
    Args:
        original_ir_path: Path to original_ir.json
        updated_logical_ir: Updated LogicalIR with candidate keys and primary keys
    """
    # Load existing IR (might be DatasetIR or just LogicalIR)
    try:
        existing_ir = load_ir_from_json(original_ir_path)
        
        # Check if it's a DatasetIR or LogicalIR
        if hasattr(existing_ir, 'logical'):
            # It's a DatasetIR, update the logical part
            existing_ir.logical = updated_logical_ir
        else:
            # It's just a LogicalIR, replace it
            existing_ir = updated_logical_ir
        
        # Save updated IR
        save_ir_to_json(existing_ir, original_ir_path)
        
    except Exception as e:
        print(f"[WARNING] Failed to update original_ir.json: {e}")
        # Try to save just the LogicalIR
        try:
            save_ir_to_json(updated_logical_ir, original_ir_path)
        except Exception as e2:
            print(f"[ERROR] Failed to save LogicalIR: {e2}")

