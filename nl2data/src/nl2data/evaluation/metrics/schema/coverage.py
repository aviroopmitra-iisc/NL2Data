"""Schema coverage metrics."""

from typing import Dict
from nl2data.ir.logical import LogicalIR


def schema_coverage(
    gold_ir: LogicalIR,
    synth_ir: LogicalIR,
) -> Dict[str, float]:
    """
    Compute coverage metrics comparing synthetic schema to gold schema.
    
    Uses exact name matching (for backward compatibility).
    For fuzzy matching, use the matching layer instead.
    
    Args:
        gold_ir: Gold/reference LogicalIR
        synth_ir: Synthetic LogicalIR to evaluate
        
    Returns:
        Dictionary with precision, recall, F1 for tables, columns, PKs, FKs
    """
    gold_tables = set(gold_ir.tables.keys())
    synth_tables = set(synth_ir.tables.keys())
    
    # Table-level metrics
    table_intersection = gold_tables & synth_tables
    table_precision = len(table_intersection) / len(synth_tables) if synth_tables else 0.0
    table_recall = len(table_intersection) / len(gold_tables) if gold_tables else 0.0
    table_f1 = (
        2 * table_precision * table_recall / (table_precision + table_recall)
        if (table_precision + table_recall) > 0
        else 0.0
    )
    
    # Column-level metrics (for common tables)
    common_tables = table_intersection
    total_gold_cols = 0
    total_synth_cols = 0
    total_common_cols = 0
    
    for table_name in common_tables:
        gold_table = gold_ir.tables[table_name]
        synth_table = synth_ir.tables[table_name]
        
        gold_cols = {c.name for c in gold_table.columns}
        synth_cols = {c.name for c in synth_table.columns}
        
        total_gold_cols += len(gold_cols)
        total_synth_cols += len(synth_cols)
        total_common_cols += len(gold_cols & synth_cols)
    
    col_precision = total_common_cols / total_synth_cols if total_synth_cols > 0 else 0.0
    col_recall = total_common_cols / total_gold_cols if total_gold_cols > 0 else 0.0
    col_f1 = (
        2 * col_precision * col_recall / (col_precision + col_recall)
        if (col_precision + col_recall) > 0
        else 0.0
    )
    
    # PK/FK metrics
    gold_pks = {
        (tname, tuple(t.primary_key)) for tname, t in gold_ir.tables.items() if t.primary_key
    }
    synth_pks = {
        (tname, tuple(t.primary_key)) for tname, t in synth_ir.tables.items() if t.primary_key
    }
    pk_intersection = gold_pks & synth_pks
    pk_precision = len(pk_intersection) / len(synth_pks) if synth_pks else 0.0
    pk_recall = len(pk_intersection) / len(gold_pks) if gold_pks else 0.0
    pk_f1 = (
        2 * pk_precision * pk_recall / (pk_precision + pk_recall)
        if (pk_precision + pk_recall) > 0
        else 0.0
    )
    
    # FK edges
    gold_fks = set()
    for tname, table in gold_ir.tables.items():
        for fk in table.foreign_keys:
            gold_fks.add((tname, fk.column, fk.ref_table, fk.ref_column))
    
    synth_fks = set()
    for tname, table in synth_ir.tables.items():
        for fk in table.foreign_keys:
            synth_fks.add((tname, fk.column, fk.ref_table, fk.ref_column))
    
    fk_intersection = gold_fks & synth_fks
    fk_precision = len(fk_intersection) / len(synth_fks) if synth_fks else 0.0
    fk_recall = len(fk_intersection) / len(gold_fks) if gold_fks else 0.0
    fk_f1 = (
        2 * fk_precision * fk_recall / (fk_precision + fk_recall)
        if (fk_precision + fk_recall) > 0
        else 0.0
    )
    
    return {
        "table_precision": table_precision,
        "table_recall": table_recall,
        "table_f1": table_f1,
        "column_precision": col_precision,
        "column_recall": col_recall,
        "column_f1": col_f1,
        "pk_precision": pk_precision,
        "pk_recall": pk_recall,
        "pk_f1": pk_f1,
        "fk_precision": fk_precision,
        "fk_recall": fk_recall,
        "fk_f1": fk_f1,
    }


def compute_coverage_factors(
    table_matches: Dict[str, str],
    column_matches: Dict[str, Dict[str, str]],
    real_ir: LogicalIR,
) -> Dict[str, float]:
    """
    Compute coverage factors for penalty application.
    
    Args:
        table_matches: Dictionary mapping real_table -> synth_table
        column_matches: Dictionary mapping table_name -> {real_col -> synth_col}
        real_ir: Real LogicalIR
        
    Returns:
        Dictionary with:
        - table_coverage: C_T
        - column_coverage: Dict[table_name, C_k]
    """
    total_real_tables = len(real_ir.tables)
    table_coverage = len(table_matches) / total_real_tables if total_real_tables > 0 else 0.0
    
    column_coverage = {}
    for table_name, table in real_ir.tables.items():
        if table_name in table_matches:
            total_cols = len(table.columns)
            matched_cols = len(column_matches.get(table_name, {}))
            column_coverage[table_name] = matched_cols / total_cols if total_cols > 0 else 0.0
        else:
            column_coverage[table_name] = 0.0
    
    return {
        "table_coverage": table_coverage,
        "column_coverage": column_coverage,
    }
