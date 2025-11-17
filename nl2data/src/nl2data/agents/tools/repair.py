"""Automatic repair functions for common validation issues."""

from typing import List
from nl2data.ir.validators import QaIssue
from nl2data.ir.generation import GenerationIR, ColumnGenSpec
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def repair_missing_gen_specs(
    logical_ir: LogicalIR, generation_ir: GenerationIR
) -> GenerationIR:
    """
    Repair MISSING_GEN_SPEC issues by auto-generating missing specs.
    
    Args:
        logical_ir: LogicalIR to get all columns from
        generation_ir: Current GenerationIR with missing specs
    
    Returns:
        Updated GenerationIR with missing specs added
    """
    existing_specs = {(cg.table, cg.column) for cg in generation_ir.columns}
    
    new_specs = []
    for table_name, table in logical_ir.tables.items():
        for col in table.columns:
            if (table_name, col.name) not in existing_specs:
                # Create default spec
                from nl2data.ir.generation import DistUniform, DistCategorical, DistSeasonal
                
                # Determine default distribution based on type
                if col.sql_type == "INT" or col.sql_type == "FLOAT":
                    dist = DistUniform(low=0.0, high=100.0)
                elif col.sql_type == "BOOL":
                    dist = DistCategorical(
                        domain={"values": ["true", "false"], "probs": None}
                    )
                elif col.sql_type in ["DATE", "DATETIME"]:
                    dist = DistSeasonal(
                        granularity="month",
                        weights={
                            "January": 0.08, "February": 0.08, "March": 0.08,
                            "April": 0.08, "May": 0.08, "June": 0.08,
                            "July": 0.08, "August": 0.08, "September": 0.08,
                            "October": 0.08, "November": 0.08, "December": 0.12
                        }
                    )
                else:
                    dist = DistCategorical(
                        domain={"values": [f"{col.name}_value_{i}" for i in range(10)], "probs": None}
                    )
                
                new_spec = ColumnGenSpec(
                    table=table_name,
                    column=col.name,
                    distribution=dist,
                    provider=None
                )
                new_specs.append(new_spec)
                logger.info(f"Auto-generated spec for {table_name}.{col.name}")
    
    if new_specs:
        logger.info(f"Repaired {len(new_specs)} MISSING_GEN_SPEC issues")
        # Create new GenerationIR with added specs
        all_columns = list(generation_ir.columns) + new_specs
        return GenerationIR(columns=all_columns)
    
    return generation_ir


def repair_table_name_typos(
    logical_ir: LogicalIR, generation_ir: GenerationIR, issues: List[QaIssue]
) -> GenerationIR:
    """
    Repair GEN_TABLE_MISSING issues by fixing table name typos.
    
    Uses fuzzy matching to find the closest table name.
    
    Args:
        logical_ir: LogicalIR with valid table names
        generation_ir: Current GenerationIR with wrong table names
        issues: List of GEN_TABLE_MISSING issues
    
    Returns:
        Updated GenerationIR with corrected table names
    """
    valid_tables = set(logical_ir.tables.keys())
    fixed_count = 0
    
    # Build mapping of wrong table names to correct ones
    table_fixes = {}
    for issue in issues:
        if issue.code == "GEN_TABLE_MISSING":
            wrong_table = issue.details.get("table", "")
            if wrong_table and wrong_table not in valid_tables:
                # Find closest match
                best_match = None
                best_score = 0
                
                for valid_table in valid_tables:
                    # Simple similarity: check if wrong_table is substring or vice versa
                    if wrong_table.lower() in valid_table.lower() or valid_table.lower() in wrong_table.lower():
                        score = min(len(wrong_table), len(valid_table)) / max(len(wrong_table), len(valid_table))
                        if score > best_score:
                            best_score = score
                            best_match = valid_table
                
                # Also check for common typos (fact_ vs dim_ prefix issues)
                if not best_match:
                    for valid_table in valid_tables:
                        # Remove common prefixes and compare
                        wrong_base = wrong_table.replace("fact_", "").replace("dim_", "")
                        valid_base = valid_table.replace("fact_", "").replace("dim_", "")
                        if wrong_base == valid_base:
                            best_match = valid_table
                            break
                
                if best_match:
                    table_fixes[wrong_table] = best_match
                    logger.info(f"Repairing table name: '{wrong_table}' -> '{best_match}'")
    
    # Apply fixes
    if table_fixes:
        updated_columns = []
        for cg in generation_ir.columns:
            if cg.table in table_fixes:
                # Create new spec with corrected table name
                new_cg = ColumnGenSpec(
                    table=table_fixes[cg.table],
                    column=cg.column,
                    distribution=cg.distribution,
                    provider=cg.provider
                )
                updated_columns.append(new_cg)
                fixed_count += 1
            else:
                updated_columns.append(cg)
        
        if fixed_count > 0:
            logger.info(f"Repaired {fixed_count} GEN_TABLE_MISSING issues")
            return GenerationIR(columns=updated_columns)
    
    return generation_ir


def auto_repair_issues(
    logical_ir: LogicalIR, generation_ir: GenerationIR, issues: List[QaIssue]
) -> GenerationIR:
    """
    Automatically repair common validation issues.
    
    Args:
        logical_ir: LogicalIR
        generation_ir: Current GenerationIR
        issues: List of validation issues
    
    Returns:
        Repaired GenerationIR
    """
    repaired_ir = generation_ir
    
    # Group issues by code
    issue_codes = {issue.code for issue in issues}
    
    # Repair MISSING_GEN_SPEC
    if "MISSING_GEN_SPEC" in issue_codes:
        repaired_ir = repair_missing_gen_specs(logical_ir, repaired_ir)
    
    # Repair GEN_TABLE_MISSING
    if "GEN_TABLE_MISSING" in issue_codes:
        repaired_ir = repair_table_name_typos(logical_ir, repaired_ir, issues)
    
    return repaired_ir

