"""Automatic repair functions for common validation issues."""

import json
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
from nl2data.ir.validators import QaIssue
from nl2data.ir.generation import GenerationIR, ColumnGenSpec
from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec
from nl2data.config.logging import get_logger

if TYPE_CHECKING:
    from nl2data.ir.requirement import RequirementIR
from nl2data.prompts.loader import load_prompt, render_prompt

logger = get_logger(__name__)


def generate_categorical_values_with_llm(
    table: TableSpec,
    col: ColumnSpec,
    logical_ir: LogicalIR,
) -> Optional[List[str]]:
    """
    Use LLM to generate appropriate categorical values for a column.
    
    Args:
        table: Table specification
        col: Column specification
        logical_ir: LogicalIR for context
        
    Returns:
        List of categorical values as strings, or None if LLM call fails
    """
    try:
        from nl2data.agents.tools.llm_client import chat
        
        # Load prompt template
        try:
            template = load_prompt("repair/categorical_values.txt")
        except FileNotFoundError:
            logger.warning("Categorical values prompt template not found, using fallback")
            return None
        
        # Build context about other columns
        other_columns = []
        for other_col in table.columns:
            if other_col.name != col.name:
                other_columns.append(
                    f"- {other_col.name} ({other_col.sql_type}, role: {other_col.role or 'none'})"
                )
        other_columns_str = "\n".join(other_columns) if other_columns else "None"
        
        # Render prompt with context
        prompt = render_prompt(
            template,
            table_name=table.name,
            column_name=col.name,
            sql_type=col.sql_type,
            column_role=col.role or "none",
            is_unique=str(col.unique),
            is_nullable=str(not col.not_null if hasattr(col, 'not_null') else True),
            table_description=table.description or f"Table containing {col.name} and other related columns",
            other_columns=other_columns_str,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates realistic categorical values for database columns."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info(f"Calling LLM to generate categorical values for {table.name}.{col.name}")
        response = chat(messages)
        
        # Parse JSON response
        # Try to extract JSON from response (might have markdown code blocks)
        response_clean = response.strip()
        if "```json" in response_clean:
            # Extract JSON from code block
            start = response_clean.find("```json") + 7
            end = response_clean.find("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
        elif "```" in response_clean:
            # Extract from generic code block
            start = response_clean.find("```") + 3
            end = response_clean.find("```", start)
            if end > start:
                response_clean = response_clean[start:end].strip()
        
        # Parse JSON
        try:
            data = json.loads(response_clean)
            values = data.get("values", [])
            
            # Validate values
            if not isinstance(values, list) or len(values) == 0:
                logger.warning(f"LLM returned invalid values format: {values}")
                return None
            
            # Ensure all values are strings
            values = [str(v) for v in values]
            
            # Limit to reasonable number (8-15 as requested in prompt)
            if len(values) > 20:
                values = values[:20]
                logger.info(f"Truncated LLM-generated values to 20 items")
            
            logger.info(f"LLM generated {len(values)} categorical values for {table.name}.{col.name}")
            return values
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return None
            
    except Exception as e:
        logger.warning(f"LLM call failed for {table.name}.{col.name}: {e}")
        return None


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
                    # For TEXT/categorical columns, try to use LLM to generate values
                    # Fall back to placeholder values (column_name_value_0, column_name_value_1, ...) if LLM fails
                    categorical_values = generate_categorical_values_with_llm(table, col, logical_ir)
                    
                    if categorical_values:
                        dist = DistCategorical(
                            domain={"values": categorical_values, "probs": None}
                        )
                        logger.info(f"Using LLM-generated values for {table_name}.{col.name}")
                    else:
                        # Fallback: Use previous generation method (column_name_value_0, column_name_value_1, ...)
                        dist = DistCategorical(
                            domain={"values": [f"{col.name}_value_{i}" for i in range(10)], "probs": None}
                        )
                        logger.info(f"LLM failed, using fallback placeholder values for {table_name}.{col.name} (format: {col.name}_value_0, {col.name}_value_1, ...)")
                
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


def repair_missing_row_counts(
    logical_ir: LogicalIR,
    requirement_ir: Optional["RequirementIR"] = None
) -> LogicalIR:
    """
    Repair MISSING_ROW_COUNT issues by inferring row counts from scale hints or using defaults.
    
    Args:
        logical_ir: LogicalIR with missing row counts
        requirement_ir: Optional RequirementIR to get scale hints from
        
    Returns:
        Updated LogicalIR with row counts set
    """
    from nl2data.generation.constants import DEFAULT_FACT_ROWS, DEFAULT_DIMENSION_ROWS
    
    # Build scale hint mapping from RequirementIR if available
    scale_hints = {}
    if requirement_ir and requirement_ir.scale:
        for scale_item in requirement_ir.scale:
            if scale_item.table and scale_item.row_count:
                scale_hints[scale_item.table] = scale_item.row_count
    
    # Update tables with missing row_count
    updated_tables = {}
    for table_name, table in logical_ir.tables.items():
        if table.row_count is None:
            # Try to get from scale hints first
            if table_name in scale_hints:
                row_count = scale_hints[table_name]
                logger.info(f"Repaired {table_name}: set row_count={row_count} from scale hints")
            # Otherwise use defaults based on table kind
            elif table.kind == "fact":
                row_count = DEFAULT_FACT_ROWS
                logger.info(f"Repaired {table_name}: set row_count={row_count} (default for fact table)")
            elif table.kind == "dimension":
                row_count = DEFAULT_DIMENSION_ROWS
                logger.info(f"Repaired {table_name}: set row_count={row_count} (default for dimension table)")
            else:
                # Unknown kind - use fact table default as fallback
                row_count = DEFAULT_FACT_ROWS
                logger.info(f"Repaired {table_name}: set row_count={row_count} (default fallback)")
            
            # Create updated table with row_count
            updated_table = table.model_copy(update={"row_count": row_count})
            updated_tables[table_name] = updated_table
        else:
            updated_tables[table_name] = table
    
    # Create new LogicalIR with updated tables
    updated_ir = logical_ir.model_copy(update={"tables": updated_tables})
    return updated_ir


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


def repair_derived_column_dependencies(
    logical_ir: LogicalIR, generation_ir: GenerationIR, issues: List[QaIssue]
) -> GenerationIR:
    """
    Repair MISSING_DERIVED_DEP issues by replacing missing column references
    with similar available columns.
    
    Args:
        logical_ir: LogicalIR to get available columns from
        generation_ir: Current GenerationIR with broken derived columns
        issues: List of MISSING_DERIVED_DEP issues
    
    Returns:
        Updated GenerationIR with fixed derived expressions
    """
    import re
    
    # Get all MISSING_DERIVED_DEP issues
    derived_dep_issues = [issue for issue in issues if issue.code == "MISSING_DERIVED_DEP"]
    
    if not derived_dep_issues:
        return generation_ir
    
    # Build mapping of (table, column) -> ColumnGenSpec for easy lookup
    spec_map = {}
    for cg in generation_ir.columns:
        spec_map[(cg.table, cg.column)] = cg
    
    # Get available columns per table
    available_cols_by_table = {}
    for table_name, table in logical_ir.tables.items():
        available_cols_by_table[table_name] = {col.name for col in table.columns}
        # Also include columns from dimension tables via foreign keys
        for fk in table.foreign_keys:
            ref_table = logical_ir.tables.get(fk.ref_table)
            if ref_table:
                available_cols_by_table[table_name].update({col.name for col in ref_table.columns})
    
    fixed_count = 0
    updated_columns = []
    
    for issue in derived_dep_issues:
        table_name = issue.details.get("table", "")
        column_name = issue.details.get("column", "")
        missing_dep = issue.details.get("dependency", "")
        expression = issue.details.get("expression", "")
        similar_cols = issue.details.get("similar_columns", [])
        available_cols = issue.details.get("available_columns", [])
        
        if not table_name or not column_name or not missing_dep or not expression:
            logger.warning(f"Cannot repair {issue.location}: missing required details")
            # Keep original spec
            if (table_name, column_name) in spec_map:
                updated_columns.append(spec_map[(table_name, column_name)])
            continue
        
        # Find the spec for this derived column
        spec_key = (table_name, column_name)
        if spec_key not in spec_map:
            logger.warning(f"Cannot repair {issue.location}: spec not found")
            continue
        
        spec = spec_map[spec_key]
        from nl2data.ir.generation import DistDerived
        
        if not isinstance(spec.distribution, DistDerived):
            logger.warning(f"Cannot repair {issue.location}: not a derived column")
            updated_columns.append(spec)
            continue
        
        # Find replacement column
        replacement = None
        
        # Strategy 1: Use similar_columns from validator (most reliable)
        if similar_cols:
            # similar_cols format: ["table.column", ...]
            # Extract just column names and pick the first one
            for similar in similar_cols:
                if "." in similar:
                    _, col_name = similar.rsplit(".", 1)
                else:
                    col_name = similar
                
                # Check if this column is actually available in the table
                table_cols = available_cols_by_table.get(table_name, set())
                if col_name in table_cols:
                    replacement = col_name
                    logger.info(f"Using similar column '{replacement}' for missing '{missing_dep}' in {issue.location}")
                    break
        
        # Strategy 2: Fuzzy match from available columns
        if not replacement:
            table_cols = available_cols_by_table.get(table_name, set())
            if table_cols:
                # Try to find a column that contains the missing dependency name or vice versa
                missing_lower = missing_dep.lower()
                for col in table_cols:
                    col_lower = col.lower()
                    # Check if one is substring of the other
                    if missing_lower in col_lower or col_lower in missing_lower:
                        replacement = col
                        logger.info(f"Using fuzzy-matched column '{replacement}' for missing '{missing_dep}' in {issue.location}")
                        break
        
        # Strategy 3: Use first available column from the table (last resort)
        if not replacement:
            table_cols = available_cols_by_table.get(table_name, set())
            if table_cols:
                # Prefer columns that are not the derived column itself
                candidates = [c for c in table_cols if c != column_name]
                if candidates:
                    replacement = candidates[0]
                    logger.warning(f"Using first available column '{replacement}' as fallback for missing '{missing_dep}' in {issue.location}")
        
        if not replacement:
            logger.error(f"Cannot repair {issue.location}: no replacement found for '{missing_dep}'")
            # Keep original spec (will still fail validation, but at least we tried)
            updated_columns.append(spec)
            continue
        
        # Replace the missing dependency in the expression
        # Use word boundaries to avoid partial matches (e.g., "user_id" shouldn't match "user_id_old")
        # But be careful - the expression might have the column in different contexts
        old_expr = expression
        
        # Pattern: match the column name as a whole word (not part of another identifier)
        # This handles cases like "latency_p50_value" vs "latency_p50"
        pattern = r'\b' + re.escape(missing_dep) + r'\b'
        
        # Replace all occurrences
        new_expr = re.sub(pattern, replacement, old_expr)
        
        if new_expr == old_expr:
            logger.warning(f"Replacement pattern didn't match in expression: {old_expr}")
            updated_columns.append(spec)
            continue
        
        logger.info(f"Repaired {issue.location}: replaced '{missing_dep}' with '{replacement}'")
        logger.debug(f"  Old expression: {old_expr}")
        logger.debug(f"  New expression: {new_expr}")
        
        # Create new spec with fixed expression
        from nl2data.ir.generation import ColumnGenSpec
        new_dist = DistDerived(
            expression=new_expr,
            dtype=spec.distribution.dtype
        )
        new_spec = ColumnGenSpec(
            table=spec.table,
            column=spec.column,
            distribution=new_dist,
            provider=spec.provider
        )
        updated_columns.append(new_spec)
        fixed_count += 1
        
        # Remove from spec_map so we don't add it again
        del spec_map[spec_key]
    
    # Add all remaining specs that weren't modified
    for spec in spec_map.values():
        updated_columns.append(spec)
    
    if fixed_count > 0:
        logger.info(f"Repaired {fixed_count} MISSING_DERIVED_DEP issues")
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
    
    # Repair MISSING_DERIVED_DEP
    if "MISSING_DERIVED_DEP" in issue_codes:
        repaired_ir = repair_derived_column_dependencies(logical_ir, repaired_ir, issues)
    
    return repaired_ir

