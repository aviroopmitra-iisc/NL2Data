"""Distribution engineer agent for generation specifications."""

from typing import List
from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.generation import GenerationIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def _pre_validate_derived_expressions(data: dict, logical_ir) -> None:
    """
    Pre-validate derived expressions before IR validation.
    
    Checks:
    - Invalid functions (datetime, normal, etc.)
    - Self-references (circular dependencies)
    - References to non-existent columns
    - Basic syntax issues
    
    Args:
        data: GenerationIR JSON data
        logical_ir: LogicalIR for column reference checking
    """
    from nl2data.ir.generation import DistDerived
    from nl2data.generation.derived_program import compile_derived, ALLOWED_FUNCS
    import re
    
    if "columns" not in data:
        return
    
    tables = logical_ir.tables if logical_ir else {}
    invalid_functions = {"datetime", "normal", "random", "uniform", "uniform_int", "uniform_float"}
    
    for col_spec in data.get("columns", []):
        dist = col_spec.get("distribution", {})
        if dist.get("kind") == "derived":
            expr = dist.get("expression", "")
            table_name = col_spec.get("table", "")
            col_name = col_spec.get("column", "")
            
            # Check for self-reference (circular dependency)
            # Simple check: if column name appears as a standalone word in expression
            # Use word boundaries to avoid false positives (e.g., "price" in "price_multiplier")
            col_name_pattern = r'\b' + re.escape(col_name) + r'\b'
            if re.search(col_name_pattern, expr, re.IGNORECASE):
                logger.error(
                    f"DistributionEngineer: Self-reference detected! "
                    f"Derived column '{table_name}.{col_name}' references itself in expression: {expr[:100]}. "
                    f"This will cause a circular dependency. Consider making this column sampled instead, "
                    f"or fix the expression to reference the correct column (e.g., from a dimension table)."
                )
                # Note: We log as error but don't raise - let validation catch it
            
            # Check for invalid functions
            for invalid_func in invalid_functions:
                if f"{invalid_func}(" in expr.lower():
                    logger.warning(
                        f"DistributionEngineer: Found invalid function '{invalid_func}()' "
                        f"in derived expression for {table_name}.{col_spec.get('column', '?')}: {expr[:100]}"
                    )
            
            # Try to compile and check dependencies
            try:
                prog = compile_derived(expr, dist.get("dtype"))
                
                # Check if dependencies exist
                table = tables.get(table_name)
                if table:
                    table_cols = {c.name for c in table.columns}
                    for dep in prog.dependencies:
                        if dep not in table_cols:
                            # Check dimension tables
                            found = False
                            if table.foreign_keys:
                                for fk in table.foreign_keys:
                                    ref_table = tables.get(fk.ref_table)
                                    if ref_table and dep in {c.name for c in ref_table.columns}:
                                        found = True
                                        break
                            
                            if not found:
                                logger.warning(
                                    f"DistributionEngineer: Derived column {table_name}.{col_spec.get('column', '?')} "
                                    f"depends on '{dep}' which may not exist. Expression: {expr[:100]}"
                                )
            except Exception as e:
                # Will be caught by full validation later, just log warning
                logger.warning(
                    f"DistributionEngineer: Pre-validation warning for {table_name}.{col_spec.get('column', '?')}: {e}"
                )


def _generate_missing_specs(logical_ir, data: dict) -> List[dict]:
    """
    Generate default generation specs for missing columns.
    
    Args:
        logical_ir: LogicalIR to get all columns from
        data: GenerationIR JSON data (before validation)
    
    Returns:
        List of ColumnGenSpec dicts for missing columns
    """
    # Get existing specs
    existing_specs = {(col.get("table", ""), col.get("column", "")) 
                      for col in data.get("columns", [])}
    
    missing_specs = []
    
    # Iterate through all tables and columns in LogicalIR
    for table_name, table in logical_ir.tables.items():
        for col in table.columns:
            # Skip if spec already exists
            if (table_name, col.name) in existing_specs:
                continue
            
            # Create default spec based on column type and name
            col_name_lower = col.name.lower()
            sql_type = col.sql_type
            
            # Determine default distribution/provider based on patterns
            spec = {
                "table": table_name,
                "column": col.name,
                "distribution": None,
                "provider": None
            }
            
            # Primary keys: uniform distribution
            if col.role == "primary_key" or col.name in table.primary_key:
                if sql_type == "INT":
                    spec["distribution"] = {
                        "kind": "uniform",
                        "low": 1.0,
                        "high": 1000000.0
                    }
                else:
                    # For non-integer PKs, use uniform with reasonable range
                    spec["distribution"] = {
                        "kind": "uniform",
                        "low": 0.0,
                        "high": 1000000.0
                    }
            
            # Foreign keys: will be handled by FK logic, but add uniform as fallback
            elif col.role == "foreign_key":
                spec["distribution"] = {
                    "kind": "uniform",
                    "low": 1.0,
                    "high": 10000.0
                }
            
            # Date/Datetime columns: seasonal distribution
            elif sql_type in ("DATE", "DATETIME"):
                spec["distribution"] = {
                    "kind": "seasonal",
                    "granularity": "month",
                    "weights": {
                        "January": 0.08, "February": 0.08, "March": 0.08,
                        "April": 0.08, "May": 0.08, "June": 0.08,
                        "July": 0.08, "August": 0.08, "September": 0.08,
                        "October": 0.08, "November": 0.08, "December": 0.12
                    }
                }
            
            # Boolean columns: categorical with True/False
            elif sql_type == "BOOL":
                spec["distribution"] = {
                    "kind": "categorical",
                    "domain": {
                        "values": ["True", "False"],
                        "probs": None
                    }
                }
            
            # Text columns: categorical with common values
            elif sql_type == "TEXT":
                # Check for common patterns
                if col_name_lower.endswith("_id") or col_name_lower.endswith("_code"):
                    # ID/code patterns: categorical with generic codes
                    spec["distribution"] = {
                        "kind": "categorical",
                        "domain": {
                            "values": ["A001", "A002", "A003", "A004", "A005"],
                            "probs": None
                        }
                    }
                elif col_name_lower.endswith("_name") or "name" in col_name_lower:
                    # Name patterns: categorical with generic names
                    spec["distribution"] = {
                        "kind": "categorical",
                        "domain": {
                            "values": ["Item1", "Item2", "Item3", "Item4", "Item5"],
                            "probs": None
                        }
                    }
                else:
                    # Generic text: categorical with generic values
                    spec["distribution"] = {
                        "kind": "categorical",
                        "domain": {
                            "values": ["Value1", "Value2", "Value3", "Value4", "Value5"],
                            "probs": None
                        }
                    }
            
            # Numeric columns (INT, FLOAT): uniform distribution
            elif sql_type in ("INT", "FLOAT"):
                if sql_type == "INT":
                    # Integer: uniform with reasonable range
                    spec["distribution"] = {
                        "kind": "uniform",
                        "low": 0.0,
                        "high": 1000.0
                    }
                else:
                    # Float: uniform with reasonable range
                    spec["distribution"] = {
                        "kind": "uniform",
                        "low": 0.0,
                        "high": 100.0
                    }
            
            # Fallback: uniform distribution
            else:
                spec["distribution"] = {
                    "kind": "uniform",
                    "low": 0.0,
                    "high": 100.0
                }
            
            missing_specs.append(spec)
    
    return missing_specs


def _fix_derived_expression_mistakes(data: dict) -> dict:
    """
    Auto-correct common LLM mistakes in derived expressions.
    
    Fixes:
    - datetime() -> date() (for date extraction)
    - normal() -> Use uniform distribution instead
    - Other common function name mistakes
    """
    import json
    import re
    
    # Convert to JSON string for pattern matching
    json_str = json.dumps(data)
    original_json = json_str
    
    # Common function corrections
    # Note: datetime() is not allowed, but date() is - for date extraction use date()
    # For datetime creation, we can't fix automatically - it needs to be a sampled column
    corrections = [
        # These are invalid and should be caught by validation, but we can warn
        (r'"expression"\s*:\s*"([^"]*?)datetime\(', r'"expression": "\1date('),  # datetime() -> date() for extraction
    ]
    
    for pattern, replacement in corrections:
        json_str = re.sub(pattern, replacement, json_str, flags=re.IGNORECASE)
    
    # Parse back to dict
    try:
        if json_str != original_json:
            logger.warning("DistributionEngineer: Auto-corrected some derived expressions")
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If correction broke JSON, return original
        logger.warning("DistributionEngineer: Auto-correction broke JSON, using original")
        return data


class DistributionEngineer(BaseAgent):
    """Designs generation specifications from RequirementIR and LogicalIR."""

    name = "dist_engineer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate GenerationIR from RequirementIR and LogicalIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with generation_ir set
        """
        if board.logical_ir is None:
            logger.warning(
                "DistributionEngineer: LogicalIR not found, skipping"
            )
            return board

        logger.info("DistributionEngineer: Generating GenerationIR")

        try:
            sys_tmpl = load_prompt("roles/dist_system.txt")
            usr_tmpl = load_prompt("roles/dist_user.txt")

            system_content = sys_tmpl
            logical_json = board.logical_ir.model_dump_json(indent=2)
            requirement_json = (
                board.requirement_ir.model_dump_json(indent=2)
                if board.requirement_ir
                else "null"
            )
            
            # Extract table names and column info for prompt
            table_info = []
            for tname, table in board.logical_ir.tables.items():
                cols = [c.name for c in table.columns]
                table_info.append(f"Table '{tname}': {cols}")
            
            table_list = "\n".join(table_info)
            total_cols = sum(len(t.columns) for t in board.logical_ir.tables.values())
            valid_table_names = ", ".join(sorted(board.logical_ir.tables.keys()))

            user_content = render_prompt(
                usr_tmpl,
                LOGICAL_JSON=logical_json,
                REQUIREMENT_JSON=requirement_json,
                TABLE_LIST=table_list,
                TOTAL_COLUMNS=total_cols,
                VALID_TABLE_NAMES=valid_table_names,
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # Retry logic for JSON parsing AND IR validation (max 2 attempts)
            max_retries = 2
            data = None
            validation_error = None
            existing_specs_list = None  # Store for merging continuation responses
            
            for attempt in range(max_retries):
                try:
                    # Step 1: Call LLM and parse JSON
                    raw = chat(messages)
                    data = extract_json(raw)
                    
                    # If this is a continuation attempt, merge with previous specs if needed
                    if attempt > 0 and existing_specs_list is not None:
                        # Check if response is partial (only missing columns)
                        new_specs = {(col.get("table", ""), col.get("column", "")) 
                                    for col in data.get("columns", [])}
                        existing_specs_set = {(col.get("table", ""), col.get("column", "")) 
                                            for col in existing_specs_list}
                        
                        # If new specs don't include all existing ones, merge them
                        if not existing_specs_set.issubset(new_specs):
                            logger.info("Merging continuation response with previous specs")
                            # Merge: keep existing specs, add new ones
                            merged_columns = list(existing_specs_list)
                            for new_col in data.get("columns", []):
                                col_key = (new_col.get("table", ""), new_col.get("column", ""))
                                if col_key not in existing_specs_set:
                                    merged_columns.append(new_col)
                            data["columns"] = merged_columns
                            logger.info(f"Merged specs: {len(existing_specs_list)} existing + {len(data.get('columns', [])) - len(existing_specs_list)} new = {len(data.get('columns', []))} total")
                    
                    # Log the extracted data for debugging
                    logger.debug(f"Extracted JSON data (first 1000 chars): {str(data)[:1000]}")
                    
                    # Step 2: Post-process: Fix common LLM mistakes in derived expressions
                    data = _fix_derived_expression_mistakes(data)
                    
                    # Step 3: Pre-validate derived expressions before saving IR
                    _pre_validate_derived_expressions(data, board.logical_ir)
                    
                    # Step 4: Validate table names BEFORE validation
                    valid_tables = set(board.logical_ir.tables.keys())
                    gen_tables = {col.get("table", "") for col in data.get("columns", [])}
                    invalid_tables = gen_tables - valid_tables
                    
                    if invalid_tables:
                        error_msg = (
                            f"Invalid table names detected: {invalid_tables}. "
                            f"Valid table names: {sorted(valid_tables)}"
                        )
                        logger.error(error_msg)
                        if attempt < max_retries - 1:
                            # Retry with explicit table list
                            table_list = ", ".join(sorted(valid_tables))
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"⚠️⚠️⚠️ CRITICAL ERROR: The previous response used INVALID table names: {invalid_tables}. ⚠️⚠️⚠️\n\n"
                                    f"You MUST use ONLY these exact table names (NO EXCEPTIONS): {table_list}\n\n"
                                    f"DO NOT use table names from:\n"
                                    f"- Previous queries (e.g., fact_orders, dim_customer, fact_sales)\n"
                                    f"- Other domains or examples\n"
                                    f"- Templates or patterns\n\n"
                                    f"Your response was REJECTED because you used wrong table names. "
                                    f"You MUST fix ALL table references in your generation specs to use ONLY the valid table names listed above. "
                                    f"Check every single 'table' field in your JSON response and ensure it matches one of the valid names exactly."
                                )
                            })
                            continue  # Retry
                        else:
                            # Last attempt - raise error
                            raise ValueError(error_msg)
                    
                    # Step 5: Check completeness and handle large schemas
                    total_columns = sum(len(t.columns) for t in board.logical_ir.tables.values())
                    generated_specs = len(data.get("columns", []))
                    
                    # For large schemas, check if we need batch processing
                    from nl2data.generation.constants import (
                        LARGE_SCHEMA_COLUMN_THRESHOLD,
                        SCHEMA_COVERAGE_THRESHOLD,
                    )
                    if total_columns > LARGE_SCHEMA_COLUMN_THRESHOLD and generated_specs < total_columns * SCHEMA_COVERAGE_THRESHOLD:
                        # Less than 80% coverage - likely truncation or incomplete response
                        logger.warning(
                            f"Large schema detected ({total_columns} columns). "
                            f"Only {generated_specs}/{total_columns} columns have specs ({generated_specs/total_columns*100:.1f}% coverage). "
                            f"This may indicate LLM output truncation."
                        )
                        
                        # If this is not the last attempt, retry with continuation prompt
                        if attempt < max_retries - 1:
                            # Find missing columns
                            existing_specs = {(col.get("table", ""), col.get("column", "")) 
                                            for col in data.get("columns", [])}
                            missing_columns = []
                            for tname, table in board.logical_ir.tables.items():
                                for col in table.columns:
                                    if (tname, col.name) not in existing_specs:
                                        missing_columns.append(f"{tname}.{col.name}")
                            
                            if missing_columns:
                                missing_list = ", ".join(missing_columns[:50])  # Limit to first 50
                                if len(missing_columns) > 50:
                                    missing_list += f" ... and {len(missing_columns) - 50} more"
                                
                                logger.info(
                                    f"Retrying with continuation prompt for {len(missing_columns)} missing columns"
                                )
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"The previous response was incomplete. "
                                        f"Please continue generating specs for the remaining {len(missing_columns)} columns: "
                                        f"{missing_list}. "
                                        f"Return the COMPLETE JSON with ALL columns (including the ones you already generated). "
                                        f"Do NOT return only the missing columns - return the full specification."
                                    )
                                })
                                # Store existing specs to merge later if needed
                                if existing_specs_list is None:
                                    existing_specs_list = data.get("columns", [])
                                continue  # Retry with continuation prompt
                    
                    # If still incomplete after retries, auto-generate missing specs
                    if generated_specs < total_columns:
                        logger.warning(
                            f"Only {generated_specs}/{total_columns} columns have specs. "
                            f"Auto-generating missing ones..."
                        )
                        missing_specs = _generate_missing_specs(board.logical_ir, data)
                        if missing_specs:
                            data["columns"].extend(missing_specs)
                            logger.info(
                                f"Auto-generated {len(missing_specs)} missing generation specs"
                            )
                    
                    # Step 6: Validate IR structure
                    board.generation_ir = GenerationIR.model_validate(data)
                    
                    # Step 7: Early validation - check for common issues immediately
                    if board.logical_ir:
                        from nl2data.ir.dataset import DatasetIR
                        from nl2data.ir.validators import validate_generation, validate_derived_columns
                        temp_ir = DatasetIR(logical=board.logical_ir, generation=board.generation_ir)
                        
                        gen_issues = validate_generation(temp_ir)
                        derived_issues = validate_derived_columns(temp_ir)
                        
                        if gen_issues or derived_issues:
                            total_issues = len(gen_issues) + len(derived_issues)
                            issue_summary = "; ".join([
                                f"{issue.code}" for issue in (gen_issues + derived_issues)[:3]
                            ])
                            logger.warning(
                                f"GenerationIR validation found {total_issues} issues: {issue_summary}"
                            )
                            # Don't fail here - let QACompiler catch it, but log for awareness
                    
                    # Success! Exit retry loop
                    break
                    
                except JSONParseError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"JSON parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}")
                        # Add a hint to the user message for retry with specific guidance
                        error_hint = str(e)
                        if "trailing comma" in error_hint.lower() or "Expecting ','" in error_hint:
                            hint = "Please check for trailing commas and ensure all JSON syntax is correct. Return ONLY valid JSON, no markdown formatting."
                        elif "Expecting property name" in error_hint or "quote" in error_hint.lower():
                            hint = "Please ensure all keys are properly quoted with double quotes. Return ONLY valid JSON."
                        else:
                            hint = "Please return ONLY valid JSON, no markdown formatting or explanations. Ensure proper JSON syntax."
                        
                        messages.append({
                            "role": "user",
                            "content": hint
                        })
                    else:
                        # Last attempt failed
                        logger.error(f"JSON parsing failed after {max_retries} attempts: {e}")
                        raise
                        
                except Exception as e:
                    # IR validation error or other error
                    validation_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"IR validation failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.warning(f"Data that failed validation (first 1000 chars): {str(data)[:1000] if data else 'N/A'}")
                        # Add a hint to the user message for retry
                        error_summary = str(e)[:200]  # Truncate long error messages
                        messages.append({
                            "role": "user",
                            "content": f"The previous response failed validation. Error: {error_summary}. "
                                      f"Please fix the JSON structure and ensure all required fields are present and correctly formatted."
                        })
                    else:
                        # Last attempt failed
                        logger.error(f"Validation error: {e}")
                        if data:
                            logger.error(f"Data that failed validation (first 2000 chars): {str(data)[:2000]}")
                        raise

            # Assign default providers for columns without explicit providers
            from nl2data.generation.providers.assign import assign_default_providers
            board = assign_default_providers(board)

            total_columns = sum(len(t.columns) for t in board.logical_ir.tables.values())
            logger.info(
                f"DistributionEngineer: Generated GenerationIR with "
                f"{len(board.generation_ir.columns)} column specifications "
                f"(expected {total_columns} columns)"
            )

        except JSONParseError as e:
            handle_agent_error("DistributionEngineer", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("DistributionEngineer", "execute", e)
            raise

        return board

