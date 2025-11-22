"""Distribution engineer agent for generation specifications."""

from typing import List
from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.generation import GenerationIR
from nl2data.config.logging import get_logger
from pydantic import ValidationError

logger = get_logger(__name__)


def _pre_validate_derived_expressions(data: dict, logical_ir) -> None:
    """
    Pre-validate derived expressions before IR validation.
    
    Checks:
    - Invalid functions (datetime, random, etc. - note: normal, lognormal, pareto are now allowed)
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
    # Only flag truly invalid functions (not in ALLOWED_FUNCS)
    # Note: normal(), uniform(), lognormal(), pareto() are now ALLOWED in derived expressions
    invalid_functions = {"datetime", "random", "uniform_int", "uniform_float"}
    
    for col_spec in data.get("columns", []):
        dist = col_spec.get("distribution", {})
        if dist.get("kind") == "derived":
            expr = dist.get("expression", "")
            table_name = col_spec.get("table", "")
            col_name = col_spec.get("column", "")
            
            # Check for self-reference (circular dependency)
            # Improved check: only flag if column name appears as a direct reference, NOT as a function call
            # e.g., "day_of_week(timestamp)" is OK - it's calling a function, not referencing itself
            col_name_pattern = r'\b' + re.escape(col_name) + r'\b'
            matches = list(re.finditer(col_name_pattern, expr, re.IGNORECASE))
            
            # Check each match to see if it's a function call or a direct reference
            is_self_reference = False
            for match in matches:
                start, end = match.span()
                # Check if this match is followed by '(' (function call) - if so, it's NOT a self-reference
                if end < len(expr) and expr[end:end+1].strip() == '(':
                    # This is a function call like "day_of_week(...)", not a self-reference
                    continue
                # If we get here, it's a direct reference to the column name (self-reference)
                is_self_reference = True
                break
            
            if is_self_reference:
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


def _fix_table_name_mismatches(data: dict, valid_tables: set) -> dict:
    """
    Post-process generated IR to fix common table name mismatches.
    
    Common patterns:
    - Singular vs plural: dim_card -> dim_cards, fact_transaction -> fact_transactions
    - Case differences (handled by case-insensitive matching)
    - Close string matches (using simple heuristics)
    
    Only fixes if there's a clear, unambiguous match.
    
    Args:
        data: Generated IR data dict
        valid_tables: Set of valid table names from LogicalIR
        
    Returns:
        Modified data dict with fixed table names
    """
    if not valid_tables:
        return data
    
    # Create mapping of invalid -> valid table names
    invalid_tables = set()
    for col in data.get("columns", []):
        table_name = col.get("table", "")
        if table_name and table_name not in valid_tables:
            invalid_tables.add(table_name)
    
    if not invalid_tables:
        return data
    
    # Build mapping: invalid_name -> valid_name
    table_fixes = {}
    
    for invalid_name in invalid_tables:
        best_match = None
        best_score = 0
        
        # Try different matching strategies
        for valid_name in valid_tables:
            score = 0
            
            # Strategy 1: Exact match after removing common suffixes/prefixes
            invalid_base = invalid_name.lower()
            valid_base = valid_name.lower()
            
            # Strategy 2: Singular/plural matching
            # Check if one is singular and other is plural
            if invalid_base + 's' == valid_base or invalid_base == valid_base + 's':
                score = 0.9
            elif invalid_base + 'es' == valid_base or invalid_base == valid_base + 'es':
                score = 0.9
            elif invalid_base.rstrip('s') == valid_base or invalid_base == valid_base.rstrip('s'):
                score = 0.8
            
            # Strategy 3: Substring matching (one contains the other)
            if invalid_base in valid_base or valid_base in invalid_base:
                # Prefer longer matches
                overlap = min(len(invalid_base), len(valid_base)) / max(len(invalid_base), len(valid_base))
                score = max(score, 0.7 * overlap)
            
            # Strategy 4: Levenshtein-like: count matching characters from start
            # This handles cases like "dim_merchant" vs "dim_merchants"
            min_len = min(len(invalid_base), len(valid_base))
            matching_chars = 0
            for i in range(min_len):
                if invalid_base[i] == valid_base[i]:
                    matching_chars += 1
                else:
                    break
            
            if matching_chars >= 5:  # At least 5 matching chars from start
                similarity = matching_chars / max(len(invalid_base), len(valid_base))
                score = max(score, 0.6 * similarity)
            
            # Strategy 5: Check if they share the same prefix (dim_, fact_, etc.)
            invalid_parts = invalid_base.split('_')
            valid_parts = valid_base.split('_')
            if len(invalid_parts) >= 2 and len(valid_parts) >= 2:
                if invalid_parts[0] == valid_parts[0]:  # Same prefix (dim_, fact_, etc.)
                    # Check if the rest is similar
                    invalid_suffix = '_'.join(invalid_parts[1:])
                    valid_suffix = '_'.join(valid_parts[1:])
                    if invalid_suffix + 's' == valid_suffix or invalid_suffix == valid_suffix + 's':
                        score = max(score, 0.95)  # Very high confidence
                    elif invalid_suffix in valid_suffix or valid_suffix in invalid_suffix:
                        score = max(score, 0.75)
            
            # Strategy 6: Handle prefix-only differences (transactions vs fact_transactions)
            # If one has a prefix (dim_/fact_) and the other doesn't, but the suffix matches
            invalid_has_prefix = invalid_base.startswith('dim_') or invalid_base.startswith('fact_')
            valid_has_prefix = valid_base.startswith('dim_') or valid_base.startswith('fact_')
            
            if invalid_has_prefix != valid_has_prefix:  # One has prefix, other doesn't
                # Extract suffix (remove dim_/fact_ prefix)
                invalid_suffix = invalid_base
                valid_suffix = valid_base
                
                for prefix in ['dim_', 'fact_']:
                    if invalid_suffix.startswith(prefix):
                        invalid_suffix = invalid_suffix[len(prefix):]
                    if valid_suffix.startswith(prefix):
                        valid_suffix = valid_suffix[len(prefix):]
                
                # If suffixes match exactly (or with plural), high confidence
                if invalid_suffix == valid_suffix:
                    score = max(score, 0.95)  # Very high confidence
                elif invalid_suffix + 's' == valid_suffix or invalid_suffix == valid_suffix + 's':
                    score = max(score, 0.90)  # High confidence
                elif invalid_suffix in valid_suffix or valid_suffix in invalid_suffix:
                    # Partial match
                    overlap = min(len(invalid_suffix), len(valid_suffix)) / max(len(invalid_suffix), len(valid_suffix))
                    if overlap > 0.8:  # At least 80% overlap
                        score = max(score, 0.85)
            
            if score > best_score:
                best_score = score
                best_match = valid_name
        
        # Only fix if we have a high-confidence match
        if best_match and best_score >= 0.7:
            table_fixes[invalid_name] = best_match
            logger.info(
                f"DistributionEngineer: Auto-fixing table name '{invalid_name}' -> '{best_match}' "
                f"(confidence: {best_score:.2f})"
            )
    
    # Apply fixes
    if table_fixes:
        # Fix columns
        for col in data.get("columns", []):
            table_name = col.get("table", "")
            if table_name in table_fixes:
                col["table"] = table_fixes[table_name]
        
        # Fix events
        for event in data.get("events", []):
            for effect in event.get("effects", []):
                table_name = effect.get("table", "")
                if table_name in table_fixes:
                    effect["table"] = table_fixes[table_name]
        
        logger.info(
            f"DistributionEngineer: Fixed {len(table_fixes)} table name(s): {table_fixes}"
        )
    
    return data


def _fix_provider_field_misuse(data: dict) -> dict:
    """
    Post-process generated IR to fix provider field misuse.
    
    LLM sometimes puts provider info in the distribution field:
    - {"kind": "provider", "provider": "name"} 
      -> Move to column-level provider field
    
    Args:
        data: Generated IR data dict
        
    Returns:
        Modified data dict with fixed provider fields
    """
    fixed_count = 0
    
    for col in data.get("columns", []):
        dist = col.get("distribution", {})
        
        # Check if distribution has "kind": "provider" (wrong structure)
        if isinstance(dist, dict) and dist.get("kind") == "provider":
            # Extract provider info from distribution
            provider_name = dist.get("provider") or dist.get("name")
            
            if provider_name:
                # Convert simple string to proper provider structure
                if isinstance(provider_name, str):
                    # Handle cases like "name" -> "faker.name"
                    if not provider_name.startswith("faker.") and not provider_name.startswith("mimesis."):
                        # Try to infer faker provider
                        provider_mapping = {
                            "name": "faker.name",
                            "email": "faker.email",
                            "phone": "faker.phone_number",
                            "phone_number": "faker.phone_number",
                            "city": "faker.city",
                            "country": "faker.country",
                            "credit_card_number": "faker.credit_card_number",
                            "company": "faker.company",
                        }
                        provider_name = provider_mapping.get(provider_name.lower(), f"faker.{provider_name}")
                    
                    # Set provider at column level
                    col["provider"] = {
                        "name": provider_name,
                        "config": dist.get("config", {})
                    }
                    
                    # Replace distribution with categorical (fallback)
                    col["distribution"] = {
                        "kind": "categorical",
                        "domain": {"values": ["fake_value"], "probs": None}
                    }
                    
                    fixed_count += 1
                    logger.info(
                        f"DistributionEngineer: Fixed provider misuse for {col.get('table', '?')}.{col.get('column', '?')}: "
                        f"moved provider '{provider_name}' from distribution to provider field"
                    )
                else:
                    # Provider is already a dict, just move it
                    col["provider"] = provider_name
                    col["distribution"] = {
                        "kind": "categorical",
                        "domain": {"values": ["fake_value"], "probs": None}
                    }
                    fixed_count += 1
    
    if fixed_count > 0:
        logger.info(
            f"DistributionEngineer: Fixed {fixed_count} provider field misuse(s)"
        )
    
    return data


def _fix_mixture_condition_format(data: dict) -> dict:
    """
    Post-process generated IR to fix mixture condition format issues.
    
    LLM sometimes outputs conditions as strings instead of dictionaries:
    - "merchant_category in ('electronics', 'travel')" 
      -> {"column": "merchant_category", "op": "in", "value": ["electronics", "travel"]}
    - "category == 'electronics'"
      -> {"column": "category", "op": "eq", "value": "electronics"}
    
    Args:
        data: Generated IR data dict
        
    Returns:
        Modified data dict with fixed condition formats
    """
    import re
    
    fixed_count = 0
    
    for col in data.get("columns", []):
        dist = col.get("distribution", {})
        if dist.get("kind") == "mixture":
            components = dist.get("components", [])
            for component in components:
                condition = component.get("condition")
                
                # Skip if already a dict or None
                if condition is None or isinstance(condition, dict):
                    continue
                
                # Try to parse string condition
                if isinstance(condition, str):
                    try:
                        # Pattern 1: "column in ('val1', 'val2', ...)"
                        match = re.match(r'(\w+)\s+in\s+\(([^)]+)\)', condition)
                        if match:
                            column_name = match.group(1)
                            values_str = match.group(2)
                            # Parse values (remove quotes)
                            values = [v.strip().strip("'\"") for v in values_str.split(',')]
                            component["condition"] = {
                                "column": column_name,
                                "op": "in",
                                "value": values
                            }
                            fixed_count += 1
                            continue
                        
                        # Pattern 2: "column == 'value'" or "column = 'value'" or "column != 'value'"
                        match = re.match(r'(\w+)\s*[=!]=\s*["\']([^"\']+)["\']', condition)
                        if match:
                            column_name = match.group(1)
                            value = match.group(2)
                            # Check for != first (inequality) before checking == or = (equality)
                            # This prevents incorrectly classifying "!=" as "==" when checking for "="
                            op = "ne" if "!=" in condition else "eq"
                            component["condition"] = {
                                "column": column_name,
                                "op": op,
                                "value": value
                            }
                            fixed_count += 1
                            continue
                        
                        # Pattern 3: "column != 'value'"
                        match = re.match(r'(\w+)\s*!=\s*["\']([^"\']+)["\']', condition)
                        if match:
                            column_name = match.group(1)
                            value = match.group(2)
                            component["condition"] = {
                                "column": column_name,
                                "op": "ne",
                                "value": value
                            }
                            fixed_count += 1
                            continue
                        
                        # If we can't parse it, set to None (no condition)
                        logger.warning(
                            f"DistributionEngineer: Could not parse condition '{condition}', "
                            f"setting to None for {col.get('table', '?')}.{col.get('column', '?')}"
                        )
                        component["condition"] = None
                        fixed_count += 1
                        
                    except Exception as e:
                        logger.warning(
                            f"DistributionEngineer: Error parsing condition '{condition}': {e}, "
                            f"setting to None for {col.get('table', '?')}.{col.get('column', '?')}"
                        )
                        component["condition"] = None
                        fixed_count += 1
    
    if fixed_count > 0:
        logger.info(
            f"DistributionEngineer: Fixed {fixed_count} mixture condition format(s)"
        )
    
    return data


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


def _format_validation_error(error: Exception) -> str:
    """
    Format a Pydantic ValidationError into a clear, actionable message for the LLM.
    
    Args:
        error: The ValidationError exception
        
    Returns:
        A formatted error message with specific field paths and expected types
    """
    if isinstance(error, ValidationError):
        error_messages = []
        error_messages.append("The JSON structure failed validation. Here are the specific errors:")
        
        # Pydantic v2 ValidationError has errors() method that returns structured error info
        for err in error.errors():
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            error_type = err.get("type", "")
            input_value = err.get("input", None)
            
            # Build a clear error message
            error_msg = f"\n- Field: {loc}"
            error_msg += f"\n  Error: {msg}"
            
            # Add helpful context based on error type
            if error_type == "string_type" and input_value is not None:
                error_msg += f"\n  Issue: Expected a string, but got {type(input_value).__name__} with value {input_value}"
                error_msg += f"\n  Fix: Convert the value to a string (e.g., {input_value} -> \"{input_value}\")"
            elif error_type == "int_parsing" or error_type == "int_parsing_size":
                error_msg += f"\n  Issue: Expected an integer, but got {type(input_value).__name__} with value {input_value}"
            elif error_type == "float_parsing":
                error_msg += f"\n  Issue: Expected a float, but got {type(input_value).__name__} with value {input_value}"
            elif "missing" in error_type:
                error_msg += f"\n  Fix: Add the missing required field"
            
            error_messages.append(error_msg)
        
        return "\n".join(error_messages)
    else:
        # For non-ValidationError exceptions, return a truncated summary
        error_str = str(error)
        if len(error_str) > 500:
            error_str = error_str[:500] + "..."
        return f"Validation error: {error_str}"


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
                    # Append assistant's response to messages for conversation history
                    messages.append({"role": "assistant", "content": raw})
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
                    
                    # Step 3: Fix provider field misuse
                    data = _fix_provider_field_misuse(data)
                    
                    # Step 4: Fix mixture condition format issues
                    data = _fix_mixture_condition_format(data)
                    
                    # Step 5: Pre-validate derived expressions before saving IR
                    _pre_validate_derived_expressions(data, board.logical_ir)
                    
                    # Step 6: Fix common table name mismatches (post-processing)
                    valid_tables = set(board.logical_ir.tables.keys())
                    data = _fix_table_name_mismatches(data, valid_tables)
                    
                    # Step 7: Validate table names AFTER auto-fixing
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
                    
                    # Step 8: Check completeness and handle large schemas
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
                    
                    # Step 9: Validate IR structure
                    board.generation_ir = GenerationIR.model_validate(data)
                    
                    # Step 10: Early validation - check for common issues immediately
                    if board.logical_ir:
                        from nl2data.ir.dataset import DatasetIR
                        from nl2data.ir.validators import validate_generation, validate_derived_columns
                        temp_ir = DatasetIR(logical=board.logical_ir, generation=board.generation_ir)
                        
                        gen_issues = validate_generation(temp_ir)
                        derived_issues = validate_derived_columns(temp_ir)
                        
                        if gen_issues or derived_issues:
                            total_issues = len(gen_issues) + len(derived_issues)
                            all_issues = gen_issues + derived_issues
                            
                            # Define critical issue codes that should trigger retry
                            CRITICAL_ISSUE_CODES = {
                                "MISSING_DERIVED_DEP",  # Derived column references non-existent column
                                "GEN_COL_MISSING",  # Generation spec references missing column
                                "GEN_TABLE_MISSING",  # Generation spec references missing table
                            }
                            
                            # Check if any critical issues exist
                            critical_issues = [
                                issue for issue in all_issues 
                                if issue.code in CRITICAL_ISSUE_CODES
                            ]
                            
                            if critical_issues:
                                # Format error message for retry
                                issue_summary = "; ".join([
                                    f"{issue.code}" for issue in critical_issues[:3]
                                ])
                                
                                # Build detailed error message with actionable guidance
                                error_parts = [
                                    f"GenerationIR validation found {len(critical_issues)} critical issue(s): {issue_summary}",
                                    "",
                                    "Issue details:"
                                ]
                                
                                for issue in critical_issues[:5]:
                                    error_parts.append(f"\n- {issue.code} at {issue.location}")
                                    error_parts.append(f"  Problem: {issue.message}")
                                    
                                    # Add helpful suggestions from issue details if available
                                    if hasattr(issue, 'details') and issue.details:
                                        details = issue.details
                                        if 'similar_columns' in details and details['similar_columns']:
                                            error_parts.append(f"  Suggested fix: Use one of these columns instead: {', '.join(details['similar_columns'])}")
                                        elif 'available_columns' in details and details['available_columns']:
                                            error_parts.append(f"  Available columns: {', '.join(details['available_columns'][:10])}")
                                
                                error_msg = "\n".join(error_parts)
                                
                                logger.warning(
                                    f"GenerationIR validation found {len(critical_issues)} critical issues: {issue_summary}"
                                )
                                # Raise ValueError to trigger retry mechanism
                                raise ValueError(error_msg)
                            else:
                                # Non-critical issues - log warning but continue
                                issue_summary = "; ".join([
                                    f"{issue.code}" for issue in all_issues[:3]
                                ])
                                logger.warning(
                                    f"GenerationIR validation found {total_issues} non-critical issues: {issue_summary}"
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
                        # Format error message clearly for the LLM
                        formatted_error = _format_validation_error(e)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The previous response failed validation.\n\n{formatted_error}\n\n"
                                f"Please fix these specific issues in your JSON response. Pay special attention to the field paths "
                                f"and type conversions mentioned above."
                            )
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

