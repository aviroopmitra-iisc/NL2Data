"""Logical designer agent for relational schema design."""

from nl2data.agents.base import BaseAgent, Blackboard
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.agents.tools.error_handling import handle_agent_error
from nl2data.prompts.loader import load_prompt, render_prompt
from nl2data.ir.logical import LogicalIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def _fix_common_llm_mistakes(data: dict) -> dict:
    """
    Auto-correct common LLM mistakes in LogicalIR JSON.
    
    Fixes:
    - INT66 -> INT (common typo)
    - INT32/INT64 -> INT (legacy types)
    - FLOAT32/FLOAT64 -> FLOAT (legacy types)
    - Other common SQL type typos
    """
    import json
    
    # Convert to JSON string and back to dict for deep traversal
    json_str = json.dumps(data)
    
    # Fix common SQL type typos
    corrections = {
        '"INT66"': '"INT"',
        '"INT32"': '"INT"',  # Legacy type -> INT
        '"INT64"': '"INT"',  # Legacy type -> INT
        '"INTEGER"': '"INT"',  # Generic INTEGER -> INT
        '"FLOAT32"': '"FLOAT"',  # Legacy type -> FLOAT
        '"FLOAT64"': '"FLOAT"',  # Legacy type -> FLOAT
        '"VARCHAR"': '"TEXT"',
        '"TIMESTAMP"': '"DATETIME"',
        '"BOOLEAN"': '"BOOL"',
    }
    
    # Apply corrections only if there's an actual change
    for wrong, correct in corrections.items():
        if wrong in json_str and wrong != correct:
            json_str = json_str.replace(wrong, correct)
            logger.warning(f"LogicalDesigner: Auto-corrected {wrong} -> {correct}")
    
    # Parse back to dict
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If correction broke JSON, return original
        logger.warning("LogicalDesigner: Auto-correction broke JSON, using original")
        return data


class LogicalDesigner(BaseAgent):
    """Designs logical relational schema from ConceptualIR."""

    name = "logical_designer"

    def run(self, board: Blackboard) -> Blackboard:
        """
        Generate LogicalIR from ConceptualIR.

        Args:
            board: Current blackboard state

        Returns:
            Updated blackboard with logical_ir set
        """
        if board.conceptual_ir is None:
            logger.warning(
                "LogicalDesigner: ConceptualIR not found, skipping"
            )
            return board

        logger.info("LogicalDesigner: Generating LogicalIR")

        try:
            sys_tmpl = load_prompt("roles/logical_system.txt")
            usr_tmpl = load_prompt("roles/logical_user.txt")

            system_content = sys_tmpl
            conceptual_json = board.conceptual_ir.model_dump_json(indent=2)
            requirement_json = (
                board.requirement_ir.model_dump_json(indent=2)
                if board.requirement_ir
                else "null"
            )

            user_content = render_prompt(
                usr_tmpl,
                CONCEPTUAL_JSON=conceptual_json,
                REQUIREMENT_JSON=requirement_json,
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # Retry logic for JSON parsing AND IR validation (max 2 attempts)
            max_retries = 2
            data = None
            validation_error = None
            
            for attempt in range(max_retries):
                try:
                    # Step 1: Call LLM and parse JSON
                    raw = chat(messages)
                    data = extract_json(raw)
                    
                    # Step 2: Auto-correct common LLM mistakes before validation
                    data = _fix_common_llm_mistakes(data)
                    
                    # Step 3: Validate IR structure
                    board.logical_ir = LogicalIR.model_validate(data)
                    
                    # Step 4: Early validation - check for common issues immediately
                    # Create temporary DatasetIR for validation (only LogicalIR is available at this stage)
                    from nl2data.ir.dataset import DatasetIR
                    from nl2data.ir.generation import GenerationIR
                    temp_ir = DatasetIR(
                        logical=board.logical_ir,
                        generation=GenerationIR(columns=[])  # Empty generation IR for logical-only validation
                    )
                    from nl2data.ir.validators import validate_logical
                    validation_issues = validate_logical(temp_ir)
                    if validation_issues:
                        issue_summary = "; ".join([f"{issue.code}: {issue.message[:50]}" for issue in validation_issues[:3]])
                        logger.warning(
                            f"LogicalIR validation found {len(validation_issues)} issues: {issue_summary}"
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
                                      f"Please fix the JSON structure and ensure all required fields are present and correctly formatted. "
                                      f"Remember: sql_type must be one of: INT, FLOAT, TEXT, DATE, DATETIME, BOOL"
                        })
                    else:
                        # Last attempt failed
                        logger.error(f"Validation error: {e}")
                        if data:
                            logger.error(f"Data that failed validation (first 2000 chars): {str(data)[:2000]}")
                        raise

            # Post-process: Fix common uniqueness issues
            # Mark name/type columns in dimension tables as unique
            for table_name, table_spec in board.logical_ir.tables.items():
                if table_spec.kind == "dimension":
                    for col in table_spec.columns:
                        col_lower = col.name.lower()
                        # Common patterns for unique identifier columns in dimensions
                        if any(pattern in col_lower for pattern in ['_name', '_type', 'type_', 'category', 'code', 'id']):
                            if col.role != "primary_key" and col.sql_type == "TEXT":
                                col.unique = True
                                logger.debug(
                                    f"LogicalDesigner: Marked {table_name}.{col.name} as unique "
                                    f"(dimension table name/type column)"
                                )

            # Copy schema_mode from RequirementIR if available
            if board.requirement_ir:
                board.logical_ir.schema_mode = board.requirement_ir.schema_mode
                logger.info(
                    f"LogicalDesigner: Copied schema_mode '{board.logical_ir.schema_mode}' "
                    f"from RequirementIR"
                )

            logger.info(
                f"LogicalDesigner: Generated LogicalIR with "
                f"{len(board.logical_ir.tables)} tables"
            )

        except JSONParseError as e:
            handle_agent_error("LogicalDesigner", "parse JSON", e)
            raise
        except Exception as e:
            handle_agent_error("LogicalDesigner", "execute", e)
            raise

        return board

