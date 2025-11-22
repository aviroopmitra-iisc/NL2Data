"""Common retry logic for agent LLM calls with JSON parsing and validation."""

from typing import TypeVar, Callable, List, Dict, Any, Optional, Type
from nl2data.agents.tools.llm_client import chat
from nl2data.agents.tools.json_parser import extract_json, JSONParseError
from nl2data.config.logging import get_logger
from nl2data.generation.constants import (
    AGENT_MAX_RETRIES,
    ERROR_MESSAGE_TRUNCATE_LENGTH,
    DEBUG_DATA_TRUNCATE_LENGTH,
    VALIDATION_DATA_TRUNCATE_LENGTH,
)
from pydantic import BaseModel, ValidationError

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


def _get_json_error_hint(error: JSONParseError) -> str:
    """
    Get specific error hint based on JSON parse error.
    
    Args:
        error: JSONParseError exception
        
    Returns:
        Specific error hint message for the LLM
    """
    error_str = str(error).lower()
    if "trailing comma" in error_str or "expecting ','" in error_str:
        return "Please check for trailing commas and ensure all JSON syntax is correct. Return ONLY valid JSON, no markdown formatting."
    elif "expecting property name" in error_str or "quote" in error_str:
        return "Please ensure all keys are properly quoted with double quotes. Return ONLY valid JSON."
    else:
        return "Please return ONLY valid JSON, no markdown formatting or explanations. Ensure proper JSON syntax."


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


def call_llm_with_retry(
    messages: List[Dict[str, str]],
    ir_model: Type[T],
    max_retries: int = AGENT_MAX_RETRIES,
    pre_process: Optional[Callable[[dict], dict]] = None,
    post_process: Optional[Callable[[dict], dict]] = None,
    custom_validation: Optional[Callable[[dict], None]] = None,
) -> T:
    """
    Common retry logic for agent LLM calls with JSON parsing and IR validation.
    
    This function handles:
    - LLM API calls
    - JSON extraction and parsing
    - IR model validation
    - Retry logic with error messages
    
    Args:
        messages: List of message dicts for LLM (will be modified with retry messages)
        ir_model: Pydantic model class to validate against
        max_retries: Maximum number of retry attempts
        pre_process: Optional function to process data before validation (e.g., fix common mistakes)
        post_process: Optional function to process data after extraction (e.g., wrap lists)
        custom_validation: Optional function for custom validation checks (raises ValueError on failure)
    
    Returns:
        Validated IR model instance
    
    Raises:
        JSONParseError: If JSON parsing fails after all retries
        ValidationError: If IR validation fails after all retries
        ValueError: If custom validation fails after all retries
    """
    data = None
    
    for attempt in range(max_retries):
        try:
            # Step 1: Call LLM and parse JSON
            raw = chat(messages)
            # Append assistant's response to messages for conversation history
            messages.append({"role": "assistant", "content": raw})
            data = extract_json(raw)
            
            # Step 2: Post-process if needed (e.g., wrap lists in dicts)
            if post_process:
                data = post_process(data)
            
            # Step 3: Pre-process if needed (e.g., fix common LLM mistakes)
            if pre_process:
                data = pre_process(data)
            
            # Step 4: Custom validation if provided
            if custom_validation:
                custom_validation(data)
            
            # Step 5: Validate IR structure
            ir_instance = ir_model.model_validate(data)
            
            # Success! Exit retry loop
            return ir_instance
            
        except JSONParseError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"JSON parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}"
                )
                error_hint = _get_json_error_hint(e)
                messages.append({
                    "role": "user",
                    "content": error_hint
                })
            else:
                raise
                
        except (ValidationError, ValueError) as e:
            # IR validation error or custom validation error
            if attempt < max_retries - 1:
                formatted_error = _format_validation_error(e)
                error_summary = str(e)[:ERROR_MESSAGE_TRUNCATE_LENGTH]
                logger.warning(
                    f"IR validation failed (attempt {attempt + 1}/{max_retries}): {error_summary}"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"The previous response failed validation.\n\n{formatted_error}\n\n"
                        f"Please fix these specific issues in your JSON response. Pay special attention to the field paths "
                        f"and type conversions mentioned above."
                    )
                })
            else:
                logger.error(f"Validation error: {e}")
                if data:
                    logger.error(
                        f"Data that failed validation (first {VALIDATION_DATA_TRUNCATE_LENGTH} chars): "
                        f"{str(data)[:VALIDATION_DATA_TRUNCATE_LENGTH]}"
                    )
                raise
                
        except Exception as e:
            # Other unexpected errors
            if attempt < max_retries - 1:
                error_summary = str(e)[:ERROR_MESSAGE_TRUNCATE_LENGTH]
                logger.warning(
                    f"Unexpected error (attempt {attempt + 1}/{max_retries}): {error_summary}"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"An error occurred: {error_summary}. "
                        f"Please review and fix the response."
                    )
                })
            else:
                logger.error(f"Unexpected error after {max_retries} attempts: {e}")
                raise
    
    # Should never reach here, but type checker needs it
    raise RuntimeError("Retry loop completed without returning")

