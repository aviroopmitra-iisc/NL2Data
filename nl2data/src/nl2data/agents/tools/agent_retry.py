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
                    f"JSON parsing failed (attempt {attempt + 1}/{max_retries}), retrying..."
                )
                messages.append({
                    "role": "user",
                    "content": "Please return ONLY valid JSON, no markdown formatting or explanations."
                })
            else:
                raise
                
        except (ValidationError, ValueError) as e:
            # IR validation error or custom validation error
            if attempt < max_retries - 1:
                error_summary = str(e)[:ERROR_MESSAGE_TRUNCATE_LENGTH]
                logger.warning(
                    f"IR validation failed (attempt {attempt + 1}/{max_retries}): {error_summary}"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"The previous response failed validation. Error: {error_summary}. "
                        f"Please fix the JSON structure and ensure all required fields are present and correctly formatted."
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

