"""Robust JSON extraction from LLM output."""

import json
import re
from typing import Any, Dict
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def _fix_common_json_issues(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues.
    
    Fixes:
    - Trailing commas
    - Missing quotes around keys
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    """
    original = json_str
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Replace single quotes with double quotes (simple cases)
    # Only do this carefully to avoid breaking string content
    # Match: 'key': value (but not inside strings)
    json_str = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)
    
    # Fix unescaped newlines in string values (replace with \n)
    # This is tricky - we need to be careful not to break valid JSON
    # Only fix obvious cases where newline is inside a string value
    json_str = re.sub(r':\s*"([^"]*)\n([^"]*)"', r': "\1\\n\2"', json_str)
    
    if json_str != original:
        logger.debug("Fixed common JSON issues (trailing commas, quotes, etc.)")
    
    return json_str


class JSONParseError(Exception):
    """Raised when JSON parsing fails."""

    pass


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM output text.

    Handles cases where LLM output includes markdown code blocks,
    explanatory text, or other formatting.

    Args:
        text: Raw LLM output text

    Returns:
        Parsed JSON as dictionary

    Raises:
        JSONParseError: If no valid JSON can be extracted, with helpful error message
    """
    errors = []  # Collect all parsing errors for better error message
    
    # Try to find JSON in markdown code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"Code block parse error: {e}")
            # Try to fix and retry
            json_str = _fix_common_json_issues(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                errors.append(f"Code block parse error (after fix): {e2}")

    # Try to find JSON object directly (more greedy - match from first { to last })
    # Find the first { and try to match balanced braces
    brace_count = 0
    start_idx = text.find('{')
    if start_idx != -1:
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count == 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                errors.append(f"JSON object parse error: {e}")
                # Try to fix common JSON issues
                json_str = _fix_common_json_issues(json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    errors.append(f"JSON object parse error (after fix): {e2}")

    # Try parsing the entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        errors.append(f"Full text parse error: {e}")

    # If all else fails, raise error with helpful message
    error_summary = "; ".join(errors[-3:])  # Show last 3 errors
    error_msg = (
        f"Could not extract valid JSON from LLM output. "
        f"Errors: {error_summary}. "
        f"Please ensure the response contains valid JSON with proper syntax."
    )
    logger.error(error_msg)
    logger.debug(f"Text content (first 1000 chars): {text[:1000]}...")
    raise JSONParseError(error_msg)

