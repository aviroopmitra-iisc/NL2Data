"""Robust JSON extraction from LLM output."""

import json
import re
from typing import Any, Dict
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


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
        JSONParseError: If no valid JSON can be extracted
    """
    # Try to find JSON in markdown code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from code block: {e}")

    # Try to find JSON object directly
    json_obj_pattern = r"\{.*\}"
    match = re.search(json_obj_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON object: {e}")

    # Try parsing the entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # If all else fails, raise error
    logger.error("Could not extract valid JSON from LLM output")
    logger.debug(f"Text content: {text[:500]}...")
    raise JSONParseError("Could not extract valid JSON from LLM output")

