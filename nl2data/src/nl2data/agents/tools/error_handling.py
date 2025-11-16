"""Error handling utilities for agent operations."""

from typing import TypeVar, Callable, Optional
from nl2data.agents.tools.json_parser import JSONParseError, extract_json
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def handle_agent_error(
    agent_name: str,
    operation: str,
    error: Exception,
    raise_on_json_error: bool = True,
) -> None:
    """
    Standardized error handling for agent operations.
    
    Args:
        agent_name: Name of the agent (e.g., "ManagerAgent")
        operation: Description of the operation (e.g., "parse JSON")
        error: The exception that occurred
        raise_on_json_error: Whether to raise JSONParseError or just log it
    """
    if isinstance(error, JSONParseError):
        logger.error(f"{agent_name}: Failed to {operation}: {error}")
        if raise_on_json_error:
            raise
    else:
        logger.error(f"{agent_name}: Failed: {error}", exc_info=True)
        raise


def safe_extract_json(text: str, agent_name: str) -> dict:
    """
    Safely extract JSON from text with standardized error handling.
    
    Args:
        text: Text containing JSON
        agent_name: Name of the agent for error messages
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        JSONParseError: If JSON extraction fails
    """
    try:
        return extract_json(text)
    except JSONParseError as e:
        handle_agent_error(agent_name, "parse JSON", e, raise_on_json_error=True)
        raise  # Should not reach here, but for type checking

