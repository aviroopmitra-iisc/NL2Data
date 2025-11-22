"""Comprehensive error logging utilities for data generation."""

import traceback
import sys
from typing import Any, Optional, Dict
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    operation: Optional[str] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
    chunk_num: Optional[int] = None,
    log_level: str = "error",
) -> None:
    """
    Log an error with comprehensive details including type, message, context, and traceback.
    
    Args:
        error: The exception that occurred
        context: Additional context dictionary (e.g., {'rows': 1000, 'columns': ['col1', 'col2']})
        operation: Description of the operation being performed
        table_name: Name of the table where error occurred
        column_name: Name of the column where error occurred
        chunk_num: Chunk number if applicable
        log_level: Logging level ('error', 'warning', 'critical')
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Build context message
    context_parts = []
    if operation:
        context_parts.append(f"Operation: {operation}")
    if table_name:
        context_parts.append(f"Table: {table_name}")
    if column_name:
        context_parts.append(f"Column: {column_name}")
    if chunk_num is not None:
        context_parts.append(f"Chunk: {chunk_num}")
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        context_parts.append(f"Context: {context_str}")
    
    context_msg = " | ".join(context_parts) if context_parts else "No additional context"
    
    # Build error message
    error_msg = f"[{error_type}] {error_message}"
    if context_parts:
        error_msg += f" | {context_msg}"
    
    # Log with appropriate level and full traceback
    if log_level.lower() == "critical":
        logger.critical(error_msg, exc_info=True)
    elif log_level.lower() == "warning":
        logger.warning(error_msg, exc_info=True)
    else:
        logger.error(error_msg, exc_info=True)
    
    # Also log error type-specific details
    if isinstance(error, ValueError):
        logger.debug(f"ValueError details: Invalid value or argument - {error_message}")
    elif isinstance(error, KeyError):
        logger.debug(f"KeyError details: Missing key - {error_message}")
    elif isinstance(error, AttributeError):
        logger.debug(f"AttributeError details: Missing attribute - {error_message}")
    elif isinstance(error, TypeError):
        logger.debug(f"TypeError details: Type mismatch - {error_message}")
    elif isinstance(error, IndexError):
        logger.debug(f"IndexError details: Index out of range - {error_message}")
    elif isinstance(error, FileNotFoundError):
        logger.debug(f"FileNotFoundError details: File not found - {error_message}")
    elif isinstance(error, PermissionError):
        logger.debug(f"PermissionError details: Permission denied - {error_message}")
    elif isinstance(error, MemoryError):
        logger.critical(f"MemoryError details: Out of memory - {error_message}")
    elif isinstance(error, RuntimeError):
        logger.debug(f"RuntimeError details: Runtime issue - {error_message}")
    elif isinstance(error, ImportError):
        logger.debug(f"ImportError details: Import failed - {error_message}")
    elif isinstance(error, OSError):
        logger.debug(f"OSError details: OS-level error - {error_message}")
    elif isinstance(error, AssertionError):
        logger.debug(f"AssertionError details: Assertion failed - {error_message}")
    elif isinstance(error, NotImplementedError):
        logger.debug(f"NotImplementedError details: Feature not implemented - {error_message}")
    elif isinstance(error, OverflowError):
        logger.debug(f"OverflowError details: Numeric overflow - {error_message}")
    elif isinstance(error, ZeroDivisionError):
        logger.debug(f"ZeroDivisionError details: Division by zero - {error_message}")
    
    # Log full traceback at debug level for detailed analysis
    logger.debug(f"Full traceback for {error_type}:\n{traceback.format_exc()}")


def log_error_with_recovery(
    error: Exception,
    recovery_action: str,
    context: Optional[Dict[str, Any]] = None,
    operation: Optional[str] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
) -> None:
    """
    Log an error and the recovery action taken.
    
    Args:
        error: The exception that occurred
        recovery_action: Description of how the error was handled/recovered
        context: Additional context dictionary
        operation: Description of the operation being performed
        table_name: Name of the table where error occurred
        column_name: Name of the column where error occurred
    """
    log_error(
        error=error,
        context=context,
        operation=operation,
        table_name=table_name,
        column_name=column_name,
        log_level="warning"
    )
    logger.warning(f"Recovery action: {recovery_action}")


def safe_execute(
    func,
    error_context: Optional[Dict[str, Any]] = None,
    operation: Optional[str] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
    default_return: Any = None,
    raise_on_error: bool = True,
) -> Any:
    """
    Safely execute a function with comprehensive error logging.
    
    Args:
        func: Function to execute (callable)
        error_context: Additional context for error logging
        operation: Description of the operation
        table_name: Name of the table
        column_name: Name of the column
        default_return: Value to return if error occurs and raise_on_error=False
        raise_on_error: Whether to re-raise the exception after logging
        
    Returns:
        Result of func() or default_return if error occurred and raise_on_error=False
    """
    try:
        return func()
    except Exception as e:
        log_error(
            error=e,
            context=error_context,
            operation=operation,
            table_name=table_name,
            column_name=column_name
        )
        if raise_on_error:
            raise
        return default_return

