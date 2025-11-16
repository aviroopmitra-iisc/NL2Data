"""Retry utilities for LLM API calls."""

import time
from typing import Callable, TypeVar, Optional
from nl2data.config.settings import get_settings
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is transient
    """
    error_str = str(error)
    return (
        "Model reloaded" in error_str or
        "reloaded" in error_str.lower() or
        "timeout" in error_str.lower() or
        (hasattr(error, 'status_code') and error.status_code in [503, 504, 408])
    )


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int,
    base_delay: float,
    timeout_errors: tuple = (),
    operation_name: str = "operation",
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry (no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        timeout_errors: Tuple of timeout exception types
        operation_name: Name of operation for logging
        
    Returns:
        Result of function call
        
    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except timeout_errors as e:
            # Handle timeout errors with exponential backoff
            if attempt < max_retries - 1:
                retry_delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"{operation_name} timed out on attempt {attempt + 1}/{max_retries}. "
                    f"Retrying in {retry_delay:.1f} seconds..."
                )
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {e}", exc_info=True)
                raise
        except Exception as e:
            # Check if it's a transient error that we should retry
            if is_transient_error(e) and attempt < max_retries - 1:
                retry_delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Transient error on attempt {attempt + 1}/{max_retries}: {str(e)}. "
                    f"Retrying in {retry_delay:.1f} seconds..."
                )
                time.sleep(retry_delay)
                continue
            else:
                # Non-transient error or last attempt
                logger.error(f"{operation_name} failed: {e}", exc_info=True)
                raise
    
    # Should not reach here, but for type checking
    raise RuntimeError(f"{operation_name} failed after {max_retries} attempts")

