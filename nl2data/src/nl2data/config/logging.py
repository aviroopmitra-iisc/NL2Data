"""Logging configuration for NL2Data."""

import logging
import sys
from pathlib import Path
from typing import Optional
from .settings import get_settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        format_string: Optional custom format string
    """
    settings = get_settings()
    log_level = level or settings.log_level
    log_file_path = log_file or settings.log_file

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger("nl2data")
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Ensure logging is set up
    if not logging.getLogger("nl2data").handlers:
        setup_logging()

    # Return logger with nl2data prefix
    if name.startswith("nl2data"):
        return logging.getLogger(name)
    return logging.getLogger(f"nl2data.{name}")

