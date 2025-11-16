"""Utilities for loading and rendering prompt files."""

from pathlib import Path
from typing import Dict, Any
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Base directory for prompts
PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(path: str) -> str:
    """
    Load a prompt file from the prompts directory.

    Args:
        path: Relative path from prompts/ directory, e.g., 'roles/manager_system.txt'

    Returns:
        Prompt file contents as string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    full_path = PROMPTS_DIR / path
    if not full_path.exists():
        logger.error(f"Prompt file not found: {full_path}")
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    try:
        content = full_path.read_text(encoding="utf-8")
        logger.debug(f"Loaded prompt from {path}")
        return content
    except Exception as e:
        logger.error(f"Error loading prompt from {path}: {e}")
        raise


def render_prompt(template: str, **kwargs: Any) -> str:
    """
    Render a prompt template with placeholders.

    Uses str.format() for simple templating. Placeholders in files
    should use {PLACEHOLDER_NAME} format.

    Args:
        template: Template string with placeholders
        **kwargs: Values to fill placeholders

    Returns:
        Rendered prompt string
    """
    try:
        rendered = template.format(**kwargs)
        logger.debug(f"Rendered prompt with {len(kwargs)} placeholders")
        return rendered
    except KeyError as e:
        logger.error(f"Missing placeholder in template: {e}")
        raise
    except Exception as e:
        logger.error(f"Error rendering prompt: {e}")
        raise

