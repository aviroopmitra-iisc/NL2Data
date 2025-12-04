"""Main evaluation functions."""

from .single_table import evaluate as evaluate_single_table
from .multi_table import evaluate_multi_table

__all__ = [
    "evaluate_single_table",
    "evaluate_multi_table",
]
