"""Mimesis-based value provider."""

import pandas as pd
from typing import Any
from .base import ValueProvider

try:
    from mimesis import Person
    MIMESIS_AVAILABLE = True
except ImportError:
    MIMESIS_AVAILABLE = False
    Person = None


class MimesisProvider:
    """Value provider using Mimesis library."""

    def __init__(self, locale: str = "en", field: str = "full_name", **kwargs):
        """
        Initialize Mimesis provider.

        Args:
            locale: Locale for Mimesis (default: "en")
            field: Mimesis field name (e.g., "full_name", "email", "telephone")
            **kwargs: Additional arguments passed to Mimesis
        """
        if not MIMESIS_AVAILABLE:
            raise ImportError(
                "Mimesis is not installed. Install it with: pip install mimesis"
            )

        self.pr = Person(locale)
        self.field = field
        self.kwargs = kwargs

    def sample(self, n: int, ctx=None, **kwargs) -> pd.Series:
        """
        Sample n values using Mimesis.

        Args:
            n: Number of values to generate
            ctx: Optional context (unused)
            **kwargs: Additional arguments (merged with instance kwargs)

        Returns:
            Series with generated values
        """
        merged_kwargs = {**self.kwargs, **kwargs}
        values = []
        for _ in range(n):
            try:
                value = getattr(self.pr, self.field)(**merged_kwargs)
                values.append(value)
            except AttributeError:
                raise ValueError(
                    f"Mimesis field '{self.field}' not available. "
                    f"Available fields include: full_name, email, telephone, etc."
                )
        return pd.Series(values)

