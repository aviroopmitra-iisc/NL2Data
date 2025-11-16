"""Faker-based value provider."""

import pandas as pd
from typing import Any
from .base import ValueProvider

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    Faker = None


class FakerProvider:
    """Value provider using Faker library."""

    def __init__(self, locale: str = "en_US", field: str = "name", **kwargs):
        """
        Initialize Faker provider.

        Args:
            locale: Locale for Faker (default: "en_US")
            field: Faker field name (e.g., "email", "phone_number", "address")
            **kwargs: Additional arguments passed to Faker
        """
        if not FAKER_AVAILABLE:
            raise ImportError(
                "Faker is not installed. Install it with: pip install faker"
            )

        self.fk = Faker(locale)
        self.field = field
        self.kwargs = kwargs

    def sample(self, n: int, ctx=None, **kwargs) -> pd.Series:
        """
        Sample n values using Faker.

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
                value = getattr(self.fk, self.field)(**merged_kwargs)
                values.append(value)
            except AttributeError:
                raise ValueError(
                    f"Faker field '{self.field}' not available. "
                    f"Available fields include: email, phone_number, address, etc."
                )
        return pd.Series(values)

