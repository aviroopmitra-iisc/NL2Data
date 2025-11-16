"""Base protocol for value providers."""

from typing import Protocol
import pandas as pd


class ValueProvider(Protocol):
    """
    Protocol for value providers that generate realistic data.

    Providers generate values offline (no HTTP calls) using libraries
    like Faker, Mimesis, or pre-downloaded datasets.
    """

    def sample(self, n: int, ctx=None, **kwargs) -> pd.DataFrame | pd.Series:
        """
        Sample n values.

        Args:
            n: Number of values to generate
            ctx: Optional context (e.g., table name, column name)
            **kwargs: Additional provider-specific arguments

        Returns:
            DataFrame or Series with generated values
        """
        ...

