"""Lookup-based value provider for geographic data."""

import pandas as pd
from pathlib import Path
from typing import Any
from .base import ValueProvider
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class LookupProvider:
    """Value provider that samples from a pre-loaded DataFrame (e.g., Parquet)."""

    def __init__(
        self,
        parquet: str | Path | None = None,
        df: pd.DataFrame | None = None,
        weight_col: str | None = None,
        value_col: str | None = None,
        **kwargs,
    ):
        """
        Initialize lookup provider.

        Args:
            parquet: Path to Parquet file to load
            df: Pre-loaded DataFrame (alternative to parquet)
            weight_col: Column name for sampling weights (e.g., "pop" for population)
            value_col: Column name to return (if None, returns full row or first column)
            **kwargs: Additional arguments (unused)
        """
        if df is not None:
            self.df = df.copy()
        elif parquet is not None:
            parquet_path = Path(parquet)
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Parquet file not found: {parquet_path}. "
                    f"Please download the dataset first."
                )
            self.df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded lookup dataset from {parquet_path}: {len(self.df)} rows")
        else:
            raise ValueError("Either 'parquet' or 'df' must be provided")

        self.weight_col = weight_col
        self.value_col = value_col
        self.kwargs = kwargs

    def sample(self, n: int, ctx=None, **kwargs) -> pd.Series | pd.DataFrame:
        """
        Sample n values from the lookup DataFrame.

        Args:
            n: Number of values to generate
            ctx: Optional context (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Series (if value_col specified) or DataFrame (if value_col is None)
        """
        weights = None
        if self.weight_col and self.weight_col in self.df.columns:
            weights = self.df[self.weight_col]

        sampled = self.df.sample(n, weights=weights, replace=True)

        if self.value_col:
            if self.value_col not in sampled.columns:
                raise ValueError(
                    f"Column '{self.value_col}' not found in lookup DataFrame. "
                    f"Available columns: {list(sampled.columns)}"
                )
            return sampled[self.value_col]
        else:
            # Return first column if only one column, otherwise return full DataFrame
            if len(sampled.columns) == 1:
                return sampled.iloc[:, 0]
            return sampled


class GeoLookupProvider(LookupProvider):
    """Alias for LookupProvider with geo-specific defaults."""

    def __init__(self, dataset: str = "geonames.cities", **kwargs):
        """
        Initialize geo lookup provider.

        Args:
            dataset: Dataset name (default: "geonames.cities")
            **kwargs: Additional arguments passed to LookupProvider
        """
        # Default cache location
        cache_dir = Path.home() / ".nl2data" / "cache"
        parquet = cache_dir / f"{dataset}.parquet"

        # Set defaults if not provided
        if "weight_col" not in kwargs:
            kwargs["weight_col"] = "pop"
        if "value_col" not in kwargs:
            kwargs["value_col"] = "name"  # City name

        super().__init__(parquet=parquet, **kwargs)

