"""Seasonal distribution sampler for temporal data."""

import numpy as np
import calendar
import re
from datetime import datetime, timedelta
from typing import Dict
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


class SeasonalDateSampler(BaseSampler):
    """Seasonal distribution sampler for dates."""

    def __init__(
        self,
        weights: Dict[str, float],
        granularity: str = "month",
        base_date: str = "2020-01-01",
    ):
        """
        Initialize seasonal date sampler.

        Args:
            weights: Dictionary mapping period names to weights
                    (e.g., {"January": 0.1, "February": 0.08, ...} or {"07:00-09:00": 2.5, ...})
            granularity: "month", "week", or "hour"
            base_date: Base date string (YYYY-MM-DD)
        """
        self.weights = weights
        self.granularity = granularity
        self.base_date = datetime.fromisoformat(base_date)
        self._normalize_weights()
        logger.debug(
            f"Initialized SeasonalDateSampler: {granularity}, "
            f"{len(weights)} periods"
        )

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total == 0:
            raise ValueError("Weights cannot all be zero")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """
        Generate n seasonal date samples.

        Args:
            n: Number of samples
            **kwargs:
                - rng: Random number generator
                - year_range: Tuple (start_year, end_year) for date range

        Returns:
            Array of datetime64 dates
        """
        rng = kwargs.get("rng", np.random.default_rng())
        year_range = kwargs.get("year_range", (2020, 2023))

        # Sample periods based on weights
        periods = list(self.weights.keys())
        probs = [self.weights[p] for p in periods]
        sampled_periods = rng.choice(periods, size=n, p=probs)

        # Convert periods to dates
        dates = []
        for period in sampled_periods:
            if self.granularity == "month":
                # Parse month name to month number
                month_map = {
                    "January": 1,
                    "February": 2,
                    "March": 3,
                    "April": 4,
                    "May": 5,
                    "June": 6,
                    "July": 7,
                    "August": 8,
                    "September": 9,
                    "October": 10,
                    "November": 11,
                    "December": 12,
                }
                month = month_map.get(period, 1)
                year = rng.integers(year_range[0], year_range[1] + 1)
                # Get the actual number of days in the month (handles leap years correctly)
                max_day = calendar.monthrange(year, month)[1]
                day = rng.integers(1, max_day + 1)
                dates.append(np.datetime64(f"{year}-{month:02d}-{day:02d}"))
            elif self.granularity == "week":
                # Week granularity - convert week number to actual date
                year = rng.integers(year_range[0], year_range[1] + 1)
                # Try to extract week number from period name (e.g., "Week 1", "Week 2")
                week = None
                if isinstance(period, str):
                    match = re.search(r'week\s*(\d+)', period.lower())
                    if match:
                        week = int(match.group(1))
                # Fallback to random week if not found in period name
                if week is None or week < 1 or week > 53:
                    week = rng.integers(1, 53)
                # Calculate date from week number
                # ISO week 1 is the week containing the first Thursday of the year
                # For simplicity, we'll use the first Monday approach
                jan1 = datetime(year, 1, 1)
                # Find the first Monday (ISO weeks start on Monday)
                # weekday() returns 0=Monday, 6=Sunday
                days_offset = (7 - jan1.weekday()) % 7
                if jan1.weekday() == 0:  # Already Monday
                    days_offset = 0
                first_monday = jan1 + timedelta(days=days_offset)
                # Add weeks (week 1 starts from the first Monday)
                target_date = first_monday + timedelta(weeks=int(week) - 1)
                # Ensure we don't go beyond the year
                if target_date.year > year:
                    target_date = datetime(year, 12, 31)
                dates.append(np.datetime64(target_date.date()))
            else:
                # Hour granularity - parse time ranges like "07:00-09:00" and assign random time within range
                year = rng.integers(year_range[0], year_range[1] + 1)
                month = rng.integers(1, 13)
                max_day = calendar.monthrange(year, month)[1]
                day = rng.integers(1, max_day + 1)
                
                # Parse hour range from period (e.g., "07:00-09:00" -> 7 to 9)
                hour_start, hour_end = 0, 23
                if isinstance(period, str):
                    # Try to parse time range format "HH:MM-HH:MM"
                    match = re.search(r'(\d{1,2}):\d{2}-(\d{1,2}):\d{2}', period)
                    if match:
                        hour_start = int(match.group(1))
                        hour_end = int(match.group(2))
                    else:
                        # Try to parse single hour "HH:MM"
                        match = re.search(r'(\d{1,2}):\d{2}', period)
                        if match:
                            hour_start = hour_end = int(match.group(1))
                
                # Sample random hour within range
                hour = rng.integers(hour_start, hour_end + 1) if hour_start <= hour_end else hour_start
                minute = rng.integers(0, 60)
                second = rng.integers(0, 60)
                
                target_datetime = datetime(year, month, day, hour, minute, second)
                dates.append(np.datetime64(target_datetime))

        return np.array(dates)

