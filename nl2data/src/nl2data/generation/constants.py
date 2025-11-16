"""Constants for data generation."""

# Default row counts
DEFAULT_DIMENSION_ROWS = 100_000
DEFAULT_FACT_ROWS = 1_000_000

# Fallback sampling ranges
DEFAULT_INT_RANGE = (1, 1_000_000)
DEFAULT_DATE_RANGE_DAYS = 365 * 3  # 3 years

# Progress logging
PROGRESS_LOG_INTERVAL_SECONDS = 30  # Log progress every 30 seconds for large tables
LARGE_TABLE_THRESHOLD = 1_000_000  # Tables with more rows are considered "large"

# Rush hour time ranges (for semantic datetime transformations)
RUSH_HOUR_MORNING = (7, 9)  # 7 AM - 9 AM
RUSH_HOUR_EVENING = (16, 18)  # 4 PM - 6 PM

# Duration/distance ranges for rush hour transformations
DURATION_BASE_MIN = 15
DURATION_BASE_MAX = 30
DURATION_RUSH_MIN = 30
DURATION_RUSH_MAX = 60

DISTANCE_BASE_MIN = 3
DISTANCE_BASE_MAX = 8
DISTANCE_RUSH_MIN = 5
DISTANCE_RUSH_MAX = 12

SURGE_BASE_MIN = 1.0
SURGE_BASE_MAX = 1.5
SURGE_RUSH_MIN = 1.5
SURGE_RUSH_MAX = 3.0

