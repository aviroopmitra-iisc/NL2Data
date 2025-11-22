"""GenerationIR model for data generation specifications."""

from __future__ import annotations

from typing import List, Optional, Dict, Literal, Union, Any, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator, Discriminator


class CategoricalDomain(BaseModel):
    """Domain for categorical distributions."""

    values: List[str]
    probs: Optional[List[float]] = None

    @field_validator("values", mode="before")
    @classmethod
    def convert_values_to_strings(cls, v: Any) -> List[str]:
        """Convert all values to strings (handles bool, int, float, etc.)."""
        if isinstance(v, list):
            return [str(val) for val in v]
        return v


class DistUniform(BaseModel):
    """Uniform distribution specification."""

    kind: Literal["uniform"] = "uniform"
    low: float = 0.0
    high: float = 1.0


class DistNormal(BaseModel):
    """Normal distribution specification."""

    kind: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0


class DistLognormal(BaseModel):
    """Log-normal distribution specification."""

    kind: Literal["lognormal"] = "lognormal"
    mean: float  # Mean of the underlying normal distribution
    sigma: float  # Standard deviation of the underlying normal distribution

    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, v: float) -> float:
        """Ensure sigma is positive."""
        if v <= 0:
            raise ValueError("sigma must be positive")
        return v


class DistPareto(BaseModel):
    """Pareto distribution specification."""

    kind: Literal["pareto"] = "pareto"
    alpha: float  # Shape parameter (must be > 0)
    xm: float = 1.0  # Scale parameter (minimum value, must be > 0)

    @field_validator("alpha", "xm")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure alpha and xm are positive."""
        if v <= 0:
            raise ValueError("alpha and xm must be positive")
        return v


class DistPoisson(BaseModel):
    """Poisson distribution specification."""

    kind: Literal["poisson"] = "poisson"
    lam: float  # Lambda parameter (rate, must be > 0)

    @field_validator("lam")
    @classmethod
    def validate_lam(cls, v: float) -> float:
        """Ensure lam is positive."""
        if v <= 0:
            raise ValueError("lam must be positive")
        return v


class DistExponential(BaseModel):
    """Exponential distribution specification."""

    kind: Literal["exponential"] = "exponential"
    scale: float = 1.0  # Scale parameter (1/lambda, must be > 0)

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: float) -> float:
        """Ensure scale is positive."""
        if v <= 0:
            raise ValueError("scale must be positive")
        return v


class DistMixtureComponent(BaseModel):
    """Component of a mixture distribution."""

    weight: float  # Weight of this component (must sum to ~1.0 across all components)
    distribution: Distribution  # Any distribution type (recursive)
    condition: Optional[Dict[str, Any]] = None  # Optional condition for conditional mixtures


class DistMixture(BaseModel):
    """Mixture distribution specification."""

    kind: Literal["mixture"] = "mixture"
    components: List[DistMixtureComponent]

    @model_validator(mode="after")
    def check_weights_sum_to_one(self) -> "DistMixture":
        """Ensure weights sum to approximately 1.0."""
        total = sum(c.weight for c in self.components)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self


class DistZipf(BaseModel):
    """Zipf distribution specification."""

    kind: Literal["zipf"] = "zipf"
    s: float = 1.2  # Zipf exponent
    n: Optional[int] = None  # Domain size


class DistSeasonal(BaseModel):
    """Seasonal distribution specification."""

    kind: Literal["seasonal"] = "seasonal"
    granularity: Literal["month", "week", "hour"] = "month"
    weights: Dict[str, float]  # e.g., {"January": 0.1, "February": 0.08, ...} or {"07:00-09:00": 2.5, ...}


class DistCategorical(BaseModel):
    """Categorical distribution specification."""

    kind: Literal["categorical"] = "categorical"
    domain: CategoricalDomain


class DistDerived(BaseModel):
    """Derived column specification."""

    kind: Literal["derived"] = "derived"
    expression: str  # Safe DSL expression, not freeform Python
    dtype: Optional[str] = None  # "float", "int", "datetime", "bool", etc.
    depends_on: List[str] = Field(default_factory=list)  # Filled by compile step


class WindowFrame(BaseModel):
    """Window frame specification for window functions."""

    type: Literal["ROWS", "RANGE"] = "RANGE"  # ROWS or RANGE
    preceding: str  # e.g., "7d", "24h", "100" (for ROWS)
    following: Optional[str] = None  # Optional following window (defaults to CURRENT ROW)


class DistWindow(BaseModel):
    """Window function specification for computing rolling/aggregated values."""

    kind: Literal["window"] = "window"
    expression: str  # e.g., "count(*)", "sum(amount)", "mean(latency)", or column name
    partition_by: List[str] = Field(default_factory=list)  # Columns to partition by
    order_by: str  # Column to order by (must be datetime/timestamp for RANGE windows)
    frame: WindowFrame  # Window frame specification
    dtype: Optional[str] = None  # Optional dtype hint


# Use discriminated union for better type selection
# Note: DistMixture is defined before Distribution to allow recursive reference
Distribution = Annotated[
    Union[
        DistUniform,
        DistNormal,
        DistLognormal,
        DistPareto,
        DistPoisson,
        DistExponential,
        DistMixture,
        DistZipf,
        DistSeasonal,
        DistCategorical,
        DistDerived,
        DistWindow,
    ],
    Discriminator("kind"),
]


class ProviderRef(BaseModel):
    """Reference to a value provider."""

    name: str  # e.g., "faker.email", "mimesis.full_name", "lookup.city"
    config: Dict[str, Any] = Field(default_factory=dict)  # Provider-specific config


class ColumnGenSpec(BaseModel):
    """Generation specification for a column."""

    table: str
    column: str
    distribution: Optional[Distribution] = None  # Optional: can use provider instead
    provider: Optional[ProviderRef] = None  # Explicit provider hint

    @model_validator(mode="before")
    @classmethod
    def fix_distribution(cls, data: Any) -> Any:
        """Pre-process distribution data to fix common LLM errors."""
        if isinstance(data, dict) and "distribution" in data:
            dist = data["distribution"]
            if isinstance(dist, dict) and "kind" in dist:
                kind = dist["kind"]
                
                # Fix categorical: convert values to strings
                if kind == "categorical" and "domain" in dist:
                    domain = dist["domain"]
                    if isinstance(domain, dict) and "values" in domain:
                        domain["values"] = [str(v) for v in domain["values"]]
                
                # Fix uniform: ensure low/high are numbers (not dates)
                if kind == "uniform":
                    if "low" in dist and isinstance(dist["low"], str):
                        # If it's a date string, convert to numeric
                        # Try to parse as date and convert to timestamp, or default to 0.0
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(dist["low"].replace("Z", "+00:00"))
                            dist["low"] = float(dt.timestamp())
                        except (ValueError, AttributeError, TypeError):
                            dist["low"] = 0.0
                    if "high" in dist and isinstance(dist["high"], str):
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(dist["high"].replace("Z", "+00:00"))
                            dist["high"] = float(dt.timestamp())
                        except (ValueError, AttributeError, TypeError):
                            dist["high"] = 1.0
                        
        return data


class EventEffect(BaseModel):
    """Effect of an event on data generation."""
    
    table: str  # Table to affect
    column: Optional[str] = None  # Optional: specific column, or None for table-wide
    effect_type: Literal["multiply_distribution", "add_offset", "set_value", "change_distribution"] = "multiply_distribution"
    value: Union[float, str, Dict[str, Any]]  # Effect value (multiplier, offset, new value, or new distribution)


class EventSpec(BaseModel):
    """Specification for a global event that affects data generation."""
    
    name: str  # Event name (for debugging/logging)
    start_time: str  # ISO datetime string or relative time (e.g., "2024-01-15T00:00:00Z" or "50%")
    end_time: Optional[str] = None  # Optional end time (for duration events)
    effects: List[EventEffect] = Field(default_factory=list)  # Effects to apply during event


class GenerationIR(BaseModel):
    """Generation specifications for all columns."""

    columns: List[ColumnGenSpec] = Field(default_factory=list)
    events: List[EventSpec] = Field(default_factory=list)  # Global events affecting generation

    @model_validator(mode="before")
    @classmethod
    def fix_columns(cls, data: Any) -> Any:
        """Pre-process columns to fix common LLM errors."""
        if isinstance(data, dict) and "columns" in data:
            columns = data["columns"]
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict) and "distribution" in col:
                        dist = col["distribution"]
                        if isinstance(dist, dict) and "kind" in dist:
                            kind = dist["kind"]
                            
                            # Fix categorical: convert values to strings
                            if kind == "categorical" and "domain" in dist:
                                domain = dist["domain"]
                                if isinstance(domain, dict) and "values" in domain:
                                    domain["values"] = [str(v) for v in domain["values"]]
                            
                            # Fix uniform: ensure low/high are numbers
                            if kind == "uniform":
                                if "low" in dist and isinstance(dist["low"], str):
                                    dist["low"] = 0.0
                                if "high" in dist and isinstance(dist["high"], str):
                                    dist["high"] = 1.0
        return data

