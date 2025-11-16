"""GenerationIR model for data generation specifications."""

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


# Use discriminated union for better type selection
Distribution = Annotated[
    Union[
        DistUniform,
        DistNormal,
        DistZipf,
        DistSeasonal,
        DistCategorical,
        DistDerived,
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
                        except:
                            dist["low"] = 0.0
                    if "high" in dist and isinstance(dist["high"], str):
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(dist["high"].replace("Z", "+00:00"))
                            dist["high"] = float(dt.timestamp())
                        except:
                            dist["high"] = 1.0
                        
        return data


class GenerationIR(BaseModel):
    """Generation specifications for all columns."""

    columns: List[ColumnGenSpec] = Field(default_factory=list)

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

