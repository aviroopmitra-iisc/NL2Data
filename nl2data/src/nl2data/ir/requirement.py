"""RequirementIR model for capturing natural language requirements."""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


class ScaleHint(BaseModel):
    """Hint about the scale of a table."""

    table: Optional[str] = None
    row_count: Optional[int] = None


class DistributionHint(BaseModel):
    """Hint about the distribution for a column."""

    target: str  # e.g., "fact_sales.product_id"
    family: str  # "zipf" | "seasonal" | "categorical" | "numeric"
    params: Dict[str, Any] = Field(default_factory=dict)  # Allow any type for flexibility


class RequirementIR(BaseModel):
    """Structured representation of natural language requirements."""

    domain: Optional[str] = None
    narrative: str
    tables_hint: Optional[str] = None
    scale: List[ScaleHint] = Field(default_factory=list)
    distributions: List[DistributionHint] = Field(default_factory=list)
    nonfunctional_goals: List[str] = Field(default_factory=list)
    schema_mode: Literal["oltp", "star", "snowflake"] = "star"

