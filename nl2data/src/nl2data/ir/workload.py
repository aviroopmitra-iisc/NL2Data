"""WorkloadIR model for workload specifications."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class WorkloadSpec(BaseModel):
    """Specification for a workload query."""

    type: Literal["group_by", "join", "filter"]
    query_hint: Optional[str] = None
    expected_skew: Optional[Literal["low", "medium", "high"]] = None
    join_graph: Optional[List[str]] = None  # List of table names
    selectivity_hint: Optional[Literal["low", "medium", "high"]] = None


class WorkloadIR(BaseModel):
    """Workload specifications for evaluation."""

    targets: List[WorkloadSpec] = Field(default_factory=list)

