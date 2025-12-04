"""Single-table evaluation report models."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result of a single metric evaluation."""

    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None


class ColumnReport(BaseModel):
    """Evaluation report for a column."""

    table: str
    column: str
    family: str  # Distribution family
    metrics: List[MetricResult] = Field(default_factory=list)


class TableReport(BaseModel):
    """Evaluation report for a table."""

    name: str
    row_count: int
    pk_ok: bool
    fk_ok: bool


class WorkloadReport(BaseModel):
    """Evaluation report for a workload query."""

    sql: str
    type: str
    elapsed_sec: float
    rows: int
    group_gini: Optional[float] = None
    top1_share: Optional[float] = None
    passed: Optional[bool] = None


class EvaluationReport(BaseModel):
    """Complete evaluation report for single-table evaluation."""

    schema: List[TableReport] = Field(default_factory=list)
    columns: List[ColumnReport] = Field(default_factory=list)
    workloads: List[WorkloadReport] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False

