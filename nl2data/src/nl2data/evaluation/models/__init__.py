"""Report models for evaluation results."""

from .single_table import (
    MetricResult,
    ColumnReport,
    TableReport,
    WorkloadReport,
    EvaluationReport,
)
from .multi_table import (
    TableMatch,
    ColumnMatch,
    QualityScore,
    SchemaMatchResult,
    TableScore,
    RelationshipScore,
    MultiTableEvaluationReport,
)

__all__ = [
    # Single-table models
    "MetricResult",
    "ColumnReport",
    "TableReport",
    "WorkloadReport",
    "EvaluationReport",
    # Multi-table models
    "TableMatch",
    "ColumnMatch",
    "QualityScore",
    "SchemaMatchResult",
    "TableScore",
    "RelationshipScore",
    "MultiTableEvaluationReport",
]

