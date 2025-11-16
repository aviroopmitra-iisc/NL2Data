"""Evaluation framework for generated data."""

from .config import EvaluationConfig, EvalThresholds
from .report_models import EvaluationReport, MetricResult, ColumnReport, TableReport, WorkloadReport
from .report_builder import evaluate

__all__ = [
    "EvaluationConfig",
    "EvalThresholds",
    "EvaluationReport",
    "MetricResult",
    "ColumnReport",
    "TableReport",
    "WorkloadReport",
    "evaluate",
]

