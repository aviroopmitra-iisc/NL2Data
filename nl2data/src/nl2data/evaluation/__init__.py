"""Evaluation framework for generated data.

This package provides both single-table and multi-table evaluation capabilities.
"""

# Configuration
from .config import (
    EvaluationConfig,
    MultiTableEvalConfig,
    EvalThresholds,
)

# Models
from .models import (
    # Single-table
    MetricResult,
    ColumnReport,
    TableReport,
    WorkloadReport,
    EvaluationReport,
    # Multi-table
    TableMatch,
    ColumnMatch,
    QualityScore,
    SchemaMatchResult,
    TableScore,
    RelationshipScore,
    MultiTableEvaluationReport,
)

# Evaluators
from .evaluators import evaluate_single_table, evaluate_multi_table

# Backward compatibility: export evaluate as alias
evaluate = evaluate_single_table

# Metrics (for convenience)
from .metrics import (
    schema_coverage,
    check_pk_fk,
    numeric_marginals,
    categorical_marginals,
    correlation_metrics,
    fk_coverage,
)

# Execution utilities
from .execution import (
    zipf_fit,
    chi_square_test,
    ks_test,
    wasserstein_distance_metric,
    gini_coefficient,
    top_k_share,
    run_workloads,
)

__all__ = [
    # Config
    "EvaluationConfig",
    "MultiTableEvalConfig",
    "EvalThresholds",
    # Models - Single-table
    "MetricResult",
    "ColumnReport",
    "TableReport",
    "WorkloadReport",
    "EvaluationReport",
    # Models - Multi-table
    "TableMatch",
    "ColumnMatch",
    "QualityScore",
    "SchemaMatchResult",
    "TableScore",
    "RelationshipScore",
    "MultiTableEvaluationReport",
    # Evaluators
    "evaluate",
    "evaluate_single_table",
    "evaluate_multi_table",
    # Metrics
    "schema_coverage",
    "check_pk_fk",
    "numeric_marginals",
    "categorical_marginals",
    "correlation_metrics",
    "fk_coverage",
    # Execution
    "zipf_fit",
    "chi_square_test",
    "ks_test",
    "wasserstein_distance_metric",
    "gini_coefficient",
    "top_k_share",
    "run_workloads",
]
