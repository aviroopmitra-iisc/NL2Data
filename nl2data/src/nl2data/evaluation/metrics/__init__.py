"""Metric computation modules."""

# Schema metrics
from .schema.coverage import schema_coverage, compute_coverage_factors
from .schema.validation import check_pk_fk

# Table metrics
from .table.marginals import numeric_marginals, categorical_marginals
from .table.correlations import correlation_metrics, mutual_information
from .table.fidelity import table_fidelity_score

# Relational metrics
from .relational.integrity import fk_coverage, fk_coverage_duckdb
from .relational.degrees import degree_histogram, degree_distribution_divergence
from .relational.joins import join_selectivity

__all__ = [
    # Schema
    "schema_coverage",
    "compute_coverage_factors",
    "check_pk_fk",
    # Table
    "numeric_marginals",
    "categorical_marginals",
    "correlation_metrics",
    "mutual_information",
    "table_fidelity_score",
    # Relational
    "fk_coverage",
    "fk_coverage_duckdb",
    "degree_histogram",
    "degree_distribution_divergence",
    "join_selectivity",
]
